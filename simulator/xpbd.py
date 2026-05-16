# =============================================================================
# simulator/xpbd.py
#
# Pure-PyTorch XPBD three-layer skin tissue simulator.
# No Newton, no Warp — runs anywhere PyTorch is installed.
#
# Physics reference:
#   Macklin, Müller, Chentanez (2016). XPBD: Position-based simulation
#   of compliant constrained dynamics. Proc. MIG.
#
# Biomechanical parameters:
#   Li et al. (2022). In vivo stiffness measurement of epidermis, dermis,
#   and hypodermis using broadband Rayleigh-wave OCE. Biomaterials 287.
# =============================================================================

import time
import numpy as np
import torch


def build_skin_mesh(rows: int, cols: int, spacing: float):
    """
    Build a three-layer particle mesh representing skin tissue.

    Layers stacked vertically (Y-axis):
        Y = 0           Hypodermis (softest, deepest)
        Y = spacing     Dermis     (primary load-bearing)
        Y = 2*spacing   Epidermis  (stiffest, camera-visible surface)

    Returns
    -------
    pos    : (N, 3) float32   rest positions in metres
    edges  : (E, 2) int32     spring endpoint indices
    lids   : (N,)   int32     layer index per particle
    ke_arr : (E,)   float32   spring stiffness N/m
    kd_arr : (E,)   float32   spring damping coefficient
    """
    from config import (LAYER_HYPO, LAYER_DERM, LAYER_EPI,
                        LAYER_KE, LAYER_KD, INTER_KE)

    N_per = rows * cols
    N     = 3 * N_per
    pos   = np.zeros((N, 3), dtype=np.float32)
    lids  = np.zeros(N,      dtype=np.int32)

    layer_y = {
        LAYER_HYPO: 0.0,
        LAYER_DERM: spacing,
        LAYER_EPI:  2.0 * spacing,
    }

    for lyr in (LAYER_HYPO, LAYER_DERM, LAYER_EPI):
        base = lyr * N_per
        y    = layer_y[lyr]
        for r in range(rows):
            for c in range(cols):
                i       = base + r * cols + c
                pos[i]  = [(c - cols / 2.0) * spacing,
                            y,
                            (r - rows / 2.0) * spacing]
                lids[i] = lyr

    edges_list, ke_list, kd_list = [], [], []

    def add(i, j, ke, kd):
        edges_list.append([i, j])
        ke_list.append(ke)
        kd_list.append(kd)

    # Intra-layer springs: axis-aligned + 2 diagonals (shear resistance)
    for lyr in (LAYER_HYPO, LAYER_DERM, LAYER_EPI):
        base = lyr * N_per
        ke, kd = LAYER_KE[lyr], LAYER_KD[lyr]
        for r in range(rows):
            for c in range(cols):
                i = base + r * cols + c
                if c + 1 < cols:
                    add(i, base + r * cols + c + 1,       ke, kd)
                if r + 1 < rows:
                    add(i, base + (r + 1) * cols + c,     ke, kd)
                if r + 1 < rows and c + 1 < cols:
                    add(i, base + (r + 1) * cols + c + 1, ke, kd)
                if r + 1 < rows and c - 1 >= 0:
                    add(i, base + (r + 1) * cols + c - 1, ke, kd)

    # Inter-layer vertical springs (basement membrane, fascial attachment)
    for (lo, hi) in ((LAYER_HYPO, LAYER_DERM), (LAYER_DERM, LAYER_EPI)):
        ke = INTER_KE[(lo, hi)]
        for r in range(rows):
            for c in range(cols):
                add(lo * N_per + r * cols + c,
                    hi * N_per + r * cols + c, ke, 0.08)

    edges  = np.array(edges_list, dtype=np.int32)
    ke_arr = np.array(ke_list,    dtype=np.float32)
    kd_arr = np.array(kd_list,    dtype=np.float32)

    print(f"[SkinMesh] {rows}×{cols} × 3 layers = {N} particles, "
          f"{len(edges)} springs")
    return pos, edges, lids, ke_arr, kd_arr


class SkinXPBD_Pure:
    """
    GPU-accelerated XPBD tissue simulator using pure PyTorch.

    Design principles
    -----------------
    1. gravity = 0 (horizontal phantom — no body forces)
       With no external force, all spring constraints satisfied at rest
       → zero corrections → zero motion. Mathematically guaranteed.

    2. Kinematic tool constraint (direct position override):
       apply_tool() forces exact deformation at specified depth.

    3. Jacobi XPBD with degree normalisation:
       Corrections divided by particle spring-degree to prevent
       overcorrection at high-valence nodes.

    Stability: δt / δt_crit < 0.025 for all layers at ns=10 substeps.
    """

    def __init__(self, rows, cols, spacing, substeps, n_iter,
                 gravity_y, damping, dt, device):
        from config import LAYER_EPI

        self.dt        = dt
        self.substeps  = substeps
        self.n_iter    = n_iter
        self.gravity_y = gravity_y
        self.damping   = damping
        self._dev      = device

        pos, edges, lids, ke_arr, kd_arr = build_skin_mesh(rows, cols, spacing)
        N = len(pos)
        E = len(edges)

        self.layer_ids    = lids
        self.surface_mask = (lids == LAYER_EPI)
        self.rest_np      = pos.copy()

        # GPU state tensors
        self.pos      = torch.tensor(pos, dtype=torch.float32, device=device)
        self.vel      = torch.zeros(N, 3, dtype=torch.float32, device=device)
        self.inv_mass = torch.ones(N,    dtype=torch.float32, device=device)

        # Spring tensors
        self.ei       = torch.tensor(edges[:, 0], dtype=torch.long,  device=device)
        self.ej       = torch.tensor(edges[:, 1], dtype=torch.long,  device=device)
        pi            = pos[edges[:, 0]]
        pj            = pos[edges[:, 1]]
        self.rest_len = torch.tensor(
            np.linalg.norm(pi - pj, axis=1), dtype=torch.float32, device=device)
        self.alpha    = 1.0 / torch.tensor(ke_arr, dtype=torch.float32, device=device)

        # Jacobi normalisation: degree = number of springs per particle
        deg = torch.zeros(N, dtype=torch.float32, device=device)
        deg.scatter_add_(0, self.ei, torch.ones(E, dtype=torch.float32, device=device))
        deg.scatter_add_(0, self.ej, torch.ones(E, dtype=torch.float32, device=device))
        self.degree = deg.clamp(min=1.0)

        print(f"[SkinXPBD_Pure] N={N}, E={E}, "
              f"surface={int(self.surface_mask.sum())}, device={device}")

    def apply_tool(self, depth_m: float, surf_idx: np.ndarray,
                   tool_weights: np.ndarray):
        """
        One-sided kinematic constraint: push surface particles downward.
        Only moves particles above the target — never pulls upward.
        """
        if depth_m < 1e-8:
            return
        rest_y  = torch.tensor(self.rest_np[surf_idx, 1],
                               dtype=torch.float32, device=self._dev)
        weights = torch.tensor(tool_weights,
                               dtype=torch.float32, device=self._dev)
        target  = rest_y - depth_m * weights
        cur_y   = self.pos[surf_idx, 1]
        push    = cur_y > target
        if push.any():
            idx = torch.tensor(surf_idx, dtype=torch.long,
                               device=self._dev)[push]
            self.pos[idx, 1] = target[push]

    def step(self):
        """
        Advance one frame (dt = 1/60 s).

        Returns
        -------
        pos    : (N, 3) tensor — particle positions after step
        stress : (N,)   tensor — velocity magnitude (stress proxy)
        ms     : float  — wall-clock time in milliseconds
        """
        t0     = time.perf_counter()
        sub_dt = self.dt / self.substeps
        alpha_t = self.alpha / (sub_dt ** 2)

        for _ in range(self.substeps):

            # 1. Predict positions
            if abs(self.gravity_y) > 1e-9:
                self.vel[:, 1] += sub_dt * self.gravity_y
            pos_pred = self.pos + sub_dt * self.vel

            # 2. XPBD constraint projection (Jacobi)
            lam = torch.zeros(len(self.ei), dtype=torch.float32, device=self._dev)
            for _ in range(self.n_iter):
                pi    = pos_pred[self.ei]
                pj    = pos_pred[self.ej]
                diff  = pi - pj
                dist  = torch.norm(diff, dim=1).clamp(min=1e-9)
                n_hat = diff / dist.unsqueeze(1)
                C     = dist - self.rest_len
                wi    = self.inv_mass[self.ei]
                wj    = self.inv_mass[self.ej]
                dlam  = -(C + alpha_t * lam) / (wi + wj + alpha_t)
                lam   = lam + dlam
                ci    = (wi * dlam / self.degree[self.ei]).unsqueeze(1) * n_hat
                cj    = (wj * dlam / self.degree[self.ej]).unsqueeze(1) * n_hat
                dx    = torch.zeros_like(pos_pred)
                dx.scatter_add_(0, self.ei.unsqueeze(1).expand_as(ci),  ci)
                dx.scatter_add_(0, self.ej.unsqueeze(1).expand_as(cj), -cj)
                pos_pred = pos_pred + dx

            # 3. Velocity update and damping
            self.vel = (pos_pred - self.pos) / sub_dt
            self.vel = self.vel * self.damping
            self.pos = pos_pred

        stress = torch.norm(self.vel, dim=1)
        ms     = (time.perf_counter() - t0) * 1000.0
        return self.pos.clone(), stress.clone(), ms

    @property
    def n_particles(self):
        return len(self.rest_np)

    @property
    def n_surface(self):
        return int(self.surface_mask.sum())
