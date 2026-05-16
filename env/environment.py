# =============================================================================
# env/environment.py
#
# Gymnasium-style environment wrapping SkinXPBD_Pure.
#
# Observation (4-dim continuous):
#   [chamfer_dist_mm, mean_particle_speed, delta_cloud_mm, k_obs_prev]
#
# Action (discrete, 3):
#   0 = decrease k_obs by 5%
#   1 = hold     k_obs
#   2 = increase k_obs by 5%
#
# Reward:
#   r_t = -d_CD * 1000 - lambda * |delta_k_obs|
#
# The environment uses a moving sinusoidal ground truth (breathing motion)
# so that k_obs genuinely affects tracking error — without this, any gain
# value produces identical Chamfer distance and RL has nothing to learn.
# =============================================================================

import numpy as np
import torch
from scipy.spatial import KDTree
from typing import Optional, Tuple

from config import (
    DEVICE, K_OBS_MIN, K_OBS_MAX, K_OBS_INIT, K_OBS_STEP,
    LAMBDA_REG, NOISE_SIGMA, EPISODE_LEN, OBS_DELTA_MAX,
    BREATHING_AMP, BREATHING_FREQ, TOOL_SIGMA,
    N_STATE_BINS, STATE_LOW, STATE_HIGH, N_STATES,
    EP_NORMAL, EP_OCCLUSION, EP_CONTACT,
    SIM_ROWS, SIM_COLS, SIM_SPACING, SIM_SUBSTEPS, SIM_ITER,
    SIM_DT, SIM_DAMPING, SIM_GRAVITY,
)
from simulator.xpbd import SkinXPBD_Pure
from env.episode_types import apply_disturbance


def chamfer_distance(pred: np.ndarray, obs: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two point clouds.

    Eq. (19) from the NeuroSim paper:
        d_CD = 0.5 * (mean_p2o + mean_o2p)

    Returns distance in metres.
    """
    if len(pred) == 0 or len(obs) == 0:
        return 0.0
    d_p2o, _ = KDTree(obs).query(pred)
    d_o2p, _ = KDTree(pred).query(obs)
    return 0.5 * (d_p2o.mean() + d_o2p.mean())


def discretise(obs: np.ndarray) -> int:
    """
    Map continuous 4-dim observation to integer state index in [0, 625).
    Uses uniform binning with boundary clipping.
    Row-major encoding: s = b0*5^3 + b1*5^2 + b2*5 + b3
    """
    low  = np.array(STATE_LOW,  dtype=np.float32)
    high = np.array(STATE_HIGH, dtype=np.float32)
    clipped = np.clip(obs, low, high)
    normed  = (clipped - low) / (high - low + 1e-9)
    bins    = (normed * N_STATE_BINS).astype(int).clip(0, N_STATE_BINS - 1)
    idx = 0
    for b in bins:
        idx = idx * N_STATE_BINS + int(b)
    return idx


class SkinTrackingEnv:
    """
    Tissue-tracking RL environment.

    Usage
    -----
    env = SkinTrackingEnv()
    obs = env.reset(episode_type=EP_OCCLUSION)
    obs, reward, done, info = env.step(action=2)
    """

    def __init__(
        self,
        device:       str          = DEVICE,
        episode_type: Optional[int] = None,
        seed:         int           = 42,
    ):
        self.rng             = np.random.default_rng(seed)
        self._ep_type_fixed  = episode_type
        self.device          = device

        # Build simulator (done once; reset() restores state)
        print("[Env] Building simulator ...")
        self.sim = SkinXPBD_Pure(
            rows=SIM_ROWS, cols=SIM_COLS, spacing=SIM_SPACING,
            substeps=SIM_SUBSTEPS, n_iter=SIM_ITER,
            gravity_y=SIM_GRAVITY, damping=SIM_DAMPING,
            dt=SIM_DT, device=device,
        )

        # Cache rest positions and surface indices
        self._rest_pos = self.sim.rest_np.copy()
        self._surf_idx = np.where(self.sim.surface_mask)[0]

        # Precompute tool footprint weights (fixed geometry)
        surf_rest = self._rest_pos[self._surf_idx]
        xz_dist   = np.sqrt(surf_rest[:, 0] ** 2 + surf_rest[:, 2] ** 2)
        self._tool_weights = np.exp(-0.5 * (xz_dist / TOOL_SIGMA) ** 2)

        # Episode state (initialised in reset)
        self.k_obs:       float = K_OBS_INIT
        self.step_num:    int   = 0
        self.ep_type:     int   = EP_NORMAL
        self._prev_cloud: Optional[np.ndarray] = None

        print(f"[Env] Ready. Surface particles: {len(self._surf_idx)}")

    # ── Ground truth ──────────────────────────────────────────────────────

    def _ground_truth_surface(self, t_step: int) -> np.ndarray:
        """
        Surface positions at frame t_step with sinusoidal breathing motion.
        Y = rest_Y + A * sin(2π * f * t)

        This moving target is what forces the agent to actively track.
        Without it, any k_obs value gives the same Chamfer distance.
        """
        gt = self._rest_pos[self._surf_idx].copy()
        t  = t_step * SIM_DT
        gt[:, 1] += BREATHING_AMP * np.sin(2 * np.pi * BREATHING_FREQ * t)
        return gt

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self, episode_type: Optional[int] = None) -> np.ndarray:
        """
        Restore simulator to rest state. Returns initial observation.
        """
        # Hard-reset particle state
        rest_t = torch.tensor(self._rest_pos, dtype=torch.float32,
                              device=self.device)
        self.sim.pos = rest_t.clone()
        self.sim.vel = torch.zeros_like(self.sim.vel)

        # Choose episode type
        if episode_type is not None:
            self.ep_type = episode_type
        elif self._ep_type_fixed is not None:
            self.ep_type = self._ep_type_fixed
        else:
            self.ep_type = int(self.rng.integers(0, 3))

        self.k_obs        = K_OBS_INIT
        self.step_num     = 0
        self._prev_cloud  = None

        # Return initial observation
        dummy_cloud = self._ground_truth_surface(0)
        obs, _, _   = self._compute_obs(dummy_cloud)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance one 60 Hz physics frame.

        Parameters
        ----------
        action : int in {0, 1, 2}

        Returns
        -------
        obs    : (4,) float32 observation vector
        reward : float
        done   : bool (True when episode_len reached)
        info   : dict with diagnostic fields
        """
        assert 0 <= action < 3, f"Invalid action {action}"

        prev_k = self.k_obs
        if action == 0:
            self.k_obs = max(K_OBS_MIN, self.k_obs * (1.0 - K_OBS_STEP))
        elif action == 2:
            self.k_obs = min(K_OBS_MAX, self.k_obs * (1.0 + K_OBS_STEP))
        dk = abs(self.k_obs - prev_k)

        # Build noisy point cloud from moving ground truth
        gt    = self._ground_truth_surface(self.step_num)
        noise = self.rng.normal(0, NOISE_SIGMA, gt.shape).astype(np.float32)
        cloud = gt + noise

        # Apply episode-specific disturbance
        cloud = apply_disturbance(
            cloud, self.ep_type, self.step_num,
            self._tool_weights, self.rng,
        )

        # Inject observation springs into simulator
        self._inject_obs_springs(cloud)

        # Step physics
        pos, stress, _ = self.sim.step()

        # Compute observation and Chamfer distance
        obs, d_cd, delta_cloud = self._compute_obs(cloud)

        # Reward: minimise tracking error, penalise gain instability
        reward = -d_cd * 1000.0 - LAMBDA_REG * dk

        self.step_num    += 1
        self._prev_cloud  = cloud.copy()
        done = (self.step_num >= EPISODE_LEN)

        info = {
            "d_cd_mm":    d_cd * 1000.0,
            "k_obs":      self.k_obs,
            "ep_type":    self.ep_type,
            "step":       self.step_num,
            "mean_speed": float(stress.mean().item()),
        }
        return obs, reward, done, info

    # ── Private helpers ───────────────────────────────────────────────────

    def _inject_obs_springs(self, cloud: np.ndarray):
        """
        Pull epidermis particles toward nearest point-cloud point.
        Implements Eq. (13) from the NeuroSim paper.
        Force = k_obs * clamp(delta, delta_max)
        Applied as a velocity impulse: dv = F * dt
        """
        surf_pos = self.sim.pos[self._surf_idx].cpu().numpy()
        valid    = cloud[np.any(cloud != 0, axis=1)]
        if len(valid) == 0:
            return

        _, nn    = KDTree(valid).query(surf_pos)
        delta    = valid[nn] - surf_pos
        mags     = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-9)
        scale    = np.minimum(1.0, OBS_DELTA_MAX / mags)
        f_obs    = (self.k_obs * delta * scale).astype(np.float32)

        dv = torch.tensor(f_obs * SIM_DT, dtype=torch.float32, device=self.device)
        self.sim.vel[self._surf_idx] += dv

    def _compute_obs(
        self, cloud: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute the 4-dim state observation vector.

        Returns
        -------
        obs         : [d_CD_mm, mean_speed, delta_cloud_mm, k_obs]
        d_cd        : Chamfer distance in metres
        delta_cloud : point-cloud frame-to-frame shift in mm
        """
        surf_pos   = self.sim.pos[self._surf_idx].cpu().numpy()
        mean_speed = float(torch.norm(self.sim.vel, dim=1).mean().item())

        valid = cloud[np.any(cloud != 0, axis=1)]
        d_cd  = chamfer_distance(surf_pos, valid) if len(valid) > 0 else 0.0

        if self._prev_cloud is not None:
            prev_v = self._prev_cloud[np.any(self._prev_cloud != 0, axis=1)]
            if len(prev_v) > 0 and len(valid) > 0:
                delta_cloud = float(
                    np.linalg.norm(valid.mean(0) - prev_v.mean(0))
                ) * 1000.0
            else:
                delta_cloud = 0.0
        else:
            delta_cloud = 0.0

        obs = np.array(
            [d_cd * 1000.0, mean_speed, delta_cloud, self.k_obs],
            dtype=np.float32,
        )
        return obs, d_cd, delta_cloud
