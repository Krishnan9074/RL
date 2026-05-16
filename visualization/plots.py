# =============================================================================
# visualization/plots.py
#
# All figure generation for the project.
# Produces five publication-quality figures saved as PDF + PNG.
#
# Figures:
#   1. Learning curves (Chamfer distance + cumulative reward)
#   2. Evaluation bar chart (6 agents × 3 episode types)
#   3. k_obs trajectory per episode type
#   4. Q-table policy heatmap
#   5. Epsilon decay schedule
# =============================================================================

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, EPISODE_LEN, EP_NAMES,
    EP_NORMAL, EP_OCCLUSION, EP_CONTACT,
    K_OBS_MIN, K_OBS_MAX, N_STATE_BINS,
    STATE_LOW, STATE_HIGH,
)

# ── Consistent colour palette ─────────────────────────────────────────────
COLORS = {
    "Q-Learning":     "#1565C0",
    "SARSA":          "#C62828",
    "Fixed 48 N/m":   "#2E7D32",
    "Fixed 80 N/m":   "#F57F17",
    "Fixed 10 N/m":   "#6A1B9A",
    "Random Policy":  "#00695C",
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})


def _smooth(x: List[float], w: int = 25) -> np.ndarray:
    arr    = np.asarray(x, dtype=float)
    padded = np.pad(arr, w // 2, mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")[: len(arr)]


def _save(fig, name: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


class Plotter:
    """
    Generates all project figures from training history and eval results.

    Usage
    -----
    p = Plotter(outdir="results")
    p.learning_curves(ql_history, sarsa_history)
    p.eval_bar(results)
    p.kobs_trajectories(env, ql_agent, sarsa_agent)
    p.qtable_heatmap(ql_agent)
    p.epsilon_decay(ql_history, sarsa_history)
    """

    def __init__(self, outdir: str = RESULTS_DIR):
        self.outdir = outdir

    # ── Figure 1: Learning curves ─────────────────────────────────────────

    def learning_curves(
        self,
        ql_history:    Dict,
        sarsa_history: Dict,
    ):
        """Training Chamfer distance and cumulative reward vs episode."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.15,
                            left=0.09, right=0.97)

        pairs = [
            ("train_cd",     "eval_cd",
             "Mean Chamfer Distance (mm)", "Training Chamfer Distance"),
            ("train_reward", None,
             "Cumulative Reward",           "Training Cumulative Reward"),
        ]

        for ax, (train_key, eval_key, ylabel, title) in zip(axes, pairs):
            for name, hist in [("Q-Learning", ql_history),
                                ("SARSA",      sarsa_history)]:
                col = COLORS[name]
                raw = hist[train_key]
                ax.plot(raw, alpha=0.15, color=col, lw=0.7)
                ax.plot(_smooth(raw), color=col, lw=2.2, label=name)

                if eval_key and hist.get("eval_episodes"):
                    ax.scatter(hist["eval_episodes"], hist[eval_key],
                               marker="D", s=24, color=col, zorder=5,
                               edgecolors="white", linewidths=0.8)

            ax.set_xlabel("Training Episode")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.2)

        axes[0].text(0.98, 0.97, "◆ greedy eval",
                     transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=7.5,
                     style="italic", color="#555")

        fig.suptitle("Q-Learning vs SARSA — Training Convergence",
                     fontsize=12, fontweight="bold", y=0.99)
        _save(fig, "fig1_learning_curves", self.outdir)

    # ── Figure 2: Evaluation bar chart ────────────────────────────────────

    def eval_bar(self, results: Dict):
        """Grouped bar chart: mean Chamfer by agent and episode type."""
        ep_labels   = list(EP_NAMES.values())
        agent_names = list(results.keys())
        palette     = ["#1565C0", "#C62828", "#2E7D32",
                       "#F57F17", "#6A1B9A", "#00695C"]
        x     = np.arange(len(ep_labels))
        width = 0.8 / len(agent_names)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        fig.subplots_adjust(top=0.87, bottom=0.14, left=0.08, right=0.97)

        for i, name in enumerate(agent_names):
            means = [results[name][ep]["mean"] for ep in ep_labels]
            stds  = [results[name][ep]["std"]  for ep in ep_labels]
            off   = (i - len(agent_names) / 2 + 0.5) * width
            ax.bar(x + off, means, width * 0.92,
                   label=name, color=palette[i % len(palette)],
                   alpha=0.90, zorder=3, edgecolor="white", lw=0.6)
            ax.errorbar(x + off, means, yerr=stds,
                        fmt="none", color="#111",
                        capsize=2.5, lw=1.0, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(ep_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Chamfer Distance (mm)")
        ax.set_title(
            "Evaluation: Mean Chamfer Distance by Episode Type\n"
            "Lower is better · 30 test episodes per condition",
            fontweight="bold", fontsize=10,
        )
        ax.legend(fontsize=7.5, ncol=3, loc="upper left")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        _save(fig, "fig2_eval_bar", self.outdir)

    # ── Figure 3: k_obs trajectories ──────────────────────────────────────

    def kobs_trajectories(self, env, ql_agent, sarsa_agent,
                          fixed_baseline=None):
        """k_obs over one episode for each episode type."""
        from training.trainer import run_episode

        ep_codes  = [EP_NORMAL, EP_OCCLUSION, EP_CONTACT]
        ep_labels = [EP_NAMES[c] for c in ep_codes]
        frames    = np.arange(EPISODE_LEN)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        fig.subplots_adjust(wspace=0.08, top=0.85,
                            bottom=0.17, left=0.08, right=0.97)

        for ci, (code, label) in enumerate(zip(ep_codes, ep_labels)):
            ax = axes[ci]

            ql_r  = run_episode(env, ql_agent,    ep_type=code, train=False)
            sa_r  = run_episode(env, sarsa_agent, ep_type=code, train=False)

            # Fixed-48 reference line
            ax.axhline(48, color="#aaa", lw=0.9, ls=":", zorder=1)

            if fixed_baseline is not None:
                fx_r = run_episode(env, fixed_baseline, ep_type=code, train=False)
                ax.plot(frames, fx_r["k_traj"], color="#999",
                        lw=1.0, ls=":", label="Fixed 48", zorder=2)

            ax.plot(frames, sa_r["k_traj"],
                    color=COLORS["SARSA"], lw=1.6, ls="--",
                    label="SARSA", alpha=0.85, zorder=3)
            ax.plot(frames, ql_r["k_traj"],
                    color=COLORS["Q-Learning"], lw=2.0,
                    label="Q-Learning", zorder=4)

            if code == EP_OCCLUSION:
                ax.axvspan(100, 130, alpha=0.12, color="orange", zorder=0)
                ax.text(115, K_OBS_MAX * 0.90, "occlusion\nwindow",
                        ha="center", fontsize=7.5, color="#E65100",
                        fontweight="bold")
            if code == EP_CONTACT:
                ax.axvline(150, color="red", lw=1.3, ls="--",
                           alpha=0.6, zorder=0)
                ax.text(153, K_OBS_MAX * 0.90, "tool\ncontact",
                        fontsize=7.5, color="#B71C1C", fontweight="bold")

            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Frame")
            ax.grid(True, alpha=0.18)
            ax.set_xlim(0, EPISODE_LEN)
            ax.set_ylim(K_OBS_MIN * 0.5, K_OBS_MAX * 1.05)

        axes[0].set_ylabel("k_obs (N/m)")
        axes[0].legend(loc="lower right", frameon=True, fontsize=7.5)

        fig.suptitle("Learned k_obs Trajectory — Adaptive Gain Control",
                     fontsize=11, fontweight="bold", y=0.99)
        _save(fig, "fig3_kobs_traj", self.outdir)

    # ── Figure 4: Q-table heatmap ─────────────────────────────────────────

    def qtable_heatmap(self, agent):
        """
        Greedy policy on (d_CD × Δp_cloud) plane.
        Marginalises over mean_speed (dim 1) and k_obs_prev (dim 3)
        by taking max over those bins.
        """
        best_action = np.argmax(agent.Q, axis=1)
        policy      = best_action.reshape([N_STATE_BINS] * 4)
        policy_2d   = policy.max(axis=3).max(axis=1)

        cd_labels    = [f"{v:.0f}" for v in np.linspace(
            STATE_LOW[0], STATE_HIGH[0], N_STATE_BINS)]
        cloud_labels = [f"{v:.1f}" for v in np.linspace(
            STATE_LOW[2], STATE_HIGH[2], N_STATE_BINS)]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.16, right=0.92, top=0.86, bottom=0.14)

        cmap = matplotlib.colormaps.get_cmap("RdYlGn_r").resampled(3)
        im   = ax.imshow(policy_2d, cmap=cmap, vmin=-0.5, vmax=2.5,
                         aspect="auto", origin="lower")

        action_labels = {0: "↓ Decrease", 1: "= Hold", 2: "↑ Increase"}
        for i in range(N_STATE_BINS):
            for j in range(N_STATE_BINS):
                ax.text(j, i, action_labels[policy_2d[i, j]],
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if policy_2d[i, j] != 1 else "#333")

        ax.set_xticks(range(N_STATE_BINS))
        ax.set_xticklabels(cloud_labels)
        ax.set_yticks(range(N_STATE_BINS))
        ax.set_yticklabels(cd_labels)
        ax.set_xlabel("Δp_cloud (mm)  →  point-cloud frame-to-frame shift")
        ax.set_ylabel("Chamfer Distance (mm)  →  current tracking error")
        ax.set_title(
            "Greedy Policy — Q-Learning\n"
            "(max over speed × k_prev marginal dimensions)",
            fontsize=10, fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.75, pad=0.03)
        cbar.ax.set_yticklabels(["Decrease", "Hold", "Increase"])
        _save(fig, "fig4_qtable_heatmap", self.outdir)

    # ── Figure 5: Epsilon decay ───────────────────────────────────────────

    def epsilon_decay(self, ql_history: Dict, sarsa_history: Dict):
        """Exploration rate over training episodes."""
        fig, ax = plt.subplots(figsize=(8, 3.6))
        fig.subplots_adjust(top=0.87, bottom=0.17, left=0.10, right=0.97)

        ax.plot(ql_history["epsilon"],
                color=COLORS["Q-Learning"], lw=2.2, label="Q-Learning")
        ax.plot(sarsa_history["epsilon"],
                color=COLORS["SARSA"], lw=2.2, ls="--", label="SARSA")

        decay_ep = len([e for e in ql_history["epsilon"] if e > 0.05 + 1e-6])
        ax.axvline(decay_ep, color="#888", ls="--", lw=1.0)
        ax.axhline(0.05, color="#888", ls=":", lw=0.9)
        ax.text(decay_ep + 10, 0.55,
                f"ε = 0.05\n(ep {decay_ep})",
                fontsize=8, color="#555")

        ax.set_xlabel("Training Episode")
        ax.set_ylabel("ε  (exploration rate)")
        ax.set_title("ε-Greedy Exploration Schedule", fontweight="bold")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.22)
        ax.set_xlim(0, len(ql_history["epsilon"]))
        ax.set_ylim(0, 1.05)
        _save(fig, "fig5_epsilon_decay", self.outdir)
