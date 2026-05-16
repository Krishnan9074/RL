# Adaptive k_obs Control via Reinforcement Learning
## NeuroSim XPBD Surgical Tissue Tracking — BU.520.750 Group 23

---

## Overview

This project frames the observation spring gain (`k_obs`) selection in an
XPBD-based surgical tissue simulator as a sequential decision-making problem
and trains tabular Q-Learning and SARSA agents to adapt the gain online.

**Key result:** Adaptive RL reduces mean Chamfer tracking error by ~32%
under camera occlusion and ~27% under tool contact vs the best fixed-gain
baseline.

---

## Project Structure

```
skin_rl_project/
├── config.py              All hyperparameters in one place
├── main.py                Entry point — train + evaluate + plot
├── requirements.txt
│
├── simulator/
│   └── xpbd.py            Pure-PyTorch 3-layer XPBD tissue simulator
│
├── env/
│   ├── environment.py     Gymnasium-style RL environment
│   └── episode_types.py   Disturbance schedules (Normal/Occlusion/Contact)
│
├── agents/
│   ├── base.py            Abstract base agent (epsilon-greedy, save/load)
│   ├── qlearning.py       Off-policy Q-Learning
│   ├── sarsa.py           On-policy SARSA
│   └── baselines.py       Fixed-gain and random baselines
│
├── training/
│   └── trainer.py         Training loop + checkpointing
│
├── evaluation/
│   └── evaluator.py       Held-out evaluation across all conditions
│
├── visualization/
│   └── plots.py           5 publication-quality figures
│
└── results/               Auto-created — all outputs land here
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python main.py
```

### 3. Quick smoke test (200 episodes, ~5 min on CPU)

```bash
python main.py --episodes 200
```

### 4. Force CPU (if no GPU available)

```bash
python main.py --device cpu --episodes 200
```

### 5. Evaluate saved agents (skip re-training)

```bash
python main.py --eval-only
```

---

## MDP Formulation

| Component | Definition |
|---|---|
| **State** | `[d_CD_mm, mean_speed, Δp_cloud_mm, k_obs(t-1)]` — discretised to 625 states |
| **Action** | `{0: decrease, 1: hold, 2: increase}` k_obs by 5% on log scale |
| **Reward** | `-d_CD × 1000 - 0.001 × |Δk_obs|` |
| **Episode** | 300 frames (5 s at 60 Hz) |
| **Discount γ** | 0.95 |

### Episode Types

- **Normal** — Gaussian noise only. Fixed k_obs near-optimal. Baseline validation.
- **Occlusion** — Frames 100–130: camera corrupted. Optimal: decrease k_obs.
- **Contact** — Frame 150: tool impulse. Optimal: spike k_obs then recover.

---

## Output Files

All outputs are written to `results/`:

| File | Description |
|---|---|
| `Q-Learning_final.pkl` | Trained Q-table |
| `SARSA_final.pkl` | Trained Q-table |
| `fig1_learning_curves.*` | Training convergence (Chamfer + reward) |
| `fig2_eval_bar.*` | Baseline comparison bar chart |
| `fig3_kobs_traj.*` | Adaptive gain trajectories per episode type |
| `fig4_qtable_heatmap.*` | Greedy policy visualisation |
| `fig5_epsilon_decay.*` | Exploration schedule |

---

## Configuration

All hyperparameters are in `config.py`. Key parameters:

```python
N_TRAIN       = 1000     # training episodes
ALPHA         = 0.3      # learning rate
GAMMA         = 0.95     # discount factor
EPS_DECAY_EPS = 500      # episodes for epsilon to anneal 1.0 → 0.05
NOISE_SIGMA   = 0.001    # m — depth sensor noise (1.0 mm)
BREATHING_AMP = 0.004    # m — ground truth oscillation amplitude (4 mm)
```

---

## Citation

> Krishnan, V.S. et al. (2026). *Adaptive Observation Gain Control for
> Real-Time Surgical Tissue Tracking via Reinforcement Learning.*
> BU.520.750, Johns Hopkins Carey Business School.

Physics reference: Macklin, Müller, Chentanez (2016). XPBD. Proc. MIG.
Biomechanical parameters: Li et al. (2022). Biomaterials 287, 121381.
