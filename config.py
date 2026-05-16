# =============================================================================
# config.py
# Central configuration for all hyperparameters and constants.
# Edit this file to change any parameter across the entire pipeline.
# =============================================================================

# ── Device ────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Simulator ─────────────────────────────────────────────────────────────
SIM_ROWS       = 30          # grid rows per layer
SIM_COLS       = 30          # grid cols per layer
SIM_SPACING    = 0.005       # particle spacing in metres (5 mm)
SIM_SUBSTEPS   = 10          # XPBD substeps per frame
SIM_ITER       = 20          # constraint iterations per substep
SIM_DT         = 1.0 / 60.0 # physics timestep (60 Hz)
SIM_DAMPING    = 0.98        # velocity damping per substep
SIM_GRAVITY    = 0.0         # m/s² — horizontal phantom, no gravity

# Layer stiffnesses from Li et al. 2022 OCE (ke = E_eff × spacing)
LAYER_KE = {0: 10.0, 1: 48.0, 2: 80.0}   # N/m  (hypo, derm, epi)
LAYER_KD = {0: 0.15, 1: 0.08, 2: 0.05}   # damping
INTER_KE = {(0, 1): 30.0, (1, 2): 60.0}  # inter-layer spring stiffness

# Layer indices
LAYER_HYPO = 0
LAYER_DERM = 1
LAYER_EPI  = 2

# ── Environment ───────────────────────────────────────────────────────────
K_OBS_MIN      = 5.0         # N/m — minimum observation spring gain
K_OBS_MAX      = 200.0       # N/m — maximum observation spring gain
K_OBS_INIT     = 48.0        # N/m — starting gain (≈ dermis stiffness)
K_OBS_STEP     = 0.05        # 5% multiplicative step per action
LAMBDA_REG     = 0.001       # gain-stability penalty weight in reward
NOISE_SIGMA    = 0.001       # m — depth sensor Gaussian noise (1.0 mm)
EPISODE_LEN    = 300         # frames per episode (5 s at 60 Hz)
OBS_DELTA_MAX  = 0.040       # m — observation spring clamp distance

# Breathing motion (moving ground truth)
BREATHING_AMP  = 0.004       # m — 4 mm sinusoidal Y-amplitude
BREATHING_FREQ = 0.6         # Hz — ~36 breaths/min

# Tool geometry for Contact episodes
TOOL_SIGMA     = 0.007       # m — Gaussian tool footprint
PRESS_DEPTH    = 0.008       # m — impulse indentation depth
OCCLUSION_FRAMES  = (100, 130)   # frame window for occlusion disturbance
CONTACT_FRAME     = 150          # frame at which tool contact fires
OCCLUSION_NOISE_MULT = 4.0       # noise multiplier during occlusion

# ── MDP ───────────────────────────────────────────────────────────────────
N_ACTIONS      = 3           # {0: decrease, 1: hold, 2: increase}
N_STATE_BINS   = 5           # bins per state dimension
N_STATES       = N_STATE_BINS ** 4   # 625 total discrete states

# State normalisation bounds [d_CD_mm, mean_speed, delta_cloud_mm, k_obs]
STATE_LOW  = [0.0,  0.0,  0.0,  K_OBS_MIN]
STATE_HIGH = [8.0,  0.10, 8.0,  K_OBS_MAX]

# Episode type codes
EP_NORMAL    = 0
EP_OCCLUSION = 1
EP_CONTACT   = 2
EP_NAMES     = {EP_NORMAL: "Normal", EP_OCCLUSION: "Occlusion", EP_CONTACT: "Contact"}

# ── Training ──────────────────────────────────────────────────────────────
N_TRAIN        = 1000        # total training episodes
EVAL_EVERY     = 50          # evaluate greedy policy every N episodes
N_TEST         = 30          # test episodes per condition at evaluation
ALPHA          = 0.3         # Q-table learning rate
GAMMA          = 0.95        # discount factor
EPS_START      = 1.0         # initial epsilon (full exploration)
EPS_END        = 0.05        # final epsilon (near-greedy)
EPS_DECAY_EPS  = 500         # episodes over which epsilon decays
RANDOM_SEED    = 42

# ── Output ────────────────────────────────────────────────────────────────
RESULTS_DIR    = "results"
CHECKPOINT_EVERY = 200       # save agent checkpoint every N episodes
