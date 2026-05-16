# =============================================================================
# env/episode_types.py
#
# Episode type definitions and disturbance schedules.
#
# Three structured episode types make the RL problem non-trivial:
#
#   NORMAL    — Gaussian noise only. Fixed kobs near-optimal. Baseline.
#   OCCLUSION — Frames 100-130: camera corrupted with 4x noise.
#               Optimal response: decrease kobs to ignore bad data.
#   CONTACT   — Frame 150: tool impulse causes rapid surface displacement.
#               Optimal response: spike kobs to re-track quickly.
#
# These scenarios ensure the optimal policy is genuinely state-dependent.
# Without them, any fixed gain performs identically and RL has nothing to
# learn.
# =============================================================================

from config import (EP_NORMAL, EP_OCCLUSION, EP_CONTACT,
                    OCCLUSION_FRAMES, CONTACT_FRAME,
                    OCCLUSION_NOISE_MULT, NOISE_SIGMA, PRESS_DEPTH)
import numpy as np


def apply_disturbance(
    cloud:         np.ndarray,
    ep_type:       int,
    step_num:      int,
    tool_weights:  np.ndarray,
    rng:           np.random.Generator,
) -> np.ndarray:
    """
    Apply episode-type-specific disturbance to the point cloud.

    Parameters
    ----------
    cloud        : (M, 3) current ground-truth point cloud
    ep_type      : episode type code
    step_num     : current frame index
    tool_weights : (M,) Gaussian tool footprint weights
    rng          : seeded random generator

    Returns
    -------
    cloud : (M, 3) distorted point cloud (modified in-place copy)
    """
    cloud = cloud.copy()

    if ep_type == EP_OCCLUSION:
        occ_start, occ_end = OCCLUSION_FRAMES
        if occ_start <= step_num <= occ_end:
            # Add 4x extra noise to simulate camera shadow / dropout
            extra = rng.normal(
                0, NOISE_SIGMA * OCCLUSION_NOISE_MULT, cloud.shape
            ).astype(np.float32)
            cloud = cloud + extra

    elif ep_type == EP_CONTACT:
        if step_num == CONTACT_FRAME:
            # Impulse: push surface points downward by tool profile
            cloud[:, 1] -= (PRESS_DEPTH * tool_weights).astype(np.float32)

    return cloud
