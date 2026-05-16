# =============================================================================
# agents/baselines.py
#
# Fixed-gain and random baselines for comparison.
#
# These establish lower bounds: if RL cannot beat a simple constant gain,
# the sequential decision framing is not providing value.
# =============================================================================

import random
from config import K_OBS_INIT


class FixedGainBaseline:
    """
    Always holds k_obs at a fixed value by returning action=1 (hold).
    Resets k_obs to the target on each episode start.

    Parameters
    ----------
    k_obs : float — fixed gain value in N/m
    env   : SkinTrackingEnv — reference to reset k_obs on each episode
    """

    def __init__(self, k_obs: float, env):
        self.k_obs = k_obs
        self.env   = env

    def reset(self):
        """Force k_obs back to target at episode start."""
        self.env.k_obs = self.k_obs

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Always hold — action 1."""
        return 1

    def end_episode(self):
        pass


class RandomBaseline:
    """
    Selects a uniformly random action at every step.
    Lower-bound baseline — should be beaten by any learned policy.
    """

    def select_action(self, state: int, greedy: bool = False) -> int:
        return random.randint(0, 2)

    def reset(self):
        pass

    def end_episode(self):
        pass
