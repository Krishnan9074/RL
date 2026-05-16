# =============================================================================
# agents/qlearning.py
#
# Off-policy tabular Q-Learning.
#
# Update rule:
#   Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
#
# Key property: always bootstraps on the greedy maximum regardless of which
# exploration action was actually taken. This makes Q-learning optimistic
# and produces better asymptotic policies than SARSA in this environment,
# at the cost of slight overestimation during noisy training.
# =============================================================================

from agents.base import BaseTabularAgent
import numpy as np


class QLearningAgent(BaseTabularAgent):
    """
    Tabular Q-Learning with epsilon-greedy exploration.

    Parameters
    ----------
    alpha  : learning rate (0 < alpha <= 1)
    gamma  : discount factor
    eps_*  : epsilon schedule parameters
    """

    def update(
        self,
        s:    int,
        a:    int,
        r:    float,
        s_next: int,
        done: bool,
    ):
        """
        One Q-Learning update step.

        Parameters
        ----------
        s      : current state index
        a      : action taken
        r      : reward received
        s_next : next state index
        done   : whether the episode ended
        """
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q[s_next])

        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
