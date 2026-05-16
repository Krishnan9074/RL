# =============================================================================
# agents/sarsa.py
#
# On-policy tabular SARSA.
#
# Update rule:
#   Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
#
# where a' is the action actually selected under the current epsilon-greedy
# policy at state s'. Unlike Q-Learning, SARSA accounts for the exploration
# cost during training — it learns the value of the epsilon-greedy policy,
# not the greedy policy. This makes SARSA more conservative and stable in
# stochastic environments, at the cost of a slightly higher asymptotic CD.
# =============================================================================

from agents.base import BaseTabularAgent


class SARSAAgent(BaseTabularAgent):
    """
    Tabular SARSA with epsilon-greedy exploration.

    Parameters
    ----------
    alpha  : learning rate (0 < alpha <= 1)
    gamma  : discount factor
    eps_*  : epsilon schedule parameters
    """

    def update(
        self,
        s:      int,
        a:      int,
        r:      float,
        s_next: int,
        a_next: int,
        done:   bool,
    ):
        """
        One SARSA update step.

        Parameters
        ----------
        s      : current state index
        a      : action taken in state s
        r      : reward received
        s_next : next state index
        a_next : action taken in state s_next (already sampled under policy)
        done   : whether the episode ended
        """
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q[s_next, a_next]

        self.Q[s, a] += self.alpha * (target - self.Q[s, a])
