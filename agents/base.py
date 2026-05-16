# =============================================================================
# agents/base.py
# Abstract base class shared by Q-Learning and SARSA.
# =============================================================================

import pickle
import numpy as np
from abc import ABC, abstractmethod
from config import N_STATES, N_ACTIONS, EPS_START, EPS_END, EPS_DECAY_EPS


class BaseTabularAgent(ABC):
    """
    Common interface for all tabular RL agents.

    Sub-classes must implement:
        update(...)  — one TD step
    """

    def __init__(
        self,
        alpha:      float = 0.3,
        gamma:      float = 0.95,
        eps_start:  float = EPS_START,
        eps_end:    float = EPS_END,
        eps_decay:  int   = EPS_DECAY_EPS,
        n_states:   int   = N_STATES,
        n_actions:  int   = N_ACTIONS,
    ):
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.episode   = 0
        self.Q         = np.zeros((n_states, n_actions), dtype=np.float64)

    @property
    def epsilon(self) -> float:
        """Linear epsilon annealing from eps_start to eps_end."""
        frac = min(1.0, self.episode / max(self.eps_decay, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        import random
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.Q.shape[1] - 1)
        return int(np.argmax(self.Q[state]))

    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    def end_episode(self):
        """Call at the end of every training episode."""
        self.episode += 1

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"Q": self.Q, "episode": self.episode}, f)
        print(f"  Saved agent to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.Q       = d["Q"]
        self.episode = d["episode"]
        print(f"  Loaded agent from {path} (episode {self.episode})")
