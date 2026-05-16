# =============================================================================
# training/trainer.py
#
# Training loop for Q-Learning and SARSA agents.
# Handles episode rollouts, periodic evaluation, checkpointing,
# and history logging for later visualisation.
# =============================================================================

import os
import time
from typing import Optional, Dict, List

import numpy as np

from config import (
    N_TRAIN, EVAL_EVERY, N_TEST, CHECKPOINT_EVERY,
    RESULTS_DIR, EP_NORMAL, EP_OCCLUSION, EP_CONTACT,
)
from env.environment import SkinTrackingEnv, discretise
from agents.qlearning import QLearningAgent
from agents.sarsa     import SARSAAgent


# ── Episode runners ───────────────────────────────────────────────────────

def run_episode(
    env,
    agent,
    ep_type:  Optional[int] = None,
    train:    bool           = True,
) -> Dict:
    """
    Run a single episode with any agent type.
    Dispatches to the correct update signature for QL vs SARSA.

    Returns
    -------
    dict with keys: total_reward, mean_cd_mm, k_traj, ep_type
    """
    obs   = env.reset(episode_type=ep_type)
    state = discretise(obs)

    is_sarsa = isinstance(agent, SARSAAgent)
    if is_sarsa:
        action = agent.select_action(state, greedy=not train)

    total_r  = 0.0
    total_cd = 0.0
    steps    = 0
    k_traj: List[float] = []

    while True:
        if not is_sarsa:
            action = agent.select_action(state, greedy=not train)

        obs_next, reward, done, info = env.step(action)
        s_next = discretise(obs_next)

        if train:
            if is_sarsa:
                a_next = agent.select_action(s_next, greedy=False)
                agent.update(state, action, reward, s_next, a_next, done)
                action = a_next
            else:
                agent.update(state, action, reward, s_next, done)

        total_r  += reward
        total_cd += info["d_cd_mm"]
        k_traj.append(info["k_obs"])
        state = s_next
        steps += 1

        if done:
            break

    if train:
        agent.end_episode()

    return {
        "total_reward": total_r,
        "mean_cd_mm":   total_cd / steps,
        "k_traj":       k_traj,
        "ep_type":      info["ep_type"],
    }


# ── Trainer class ─────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the full training lifecycle for a single agent.

    Parameters
    ----------
    env        : SkinTrackingEnv
    agent      : QLearningAgent or SARSAAgent
    name       : human-readable label used in logs and filenames
    n_episodes : total training episodes
    """

    def __init__(
        self,
        env:        SkinTrackingEnv,
        agent,
        name:       str = "agent",
        n_episodes: int = N_TRAIN,
    ):
        self.env        = env
        self.agent      = agent
        self.name       = name
        self.n_episodes = n_episodes

        self.history: Dict[str, List] = {
            "train_reward":  [],
            "train_cd":      [],
            "eval_cd":       [],
            "eval_episodes": [],
            "epsilon":       [],
        }

        os.makedirs(RESULTS_DIR, exist_ok=True)

    def train(self, verbose: bool = True) -> Dict:
        """
        Run the full training loop.

        Returns
        -------
        history dict with training and evaluation metrics.
        """
        print(f"\n{'='*55}")
        print(f"  Training {self.name}  ({self.n_episodes} episodes)")
        print(f"{'='*55}")

        t0 = time.time()

        for ep in range(self.n_episodes):
            result = run_episode(self.env, self.agent, train=True)
            self.history["train_reward"].append(result["total_reward"])
            self.history["train_cd"].append(result["mean_cd_mm"])
            self.history["epsilon"].append(self.agent.epsilon)

            # Periodic greedy evaluation
            if (ep + 1) % EVAL_EVERY == 0:
                eval_cds = []
                for et in [EP_NORMAL, EP_OCCLUSION, EP_CONTACT]:
                    r = run_episode(self.env, self.agent,
                                    ep_type=et, train=False)
                    eval_cds.append(r["mean_cd_mm"])
                mean_eval = float(np.mean(eval_cds))
                self.history["eval_cd"].append(mean_eval)
                self.history["eval_episodes"].append(ep + 1)

                if verbose:
                    recent_cd = np.mean(self.history["train_cd"][-EVAL_EVERY:])
                    print(
                        f"  ep={ep+1:4d}  "
                        f"ε={self.agent.epsilon:.3f}  "
                        f"train_cd={recent_cd:.3f}mm  "
                        f"eval_cd={mean_eval:.3f}mm  "
                        f"({time.time()-t0:.0f}s)"
                    )

            # Periodic checkpoint
            if (ep + 1) % CHECKPOINT_EVERY == 0:
                ckpt = os.path.join(
                    RESULTS_DIR,
                    f"{self.name}_ep{ep+1}.pkl"
                )
                self.agent.save(ckpt)

        # Final checkpoint
        final_path = os.path.join(RESULTS_DIR, f"{self.name}_final.pkl")
        self.agent.save(final_path)

        elapsed = time.time() - t0
        print(f"\n✓ {self.name} done in {elapsed:.0f}s "
              f"| final eval_cd = {self.history['eval_cd'][-1]:.3f}mm")

        return self.history
