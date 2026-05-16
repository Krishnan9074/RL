# =============================================================================
# evaluation/evaluator.py
#
# Held-out evaluation across all agents and episode types.
# Runs N_TEST greedy episodes per (agent, episode_type) combination and
# reports mean ± std Chamfer distance.
# =============================================================================

from typing import Dict, List
import numpy as np

from config import N_TEST, EP_NAMES
from env.environment import discretise
from training.trainer import run_episode


class Evaluator:
    """
    Runs systematic held-out evaluation after training.

    Usage
    -----
    ev = Evaluator(env)
    ev.add("Q-Learning", ql_agent)
    ev.add("Fixed 48",   fixed_baseline)
    results = ev.run()
    ev.print_table(results)
    """

    def __init__(self, env, n_test: int = N_TEST):
        self.env    = env
        self.n_test = n_test
        self._agents: Dict[str, object] = {}

    def add(self, name: str, agent):
        """Register an agent or baseline under a display name."""
        self._agents[name] = agent

    def run(self) -> Dict:
        """
        Evaluate all registered agents across all episode types.

        Returns
        -------
        Nested dict: results[agent_name][episode_name] = {mean, std, samples}
        """
        ep_codes  = list(EP_NAMES.keys())
        ep_labels = list(EP_NAMES.values())
        results   = {}

        print(f"\n{'='*65}")
        print(f"{'Agent':22s}  {'Normal':>14s}  {'Occlusion':>14s}  {'Contact':>14s}")
        print(f"{'-'*65}")

        for name, agent in self._agents.items():
            results[name] = {}
            row = f"{name:22s}"

            for code, label in zip(ep_codes, ep_labels):
                cds: List[float] = []
                for _ in range(self.n_test):
                    r = run_episode(self.env, agent,
                                    ep_type=code, train=False)
                    cds.append(r["mean_cd_mm"])

                m, s = float(np.mean(cds)), float(np.std(cds))
                results[name][label] = {"mean": m, "std": s, "samples": cds}
                row += f"  {m:5.3f}±{s:.3f}   "

            print(row)

        print(f"{'='*65}")
        return results

    def print_table(self, results: Dict):
        """Pretty-print the results table (alias for convenience)."""
        ep_labels = list(EP_NAMES.values())
        print(f"\n{'Agent':22s}", end="")
        for ep in ep_labels:
            print(f"  {ep:>14s}", end="")
        print(f"  {'Mean':>8s}")
        print("-" * 75)
        for name, ep_res in results.items():
            means = [ep_res[ep]["mean"] for ep in ep_labels]
            print(f"{name:22s}", end="")
            for ep in ep_labels:
                m, s = ep_res[ep]["mean"], ep_res[ep]["std"]
                print(f"  {m:5.3f}±{s:.3f} ", end="")
            print(f"  {np.mean(means):6.3f}")
