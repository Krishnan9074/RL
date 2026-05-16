#!/usr/bin/env python3
# =============================================================================
# main.py
#
# Entry point for the Adaptive k_obs RL project.
# Runs the full pipeline: train -> evaluate -> plot.
#
# Usage:
#   python main.py                          # full run (1000 episodes)
#   python main.py --episodes 200           # quick smoke test
#   python main.py --eval-only              # skip training, load from results/
#   python main.py --device cpu             # force CPU
#
# Output files written to results/:
#   ql_agent_final.pkl      Q-Learning Q-table
#   sarsa_agent_final.pkl   SARSA Q-table
#   fig1_learning_curves.*  Training convergence
#   fig2_eval_bar.*         Baseline comparison
#   fig3_kobs_traj.*        Adaptive gain trajectories
#   fig4_qtable_heatmap.*   Greedy policy heatmap
#   fig5_epsilon_decay.*    Exploration schedule
# =============================================================================

import argparse
import random
import os

import numpy as np

from config import (
    DEVICE, RANDOM_SEED, N_TRAIN, ALPHA, GAMMA,
    EPS_START, EPS_END, EPS_DECAY_EPS,
    RESULTS_DIR,
)
from env.environment   import SkinTrackingEnv
from agents            import QLearningAgent, SARSAAgent, FixedGainBaseline, RandomBaseline
from training          import Trainer
from evaluation        import Evaluator
from visualization     import Plotter


def parse_args():
    p = argparse.ArgumentParser(
        description="Adaptive kobs RL — NeuroSim XPBD Tissue Tracking"
    )
    p.add_argument("--episodes", type=int,   default=N_TRAIN,
                   help="Number of training episodes (default: 1000)")
    p.add_argument("--alpha",    type=float, default=ALPHA,
                   help="Learning rate (default: 0.3)")
    p.add_argument("--gamma",    type=float, default=GAMMA,
                   help="Discount factor (default: 0.95)")
    p.add_argument("--device",   type=str,   default=DEVICE,
                   choices=["cuda", "cpu"],
                   help="Compute device (default: auto-detect)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load saved agents from results/")
    p.add_argument("--seed",     type=int,   default=RANDOM_SEED)
    return p.parse_args()


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    set_seeds(args.seed)

    print("=" * 60)
    print("  Adaptive k_obs Control via RL — NeuroSim XPBD")
    print(f"  Device:   {args.device}")
    print(f"  Episodes: {args.episodes}")
    print(f"  α={args.alpha}  γ={args.gamma}  seed={args.seed}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Environment ────────────────────────────────────────────────────────
    env = SkinTrackingEnv(device=args.device, seed=args.seed)

    # ── Agents ────────────────────────────────────────────────────────────
    ql_agent = QLearningAgent(
        alpha=args.alpha, gamma=args.gamma,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_EPS,
    )
    sarsa_agent = SARSAAgent(
        alpha=args.alpha, gamma=args.gamma,
        eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_EPS,
    )

    # ── Train or load ─────────────────────────────────────────────────────
    if args.eval_only:
        ql_path    = os.path.join(RESULTS_DIR, "Q-Learning_final.pkl")
        sarsa_path = os.path.join(RESULTS_DIR, "SARSA_final.pkl")
        ql_agent.load(ql_path)
        sarsa_agent.load(sarsa_path)
        # Dummy histories for plotting
        ql_history    = {"train_cd": [], "train_reward": [],
                         "eval_cd": [], "eval_episodes": [], "epsilon": []}
        sarsa_history = {"train_cd": [], "train_reward": [],
                         "eval_cd": [], "eval_episodes": [], "epsilon": []}
    else:
        ql_trainer = Trainer(
            env, ql_agent, name="Q-Learning", n_episodes=args.episodes
        )
        ql_history = ql_trainer.train(verbose=True)

        sarsa_trainer = Trainer(
            env, sarsa_agent, name="SARSA", n_episodes=args.episodes
        )
        sarsa_history = sarsa_trainer.train(verbose=True)

    # ── Baselines ──────────────────────────────────────────────────────────
    baselines = {
        "Fixed 48 N/m":  FixedGainBaseline(48.0,  env),
        "Fixed 80 N/m":  FixedGainBaseline(80.0,  env),
        "Fixed 10 N/m":  FixedGainBaseline(10.0,  env),
        "Random Policy": RandomBaseline(),
    }

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n── Evaluation ─────────────────────────────────────────────")
    ev = Evaluator(env)
    ev.add("Q-Learning", ql_agent)
    ev.add("SARSA",      sarsa_agent)
    for name, base in baselines.items():
        ev.add(name, base)
    results = ev.run()
    ev.print_table(results)

    # ── Figures ───────────────────────────────────────────────────────────
    if not args.eval_only:
        print("\n── Generating Figures ─────────────────────────────────────")
        plotter = Plotter(outdir=RESULTS_DIR)

        plotter.learning_curves(ql_history, sarsa_history)
        plotter.eval_bar(results)
        plotter.kobs_trajectories(
            env, ql_agent, sarsa_agent,
            fixed_baseline=baselines["Fixed 48 N/m"],
        )
        plotter.qtable_heatmap(ql_agent)
        plotter.epsilon_decay(ql_history, sarsa_history)

    print(f"\n✓ All outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
