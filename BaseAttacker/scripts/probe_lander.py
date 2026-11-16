"""Feasibility probe for the LunarLander attack (no attacker involved).

Question: if the environment is poisoned by hand from the start, does a victim that
trains in it end up behaving differently, and closer to the target? If no hand-set
poisoning moves the victim, no learned attacker can either, and the budget or the
target needs rethinking before the attack is judged.

For one victim algorithm and one fixed poisoning pattern, trains fresh victims and
reports, over the last 50 training episodes:
  return        the victim's own mean episode return (is it learning at all?)
  target_match  share of recorded target states where its greedy action is the target's
  final_x       mean horizontal position at the end of an episode (target drifts right)
  main_engine   share of steps firing the main engine (target: 0.57)

    python scripts/probe_lander.py --algo vpg --pattern right --episodes 1500 --seeds 3

Patterns: none | all | left | right  (engine-failure level 5 = always fail, per region)
"""

import argparse

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.lunar_lander import LunarLanderAttack  # noqa: E402
from registry import VICTIM_ALGO_MAP  # noqa: E402

PATTERNS = {
    "none": np.zeros(10),
    "all": np.full(10, 5.0),
    "left": np.array([5.0] * 5 + [0.0] * 5),
    "right": np.array([0.0] * 5 + [5.0] * 5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algo", default="vpg", choices=["dqn", "vpg", "a2c"])
    ap.add_argument("--pattern", default="none", choices=sorted(PATTERNS))
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--report_every", type=int, default=250)
    a = ap.parse_args()

    env = LunarLanderAttack()
    recorded = env.target_trajectories()
    target_states = np.stack([t[2] for t in recorded]).astype(np.float32)
    target_actions = np.array([int(t[3]) for t in recorded])

    for seed in range(a.seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        env.reset_dynamics()
        level = PATTERNS[a.pattern].copy()
        while np.any(level > 0):  # perturbations are at most 1 per step
            step = np.clip(level, 0, 1)
            env.apply_perturbation(step)
            level -= step
        victim = VICTIM_ALGO_MAP[a.algo].from_env(env)

        returns, final_x, main_engine = [], [], []
        for start in range(0, a.episodes, 10):
            stats = victim.train(env, 10)
            returns += stats["episode_rewards"]
            transitions = victim.get_trajectories()
            if transitions:
                final_x.append(float(transitions[-1][2][0]))
                main_engine.append(float(np.mean([t[3] == 2 for t in transitions])))
            done = start + 10
            if done % a.report_every == 0 or done >= a.episodes:
                match = float(
                    np.mean(victim.greedy_actions_for(target_states) == target_actions)
                )
                print(
                    f"PROBE algo={a.algo} pattern={a.pattern} seed={seed} episodes={done}"
                    f" | return {np.mean(returns[-50:]):7.1f} | target_match {match:.2f}"
                    f" | final_x {np.mean(final_x[-5:]):+.2f} | main_engine {np.mean(main_engine[-5:]):.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
