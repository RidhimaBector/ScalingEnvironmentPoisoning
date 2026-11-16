"""Feasibility probe for the grid (no attacker involved).

Sets the altitude to a hand-built map from the first attack epoch onward and reports
how often a Q-learning victim ends up on the exact target route. It shows whether the
target is reachable at all for a grid size, and which kind of map does it.

Maps (perimeter cells at altitude 10 in all of them):
  pit      interior at 0
  stripes  interior rows alternate 0 / 10
  checker  interior cells alternate 0 / 10

    python scripts/probe_grid_maps.py --sizes 4 5 6 8 10 --victims 8

Result of 2026-11-16 (8 victims, share on the exact route over attack epochs 11-15):
  4x4: pit 1.00 | 5x5: stripes 1.00 | 6x6: checker 0.65 | 8x8: checker 0.68 | 10x10: checker 0.85
  The pit map, which the 4x4 attacker learns, scores 0.00 on 8x8: a flat low interior
  becomes a shortcut to the goal once the perimeter is long.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.grid3d import Grid3D  # noqa: E402
from utils.metrics import greedy_path  # noqa: E402
from victim.victim_qlearning import VictimQLearning  # noqa: E402


def hand_map(env, kind):
    target = env.make_target_policy()
    on_route = set(greedy_path(env.INIT_T, target.argmax(1), 0, env._goal_state))
    rows, cols = env.shape
    alt = np.zeros(env.nS)
    for s in range(env.nS):
        r, c = divmod(s, cols)
        if s in on_route:
            alt[s] = 10.0
        elif kind == "stripes":
            alt[s] = 0.0 if r % 2 == 1 else 10.0
        elif kind == "checker":
            alt[s] = 0.0 if (r + c) % 2 == 1 else 10.0
    return alt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--victims", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--victim_episodes", type=int, default=80)
    a = ap.parse_args()

    for n in a.sizes:
        for kind in ("pit", "stripes", "checker"):
            route, match = [], []
            for seed in range(a.victims):
                np.random.seed(seed)
                env = Grid3D(grid_dimensions=(n, n))
                target = env.make_target_policy()
                mask, actions = target.sum(1) > 0, target.argmax(1)
                goal_map = hand_map(env, kind)
                victim = VictimQLearning(env.nS, env.nA)
                r, m = [], []
                for _ in range(a.epochs):
                    env.apply_perturbation(np.clip(goal_map - env.get_dynamics(), -1, 1))
                    victim.train(env, a.victim_episodes)
                    greedy = victim.get_greedy_actions()
                    m.append(np.mean(greedy[mask] == actions[mask]))
                    r.append(env.oracle_route_match(actions, greedy))
                route.append(r)
                match.append(m)
            route, match = np.mean(route, 0), np.mean(match, 0)
            print(f"{n}x{n} {kind:8s} exact route, last 5 epochs: {route[-5:].mean():.2f}"
                  f"  final epoch: {route[-1]:.2f}  target match, last 5: {match[-5:].mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
