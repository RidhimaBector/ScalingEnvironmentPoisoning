"""Where do attacked grid victims leave the target route? (evaluation only)

    python scripts/diagnose_grid_route.py <folder of run folders> [victims per run] [key=value ...]

Trailing key=value pairs override the runs' saved settings (e.g. hold_threshold=1.0).

For each run: 20 fresh victims (same seeds as score_attack.py), attacked by the saved
best checkpoint. At the end of the attack episode, walk the target route and find the
first state where the victim's greedy route differs, and why:
  action    the victim's greedy action in that state is not the target action
  dynamics  the action is right, but the most likely next state under the attacked
            dynamics is not the next state of the target route
"""

import glob
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score_attack import load_run
from utils.metrics import greedy_path

root = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
overrides = {k: float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
for run in sorted(glob.glob(root + "/*")):
    cfg, attack_env, translator, attacker, used = load_run(run, overrides=overrides)
    target = attack_env._target
    t_act = target.argmax(1)
    on_target = target.sum(1) > 0
    why, where, n_wrong, wrong_states, exact_any, exact_last = (
        Counter(),
        Counter(),
        [],
        Counter(),
        0,
        0,
    )
    route_ok_steps = []
    for ep in range(N):
        np.random.seed(5000 + ep)
        torch.manual_seed(5000 + ep)
        obs, _ = attack_env.reset()
        hits = []
        for _ in range(cfg["max_timesteps"]):
            obs, _, _, _, info = attack_env.step(
                translator.translate(
                    attacker.act_greedy(np.asarray(obs, dtype=np.float32))
                )
            )
            hits.append(info["path_match"])
        route_ok_steps.append(hits)
        exact_any += max(hits) > 0
        exact_last += hits[-1] > 0
        env, algo = attack_env._victims.victims[0]
        g = np.asarray(algo.get_greedy_actions())
        want = greedy_path(env.INIT_T, t_act, env.start_state, env._goal_state)
        bad = [s for s in want[:-1] if g[s] != t_act[s]]
        n_wrong.append(len(bad))
        wrong_states.update(bad)
        if hits[-1] > 0:
            continue
        for i, s in enumerate(want[:-1]):
            if g[s] != t_act[s]:
                why["action"] += 1
                where[i] += 1
                break
            if int(np.argmax(env.T[g[s]][s])) != want[i + 1]:
                why["dynamics"] += 1
                where[i] += 1
                break
        else:
            why["other"] += 1
    name = run.split("__")[2] + " " + run.split("__")[3]
    steps = np.mean(route_ok_steps, axis=0)
    print(f"\n{name}  (checkpoint {used}, route length {len(want) - 1} moves)")
    print(
        f"  exact route at last step: {exact_last}/{N};  at any of the {cfg['max_timesteps']} steps: {exact_any}/{N}"
    )
    print(
        f"  share of victims on route per attack step: {' '.join(f'{x:.2f}' for x in steps)}"
    )
    print(
        f"  wrong actions on the route per victim: mean {np.mean(n_wrong):.2f}, counts {dict(sorted(Counter(n_wrong).items()))}"
    )
    print(f"  first break is caused by: {dict(why)}")
    print(
        f"  position on route of first break (0 = start): {dict(sorted(where.items()))}"
    )
    print(
        f"  route states most often wrong (state: victims): {dict(wrong_states.most_common(6))}"
    )
    print(f"  target route: {want}")
    print(
        f"  final altitude map:\n{np.round(np.asarray(env.get_dynamics()).reshape(env.shape), 1)}"
    )
