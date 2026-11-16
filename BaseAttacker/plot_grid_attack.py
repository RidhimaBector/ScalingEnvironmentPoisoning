"""Draw what a trained attacker does to a grid victim.

Runs the saved attacker without exploration against fresh victims, prints ground-truth
success, and saves a figure: target route, natural route, attacked route, how often
victims visit each cell before and after, and the altitude map the attacker built.

    python plot_grid_attack.py --run_dir results/<run>

Grid environments only; score_attack.py scores any environment.
"""
import argparse
import os


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from score_attack import load_run
from utils.metrics import greedy_path

ARROWS = {0: (0, -0.3), 1: (0.3, 0), 2: (0, 0.3), 3: (-0.3, 0)}  # N E S W as (dx, dy)


def short(path):
    """Collapse a path that got stuck in one cell: [0, 4, 4, 4, ...] -> '[0, 4] (stuck)'."""
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            return f"{path[:i]} (stuck at {path[i]})"
    return str(path)


def rollout_visits(env, algo, n=300, max_len=200):
    """State-visit frequencies of the victim's own softmax policy."""
    visits = np.zeros(env.nS)
    goal = int(np.ravel_multi_index((env.shape[0] - 1, 0), env.shape))
    for _ in range(n):
        s = 0
        seen = {0}
        for _ in range(max_len):
            s = int(np.random.choice(env.nS, p=env.T[algo.act(s)][s]))
            seen.add(s)
            if s == goal:
                break
        for v in seen:
            visits[v] += 1
    return visits / n


def draw_policy(ax, shape, actions, path, title, color):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, shape[1] - 0.5); ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_xticks(range(shape[1])); ax.set_yticks(range(shape[0])); ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    for s in range(shape[0] * shape[1]):
        r, c = divmod(s, shape[1])
        if actions is not None and actions[s] >= 0:
            dx, dy = ARROWS[int(actions[s])]
            on_path = s in path[:-1]
            ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.12,
                     color=color if on_path else '0.75', length_includes_head=True)
    r, c = divmod(path[0], shape[1]); ax.text(c, r - 0.33, 'start', ha='center', fontsize=7)
    r, c = shape[0] - 1, 0; ax.text(c, r + 0.4, 'goal', ha='center', fontsize=7)


def heat(ax, data, title, cmap, vmin=None, vmax=None, fmt='{:.1f}'):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for (r, c), v in np.ndenumerate(data):
        ax.text(c, r, fmt.format(v), ha='center', va='center', fontsize=8)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--checkpoint', default='best_model')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--out', default=None, help='figure path (default: figures/<run>_eval.png)')
    # Only needed for runs trained before run_config.json existed, or to override it:
    ap.add_argument('--ae_model_path', default=None)
    ap.add_argument('--obs_dynamics', default=None, choices=['actions', 'env', 'none'])
    ap.add_argument('--grid_size', type=int, default=None)
    a = ap.parse_args()

    overrides = {k: v for k, v in (('ae_model_path', a.ae_model_path), ('obs_dynamics', a.obs_dynamics),
                                   ('grid_size', a.grid_size)) if v is not None}
    cfg, env, translator, algo, _ = load_run(a.run_dir, a.checkpoint, overrides)
    a.max_timesteps, a.victim_n_episodes = cfg['max_timesteps'], cfg['victim_n_episodes']
    pop = env.victim_system
    venv = pop.env
    if not hasattr(venv, 'INIT_T'):
        raise SystemExit("plot_grid_attack.py draws grid environments only; "
                         "use score_attack.py for other environments.")
    shape = venv.shape
    goal = int(np.ravel_multi_index((shape[0] - 1, 0), shape))
    tgt_actions = np.argmax(pop.target, axis=1)
    tgt_path = greedy_path(venv.INIT_T, tgt_actions, 0, goal)

    # Natural victim (no attack), same total training budget
    np.random.seed(1000)
    pop.reset()
    pop.algorithm.train(venv, a.victim_n_episodes * a.max_timesteps)
    nat_actions = pop.algorithm.get_greedy_actions().copy()
    nat_path = greedy_path(venv.T, nat_actions, 0, goal)
    nat_visits = rollout_visits(venv, pop.algorithm)

    greedy = np.zeros((a.episodes, a.max_timesteps)); match = np.zeros_like(greedy)
    finals = []  # per eval episode: (path_match at last step, greedy actions, visits, altitude)
    for ep in range(a.episodes):
        np.random.seed(2000 + ep)
        obs, _ = env.reset()
        for t in range(a.max_timesteps):
            with torch.no_grad():
                act = algo.act_greedy(np.asarray(obs, dtype=np.float32))
            obs, _, _, _, info = env.step(translator.translate(act))
            greedy[ep, t] = info['greedy_accuracy']; match[ep, t] = info['path_match']
        acts = pop.algorithm.get_greedy_actions().copy()
        finals.append((match[ep, -1], acts, greedy_path(venv.T, acts, 0, goal),
                       rollout_visits(venv, pop.algorithm), venv.altitude.copy()))
    n_final_match = int(sum(f[0] for f in finals))
    # Show a representative episode: one with the majority outcome at the final step.
    majority = 1.0 if n_final_match * 2 >= len(finals) else 0.0
    _, atk_actions, atk_path, _, atk_alt = next(f for f in finals if f[0] == majority)
    atk_visits = np.mean([f[3] for f in finals], axis=0)

    print(f"run_dir={a.run_dir}  checkpoint={a.checkpoint}  episodes={a.episodes} (deterministic actor, fresh victims)")
    print("target path           :", tgt_path)
    print("natural greedy path   :", short(nat_path))
    print("attacked greedy path  :", short(atk_path), "(representative eval episode)")
    print(f"exact target path at final attack step: {n_final_match}/{len(finals)} eval episodes")
    print("greedy-acc by step    :", np.round(greedy.mean(0), 2))
    print("path-match by step    :", np.round(match.mean(0), 2))
    print(f"greedy-acc  mean over last 5 steps: {greedy[:, -5:].mean():.3f}")
    print(f"path-match  mean over last 5 steps: {match[:, -5:].mean():.3f}")
    print(f"victim rollouts visiting every target-path cell: natural={np.min(nat_visits[tgt_path]):.2f}  attacked={np.min(atk_visits[tgt_path]):.2f}")
    print("final altitude:\n", np.round(atk_alt, 1))

    fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
    draw_policy(axs[0, 0], shape, tgt_actions * (pop.target.sum(1) > 0) - (pop.target.sum(1) == 0), tgt_path, 'Target path (attacker goal)', 'tab:red')
    draw_policy(axs[0, 1], shape, nat_actions, nat_path, f'Natural victim greedy path\n{short(nat_path)}', 'tab:blue')
    draw_policy(axs[0, 2], shape, atk_actions, atk_path, f'Attacked victim greedy path ({n_final_match}/{len(finals)} exact at end)\n{short(atk_path)}', 'tab:red')
    ax = axs[0, 3]
    ax.plot(range(1, a.max_timesteps + 1), greedy.mean(0), label='target-action match (greedy)')
    ax.plot(range(1, a.max_timesteps + 1), match.mean(0), label='exact path match')
    ax.set_ylim(0, 1.05); ax.set_xlabel('attack step'); ax.set_title(f'Poisoning success ({a.episodes} eval episodes)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    heat(axs[1, 0], venv.altitude_default, 'Natural altitude', 'terrain', 0, 10)
    heat(axs[1, 1], nat_visits.reshape(shape), 'Natural victim: P(visit cell) per episode', 'Blues', 0, 1, '{:.2f}')
    heat(axs[1, 2], atk_visits.reshape(shape), 'Attacked victim: P(visit cell), eval mean', 'Reds', 0, 1, '{:.2f}')
    heat(axs[1, 3], atk_alt, 'Poisoned altitude (after 15 attack steps)', 'terrain', 0, 10)
    fig.tight_layout()
    out = a.out or os.path.join('figures', os.path.basename(os.path.normpath(a.run_dir)) + '_eval.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("figure:", out)


if __name__ == '__main__':
    main()
