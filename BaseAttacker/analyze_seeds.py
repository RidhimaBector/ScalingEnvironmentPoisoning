"""Average-over-seeds analysis for Sacred/MongoDB runs.

Usage:
    python analyze_seeds.py                          # auto: latest N runs with encoder_mode=ae
    python analyze_seeds.py --run_ids 34 35 36
    python analyze_seeds.py --run_ids 34 35 36 --out figures/ae_seeds
    python analyze_seeds.py --encoder_mode ae --n 3  # latest 3 ae runs
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient


# ---------------------------------------------------------------------------
# Data loading (shared with analyze_run.py)
# ---------------------------------------------------------------------------

def connect(mongo_url='localhost:27017', db_name='env_poisoning'):
    return MongoClient(mongo_url)[db_name]


def get_metric(db, run_id, name):
    doc = db.metrics.find_one({'run_id': run_id, 'name': name})
    if doc is None:
        return None, None
    return np.array(doc['steps']), np.array(doc['values'])


def smooth(values, window=100):
    if len(values) < window:
        window = max(1, len(values) // 5)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def latest_ae_runs(db, encoder_mode='ae', n=None):
    """Return run IDs for the most recent runs with the given encoder_mode."""
    cursor = db.runs.find(
        {'config.encoder_mode': encoder_mode},
        {'_id': 1, 'config': 1, 'status': 1}
    ).sort('_id', -1)
    if n:
        cursor = cursor.limit(n)
    return list(cursor)


# ---------------------------------------------------------------------------
# Per-seed loading
# ---------------------------------------------------------------------------

def load_episode_series(db, run_id, metric_name):
    """Return (episodes, values) sorted by episode index."""
    steps, values = get_metric(db, run_id, metric_name)
    if steps is None:
        return None, None
    order = np.argsort(steps)
    return steps[order].astype(int), values[order]


def load_all_seeds(db, run_ids, metric_name):
    """Load a metric for all seeds and align to the shortest run."""
    series = []
    for rid in run_ids:
        eps, vals = load_episode_series(db, rid, metric_name)
        if eps is not None:
            series.append((eps, vals))

    if not series:
        return None, None

    min_len = min(len(v) for _, v in series)
    aligned = np.array([v[:min_len] for _, v in series])
    eps_common = series[0][0][:min_len]
    return eps_common, aligned   # (n_seeds, n_episodes)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_mean_std(ax, xs, matrix, color, label, alpha_fill=0.2, smooth_w=100):
    """Plot mean ± std with smoothing."""
    mean = np.mean(matrix, axis=0)
    std  = np.std(matrix, axis=0)

    sm_mean = smooth(mean, smooth_w)
    sm_std  = smooth(std,  smooth_w)
    pad = len(xs) - len(sm_mean)
    xs_sm = xs[pad // 2: pad // 2 + len(sm_mean)]

    ax.plot(xs_sm, sm_mean, color=color, linewidth=2, label=label)
    ax.fill_between(xs_sm, sm_mean - sm_std, sm_mean + sm_std,
                    alpha=alpha_fill, color=color)
    return xs_sm, sm_mean, sm_std


def fig_accuracy(db, run_ids, out_dir, label_prefix='AE encoder'):
    eps, matrix = load_all_seeds(db, run_ids, 'episode.accuracy')
    if matrix is None:
        return
    _, min_matrix = load_all_seeds(db, run_ids, 'episode.min_accuracy')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw per-seed + mean
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    for i, (rid, c) in enumerate(zip(run_ids, colors)):
        eps_i, vals_i = load_episode_series(db, rid, 'episode.accuracy')
        ax.plot(eps_i, vals_i, alpha=0.15, color=c, linewidth=0.5)
        sm = smooth(vals_i, 100)
        pad = len(eps_i) - len(sm)
        ax.plot(eps_i[pad//2: pad//2+len(sm)], sm, color=c, linewidth=1.2,
                label=f'seed {i} (run {rid})')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Episode'); ax.set_ylabel('@Acc')
    ax.set_title('Per-seed accuracy')
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: mean ± std across seeds
    ax2 = axes[1]
    _plot_mean_std(ax2, eps, matrix, color='steelblue',
                   label=f'Mean @Acc ± 1σ  ({len(run_ids)} seeds)')
    ax2.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('@Acc')
    ax2.set_title(f'Mean ± std  ({len(run_ids)} seeds)')
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{label_prefix}  |  runs {run_ids}', fontsize=12, fontweight='bold')
    path = os.path.join(out_dir, 'accuracy_seeds.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_reward(db, run_ids, out_dir, label_prefix='AE encoder'):
    eps, matrix = load_all_seeds(db, run_ids, 'episode.episode_reward')
    if matrix is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    for i, (rid, c) in enumerate(zip(run_ids, colors)):
        eps_i, vals_i = load_episode_series(db, rid, 'episode.episode_reward')
        sm = smooth(vals_i, 100)
        pad = len(eps_i) - len(sm)
        ax.plot(eps_i[pad//2: pad//2+len(sm)], sm, color=c, linewidth=1,
                alpha=0.6, label=f'seed {i}')

    _plot_mean_std(ax, eps, matrix, color='black', label='Mean ± 1σ', alpha_fill=0.15)
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Episode Reward  |  {label_prefix}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'reward_seeds.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_ddpg_loss(db, run_ids, out_dir, label_prefix='AE encoder'):
    eps, matrix = load_all_seeds(db, run_ids, 'ddpg_loss')
    if matrix is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    for i, (rid, c) in enumerate(zip(run_ids, colors)):
        eps_i, vals_i = load_episode_series(db, rid, 'ddpg_loss')
        sm = smooth(vals_i, 50)
        pad = len(eps_i) - len(sm)
        ax.plot(eps_i[pad//2: pad//2+len(sm)], sm, color=c, linewidth=1,
                alpha=0.6, label=f'seed {i}')

    _plot_mean_std(ax, eps, matrix, color='crimson', label='Mean ± 1σ',
                   alpha_fill=0.15, smooth_w=50)
    ax.set_xlabel('Episode'); ax.set_ylabel('Loss')
    ax.set_title(f'DDPG Loss  |  {label_prefix}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'ddpg_loss_seeds.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_dashboard(db, run_ids, out_dir, label_prefix='AE encoder'):
    """4-panel summary averaged over seeds."""
    eps_acc, mat_acc = load_all_seeds(db, run_ids, 'episode.accuracy')
    eps_rew, mat_rew = load_all_seeds(db, run_ids, 'episode.episode_reward')
    eps_loss, mat_loss = load_all_seeds(db, run_ids, 'ddpg_loss')
    _, mat_sftmx     = load_all_seeds(db, run_ids, 'episode.accuracy')  # placeholder

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # Panel 1: accuracy mean ± std
    ax = axes[0, 0]
    _plot_mean_std(ax, eps_acc, mat_acc, 'steelblue', 'Mean @Acc ± 1σ')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1.05); ax.set_xlabel('Episode'); ax.set_ylabel('@Acc')
    ax.set_title('Attack Accuracy (avg over seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: per-seed accuracy
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_ids)))
    for i, (rid, c) in enumerate(zip(run_ids, colors)):
        eps_i, vals_i = load_episode_series(db, rid, 'episode.accuracy')
        sm = smooth(vals_i, 100)
        pad = len(eps_i) - len(sm)
        ax.plot(eps_i[pad//2: pad//2+len(sm)], sm, color=c, linewidth=1.5,
                label=f'seed {i} (run {rid})')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1.05); ax.set_xlabel('Episode'); ax.set_ylabel('@Acc')
    ax.set_title('Per-seed accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: episode reward
    ax = axes[1, 0]
    if mat_rew is not None:
        _plot_mean_std(ax, eps_rew, mat_rew, 'forestgreen', 'Mean reward ± 1σ')
    ax.set_xlabel('Episode'); ax.set_ylabel('Cumulative Reward')
    ax.set_title('Episode Reward (avg over seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: DDPG loss
    ax = axes[1, 1]
    if mat_loss is not None:
        _plot_mean_std(ax, eps_loss, mat_loss, 'crimson', 'Mean loss ± 1σ',
                       smooth_w=50)
    ax.set_xlabel('Episode'); ax.set_ylabel('Loss')
    ax.set_title('DDPG Loss (avg over seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    n_ep = len(eps_acc)
    fig.suptitle(
        f'{label_prefix}  |  {len(run_ids)} seeds  |  {n_ep} episodes each  |  '
        f'mean final @Acc = {np.mean(mat_acc[:, -1]):.3f}',
        fontsize=13, fontweight='bold'
    )

    path = os.path.join(out_dir, 'dashboard_seeds.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def print_stats(db, run_ids):
    print(f'\n=== Multi-seed Summary (runs {run_ids}) ===')
    all_final, all_max, all_mean500 = [], [], []
    for rid in run_ids:
        eps, vals = load_episode_series(db, rid, 'episode.accuracy')
        if vals is None:
            continue
        run_doc = db.runs.find_one({'_id': rid}, {'config': 1})
        seed = run_doc.get('config', {}).get('seed', '?')
        final = vals[-1]
        maxv  = np.max(vals)
        m500  = np.mean(vals[-500:]) if len(vals) >= 500 else np.mean(vals)
        print(f'  run {rid} seed={seed}: episodes={len(vals)}  '
              f'final={final:.4f}  max={maxv:.4f}  avg_last500={m500:.4f}')
        all_final.append(final); all_max.append(maxv); all_mean500.append(m500)

    if all_final:
        print(f'\n  Across seeds:')
        print(f'    Mean final @Acc    : {np.mean(all_final):.4f} ± {np.std(all_final):.4f}')
        print(f'    Mean max @Acc      : {np.mean(all_max):.4f} ± {np.std(all_max):.4f}')
        print(f'    Mean avg_last500   : {np.mean(all_mean500):.4f} ± {np.std(all_mean500):.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_ids',      type=int, nargs='+', default=None)
    parser.add_argument('--encoder_mode', type=str, default='ae')
    parser.add_argument('--n',            type=int, default=None,
                        help='Latest N runs with given encoder_mode (if --run_ids not set)')
    parser.add_argument('--out',          type=str, default=None)
    parser.add_argument('--mongo_url',    type=str, default='localhost:27017')
    parser.add_argument('--db',           type=str, default='env_poisoning')
    args = parser.parse_args()

    db = connect(args.mongo_url, args.db)

    if args.run_ids:
        run_ids = args.run_ids
    else:
        runs = latest_ae_runs(db, encoder_mode=args.encoder_mode, n=args.n)
        run_ids = sorted([r['_id'] for r in runs])
        print(f'Auto-detected runs: {run_ids}')

    if not run_ids:
        print('No runs found.'); return

    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), 'figures',
        f'seeds_{"_".join(str(r) for r in run_ids)}'
    )
    os.makedirs(out_dir, exist_ok=True)

    label = f'{args.encoder_mode} encoder'
    print_stats(db, run_ids)

    print(f'\nGenerating figures → {out_dir}')
    fig_dashboard(db, run_ids, out_dir, label)
    fig_accuracy(db, run_ids, out_dir, label)
    fig_reward(db, run_ids, out_dir, label)
    fig_ddpg_loss(db, run_ids, out_dir, label)
    print('\nDone.')


if __name__ == '__main__':
    main()
