"""Plot accuracy vs episode for the learning-rate sweep, grouped by attack_rate.

Queries MongoDB for all runs matching the sweep config and overlays them.

Usage:
    python plot_lr_sweep.py
    python plot_lr_sweep.py --rates 0.0001 0.001 0.01
    python plot_lr_sweep.py --run_ids 45 46 47        # specific run IDs
    python plot_lr_sweep.py --out figures/lr_sweep
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pymongo import MongoClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORS = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']


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


def load_runs(db, rates=None, run_ids=None, encoder_mode=None):
    """Return dict: attack_rate -> list of (steps, accuracy) arrays."""
    if run_ids:
        runs = [db.runs.find_one({'_id': rid}) for rid in run_ids]
        runs = [r for r in runs if r]
    else:
        if rates is None:
            rates = [0.0001, 0.001, 0.01]
        query = {'config.attack_rate': {'$in': [float(r) for r in rates]}}
        if encoder_mode:
            query['config.encoder_mode'] = encoder_mode
        runs = list(db.runs.find(query))

    data = defaultdict(list)
    for run in runs:
        cfg = run.get('config', {})
        rate = cfg.get('attack_rate', '?')
        run_id = run['_id']
        status = run.get('status')
        if status and status != 'COMPLETED':
            print(f'  run {run_id} (rate={rate}): status={status}, skipping')
            continue
        steps, acc = get_metric(db, run_id, 'episode.accuracy')
        if steps is None or len(steps) == 0:
            print(f'  run {run_id} (rate={rate}): no accuracy data yet, skipping')
            continue
        print(f'  run {run_id}  rate={rate}  seed={cfg.get("seed")}  episodes={len(steps)}  max_acc={max(acc):.3f}')
        data[rate].append((steps, acc))
    return data


def plot_sweep(data, out_path, smooth_window=100):
    if not data:
        print('No data found.')
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (rate, runs) in enumerate(sorted((k, v) for k, v in data.items() if k != '?')):
        color = COLORS[i % len(COLORS)]
        label = f'rate={rate}'

        # Plot individual seeds faint
        all_smoothed = []
        for steps, acc in runs:
            sm = smooth(acc, smooth_window)
            pad = len(steps) - len(sm)
            xs = steps[pad // 2: pad // 2 + len(sm)]
            ax.plot(xs, sm, color=color, linewidth=0.8, alpha=0.3)
            all_smoothed.append((xs, sm))

        # Mean ± std band across seeds (interpolate to common x grid).
        # Ignore short/partial runs so one interrupted seed can't drop the whole rate.
        max_len = max(len(xs) for xs, _ in all_smoothed)
        long_runs = [(xs, sm) for xs, sm in all_smoothed if len(xs) >= max_len * 0.9]
        if len(all_smoothed) > 1:
            x_min = max(xs[0]  for xs, _ in long_runs)
            x_max = min(xs[-1] for xs, _ in long_runs)
            if x_min < x_max:
                grid = np.arange(x_min, x_max + 1)
                interp = np.array([np.interp(grid, xs, sm) for xs, sm in long_runs])
                mean = interp.mean(axis=0)
                std  = interp.std(axis=0)
                ax.plot(grid, mean, color=color, linewidth=2.5,
                        label=f'{label}  (n={len(long_runs)}, peak={mean.max():.3f})')
                ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.15)
        else:
            xs, sm = all_smoothed[0]
            ax.plot(xs, sm, color=color, linewidth=2.5,
                    label=f'{label}  (peak={sm.max():.3f})')

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Perfect')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Attack Accuracy', fontsize=12)
    ax.set_title('LR Sweep — Attack Accuracy by Attacker Learning Rate\n'
                 f'(smoothed window={smooth_window}, shaded = ±1 std across seeds)',
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rates',    nargs='+', default=None,
                        help='Filter by attack_rate values (default: 0.0001 0.001 0.01)')
    parser.add_argument('--encoder_mode', type=str, default=None,
                        help='Filter by encoder_mode (e.g. whitebox, lstm, ae)')
    parser.add_argument('--run_ids',  nargs='+', type=int, default=None,
                        help='Specific Sacred run IDs to include')
    parser.add_argument('--smooth',   type=int,  default=100,
                        help='Smoothing window size')
    parser.add_argument('--out',      type=str,
                        default=os.path.join(SCRIPT_DIR, 'figures', 'lr_sweep.png'))
    parser.add_argument('--mongo_url', type=str, default='localhost:27017')
    parser.add_argument('--db',        type=str, default='env_poisoning')
    args = parser.parse_args()

    db = connect(args.mongo_url, args.db)
    print('Loading runs...')
    data = load_runs(db, rates=args.rates, run_ids=args.run_ids,
                     encoder_mode=args.encoder_mode)
    plot_sweep(data, args.out, smooth_window=args.smooth)


if __name__ == '__main__':
    main()
