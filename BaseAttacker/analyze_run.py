"""
Analysis and figure generation for Sacred/MongoDB experiment runs.

Usage:
    python analyze_run.py                  # latest COMPLETED or FAILED run
    python analyze_run.py --run_id 27      # specific run
    python analyze_run.py --run_id 27 --out figures/run27
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def connect(mongo_url='localhost:27017', db_name='env_poisoning'):
    client = MongoClient(mongo_url)
    return client[db_name]


def get_metric(db, run_id, name):
    doc = db.metrics.find_one({'run_id': run_id, 'name': name})
    if doc is None:
        return None, None
    return np.array(doc['steps']), np.array(doc['values'])


def get_run_info(db, run_id):
    return db.runs.find_one({'_id': run_id})


def latest_run_id(db):
    run = db.runs.find_one(sort=[('_id', -1)])
    return run['_id'] if run else None


def smooth(values, window=50):
    if len(values) < window:
        window = max(1, len(values) // 5)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def fig_accuracy(db, run_id, out_dir, config):
    ep_steps, ep_acc   = get_metric(db, run_id, 'episode.accuracy')
    _,         ep_min  = get_metric(db, run_id, 'episode.min_accuracy')
    _,         ep_sftmx = get_metric(db, run_id, 'episode.accuracy')

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw accuracy (faint)
    ax.plot(ep_steps, ep_acc, alpha=0.2, color='steelblue', linewidth=0.5)

    # Smoothed accuracy
    w = 100
    sm_acc = smooth(ep_acc, w)
    sm_min = smooth(ep_min, w)
    pad = len(ep_steps) - len(sm_acc)
    xs = ep_steps[pad // 2: pad // 2 + len(sm_acc)]

    ax.plot(xs, sm_acc, color='steelblue', linewidth=2, label='Mean @Acc (smoothed)')
    ax.fill_between(xs, sm_min, sm_acc, alpha=0.15, color='steelblue', label='Min–Mean band')

    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Perfect accuracy')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Attack Accuracy (@Acc)')
    ax.set_title(f'Attack Accuracy over Training  |  run={run_id}  '
                 f'victims={config.get("num_victims")}  algo={config.get("attack_algo")}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_softmax_accuracy(db, run_id, out_dir, config):
    ts_steps, ts_acc    = get_metric(db, run_id, 'timestep.accuracy')
    _,         ts_sftmx = get_metric(db, run_id, 'timestep.mean_accuracy_sftmx')

    # Aggregate to per-episode: use last timestep of each episode (step % 1000 == 14)
    mask = ts_steps % 1000 == 14
    ep_nums   = ts_steps[mask] // 1000
    acc_vals  = ts_acc[mask]
    sftmx_vals = ts_sftmx[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 100
    sm_acc   = smooth(acc_vals, w)
    sm_sftmx = smooth(sftmx_vals, w)
    pad = len(ep_nums) - len(sm_acc)
    xs = ep_nums[pad // 2: pad // 2 + len(sm_acc)]

    ax.plot(xs, sm_acc,   color='steelblue', linewidth=2, label='@Acc')
    ax.plot(xs, sm_sftmx, color='darkorange', linewidth=2, label='@SoftAcc')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'@Acc vs @SoftAcc  |  run={run_id}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'acc_vs_softacc.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_ddpg_loss(db, run_id, out_dir, config):
    steps, values = get_metric(db, run_id, 'ddpg_loss')
    if steps is None:
        print('  No ddpg_loss metric found, skipping.')
        return

    # ddpg_loss values alternate: each step stores a flat list [critic_loss, actor_loss]
    # or it may be just a single float — detect
    if np.ndim(values[0]) == 0:
        # single value per step — treat as combined loss
        fig, ax = plt.subplots(figsize=(10, 4))
        sm = smooth(values, 50)
        pad = len(steps) - len(sm)
        ax.plot(steps[pad // 2: pad // 2 + len(sm)], sm, color='crimson', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'DDPG Loss  |  run={run_id}')
        ax.grid(True, alpha=0.3)
    else:
        values = np.array([v for v in values])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, (ax, label, color) in enumerate(zip(
                axes, ['Critic Loss', 'Actor Loss'], ['crimson', 'purple'])):
            col = values[:, i]
            sm  = smooth(col, 50)
            pad = len(steps) - len(sm)
            ax.plot(steps[pad // 2: pad // 2 + len(sm)], sm, color=color, linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title(f'{label}  |  run={run_id}')
            ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'ddpg_loss.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_episode_reward(db, run_id, out_dir, config):
    steps, values = get_metric(db, run_id, 'episode.episode_reward')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, values, alpha=0.2, color='forestgreen', linewidth=0.5)
    sm  = smooth(values, 100)
    pad = len(steps) - len(sm)
    ax.plot(steps[pad // 2: pad // 2 + len(sm)], sm,
            color='forestgreen', linewidth=2, label='Episode reward (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Episode Reward  |  run={run_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'episode_reward.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_summary_dashboard(db, run_id, out_dir, config):
    """4-panel summary figure."""
    ep_steps, ep_acc = get_metric(db, run_id, 'episode.accuracy')
    _,  ep_min       = get_metric(db, run_id, 'episode.min_accuracy')
    _,  ep_rew       = get_metric(db, run_id, 'episode.episode_reward')
    loss_steps, loss = get_metric(db, run_id, 'ddpg_loss')
    ts_steps, ts_acc = get_metric(db, run_id, 'timestep.accuracy')
    _,  ts_sftmx     = get_metric(db, run_id, 'timestep.mean_accuracy_sftmx')

    mask = ts_steps % 1000 == 14
    ep_nums    = ts_steps[mask] // 1000
    sftmx_vals = ts_sftmx[mask]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep_steps, ep_acc, alpha=0.2, color='steelblue', linewidth=0.5)
    w = 100
    sm_acc = smooth(ep_acc, w)
    sm_min = smooth(ep_min, w)
    pad = len(ep_steps) - len(sm_acc)
    xs  = ep_steps[pad // 2: pad // 2 + len(sm_acc)]
    ax1.plot(xs, sm_acc, color='steelblue', linewidth=2, label='Mean @Acc')
    ax1.fill_between(xs, sm_min, sm_acc, alpha=0.15, color='steelblue')
    ax1.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('@Acc')
    ax1.set_title('Attack Accuracy'); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # --- Panel 2: @Acc vs @SoftAcc ---
    ax2 = fig.add_subplot(gs[0, 1])
    sm_sftmx = smooth(sftmx_vals, w)
    sm_acc2  = smooth(ep_acc, w)
    pad2 = len(ep_nums) - len(sm_sftmx)
    xs2  = ep_nums[pad2 // 2: pad2 // 2 + len(sm_sftmx)]
    ax2.plot(xs2, sm_acc2[:len(xs2)], color='steelblue', linewidth=2, label='@Acc')
    ax2.plot(xs2, sm_sftmx,          color='darkorange', linewidth=2, label='@SoftAcc')
    ax2.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Accuracy')
    ax2.set_title('@Acc vs @SoftAcc'); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # --- Panel 3: Episode Reward ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep_steps, ep_rew, alpha=0.2, color='forestgreen', linewidth=0.5)
    sm_rew = smooth(ep_rew, w)
    pad3 = len(ep_steps) - len(sm_rew)
    xs3  = ep_steps[pad3 // 2: pad3 // 2 + len(sm_rew)]
    ax3.plot(xs3, sm_rew, color='forestgreen', linewidth=2)
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Episode Reward'); ax3.grid(True, alpha=0.3)

    # --- Panel 4: DDPG Loss ---
    ax4 = fig.add_subplot(gs[1, 1])
    if loss is not None:
        if np.ndim(loss[0]) == 0:
            sm_loss = smooth(loss, 50)
            pad4 = len(loss_steps) - len(sm_loss)
            ax4.plot(loss_steps[pad4 // 2: pad4 // 2 + len(sm_loss)], sm_loss,
                     color='crimson', linewidth=1.5)
        else:
            loss_arr = np.array([v for v in loss])
            for col_i, (label, color) in enumerate(
                    zip(['Critic', 'Actor'], ['crimson', 'purple'])):
                col = loss_arr[:, col_i]
                sm  = smooth(col, 50)
                pad4 = len(loss_steps) - len(sm)
                ax4.plot(loss_steps[pad4 // 2: pad4 // 2 + len(sm)], sm,
                         color=color, linewidth=1.5, label=label)
            ax4.legend(fontsize=8)
    ax4.set_xlabel('Episode'); ax4.set_ylabel('Loss')
    ax4.set_title('DDPG Loss'); ax4.grid(True, alpha=0.3)

    # Title
    n_ep = int(ep_steps[-1]) + 1
    fig.suptitle(
        f'Run {run_id}  |  {config.get("attack_algo","ddpg").upper()}  |  '
        f'{config.get("num_victims")} victim(s)  |  {n_ep} episodes  |  '
        f'max @Acc = {np.max(ep_acc):.3f}',
        fontsize=13, fontweight='bold'
    )

    path = os.path.join(out_dir, 'dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def print_stats(db, run_id, config):
    ep_steps, ep_acc = get_metric(db, run_id, 'episode.accuracy')
    _, ep_min        = get_metric(db, run_id, 'episode.min_accuracy')
    _, ep_rew        = get_metric(db, run_id, 'episode.episode_reward')

    n_ep = len(ep_steps)
    print(f'\n=== Run {run_id} Summary ===')
    print(f'  Episodes completed : {n_ep}')
    print(f'  Victims            : {config.get("num_victims")}')
    print(f'  Algorithm          : {config.get("attack_algo")}')
    print(f'  Victim algo        : {config.get("victim_algo")}')
    print(f'  Max @Acc           : {np.max(ep_acc):.4f}')
    print(f'  Final @Acc         : {ep_acc[-1]:.4f}')
    print(f'  Avg @Acc (last 500): {np.mean(ep_acc[-500:]):.4f}')
    print(f'  Max episode reward : {np.max(ep_rew):.4f}')
    print(f'  Avg reward (last 500): {np.mean(ep_rew[-500:]):.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id',    type=int,   default=None)
    parser.add_argument('--out',       type=str,   default=None)
    parser.add_argument('--mongo_url', type=str,   default='localhost:27017')
    parser.add_argument('--db',        type=str,   default='env_poisoning')
    args = parser.parse_args()

    db = connect(args.mongo_url, args.db)

    run_id = args.run_id or latest_run_id(db)
    print(f'Analyzing run {run_id}...')

    run_info = get_run_info(db, run_id)
    config   = run_info.get('config', {})

    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), 'figures', f'run_{run_id}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print_stats(db, run_id, config)

    print(f'\nGenerating figures → {out_dir}')
    fig_summary_dashboard(db, run_id, out_dir, config)
    fig_accuracy(db, run_id, out_dir, config)
    fig_softmax_accuracy(db, run_id, out_dir, config)
    fig_episode_reward(db, run_id, out_dir, config)
    fig_ddpg_loss(db, run_id, out_dir, config)

    print('\nDone.')


if __name__ == '__main__':
    main()
