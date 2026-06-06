"""Analysis and figure generation for CSV-logged runs (pre-MongoDB / main-branch format).

CSV files in data/ have columns:
  accuracy_buffer.csv          [episode, timestep, accuracy]
  accuracy_sftmx_buffer.csv    [episode, timestep, accuracy_sftmx]
  effort_buffer.csv            [episode, timestep, effort]
  distance_buffer_K.csv        [episode, timestep, distance_K]
  distance_buffer_W.csv        [episode, timestep, distance_W]
  distance_grid_buffer_K.csv   [episode, timestep, distance_grid_K]
  distance_grid_buffer_W.csv   [episode, timestep, distance_grid_W]
  distance_behavior_buffer_K.csv  [episode, timestep, distance_behavior_K]
  distance_behavior_buffer_W.csv  [episode, timestep, distance_behavior_W]
  ddpg_loss.csv                [episode, timestep, critic_loss, actor_loss]

Usage:
    python analyze_csv.py
    python analyze_csv.py --data_dir data --out figures/csv_run
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a CSV file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=',')


def per_episode(data, col=2):
    """Extract per-episode values by taking the last timestep of each episode.

    Args:
        data: Array of shape (N, >=3) with col 0 = episode number.
        col: Which column holds the metric value.

    Returns:
        Tuple (episodes, values) as 1D arrays.
    """
    episodes = data[:, 0].astype(int)
    values = data[:, col]
    # Group by episode and take the last occurrence
    unique_eps = np.unique(episodes)
    ep_vals = np.array([values[episodes == ep][-1] for ep in unique_eps])
    return unique_eps, ep_vals


def episode_rewards(acc_data):
    """Compute cumulative reward per episode (sum of per-timestep accuracy)."""
    episodes = acc_data[:, 0].astype(int)
    values = acc_data[:, 2]
    unique_eps = np.unique(episodes)
    rewards = np.array([values[episodes == ep].sum() for ep in unique_eps])
    return unique_eps, rewards


def smooth(values, window=100):
    if len(values) < window:
        window = max(1, len(values) // 5)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def smoothed_xs(ep_array, sm_array):
    """Return the x-axis positions aligned with a smoothed series."""
    pad = len(ep_array) - len(sm_array)
    return ep_array[pad // 2: pad // 2 + len(sm_array)]


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def fig_accuracy(acc_data, out_dir, label='CSV run'):
    ep_steps, ep_acc = per_episode(acc_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep_steps, ep_acc, alpha=0.2, color='steelblue', linewidth=0.5)
    w = 100
    sm = smooth(ep_acc, w)
    xs = smoothed_xs(ep_steps, sm)
    ax.plot(xs, sm, color='steelblue', linewidth=2, label='@Acc (smoothed)')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Perfect accuracy')
    ax.set_xlabel('Episode')
    ax.set_ylabel('@Acc')
    ax.set_title(f'Attack Accuracy  |  {label}  |  max={ep_acc.max():.3f}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_softmax_accuracy(acc_data, sftmx_data, out_dir, label='CSV run'):
    ep_steps, ep_acc = per_episode(acc_data)
    _, ep_sftmx = per_episode(sftmx_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 100
    sm_acc = smooth(ep_acc, w)
    sm_sftmx = smooth(ep_sftmx, w)
    xs = smoothed_xs(ep_steps, sm_acc)
    xs2 = smoothed_xs(ep_steps, sm_sftmx)
    ax.plot(xs, sm_acc, color='steelblue', linewidth=2, label='@Acc')
    ax.plot(xs2, sm_sftmx, color='darkorange', linewidth=2, label='@SoftAcc')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'@Acc vs @SoftAcc  |  {label}')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'acc_vs_softacc.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_ddpg_loss(loss_data, out_dir, label='CSV run'):
    # columns: [episode, timestep, critic_loss, actor_loss]
    eps = loss_data[:, 0].astype(int)
    unique_eps = np.unique(eps)
    critic = np.array([loss_data[eps == ep, 2].mean() for ep in unique_eps])
    actor  = np.array([loss_data[eps == ep, 3].mean() for ep in unique_eps])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, name, color in zip(axes, [critic, actor], ['Critic Loss', 'Actor Loss'], ['crimson', 'purple']):
        sm = smooth(col, 50)
        xs = smoothed_xs(unique_eps, sm)
        ax.plot(xs, sm, color=color, linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name}  |  {label}')
        ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'ddpg_loss.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_episode_reward(acc_data, out_dir, label='CSV run'):
    ep_steps, rewards = episode_rewards(acc_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_steps, rewards, alpha=0.2, color='forestgreen', linewidth=0.5)
    sm = smooth(rewards, 100)
    xs = smoothed_xs(ep_steps, sm)
    ax.plot(xs, sm, color='forestgreen', linewidth=2, label='Episode reward (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'Episode Reward  |  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'episode_reward.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_distance_metrics(dist_K_data, dist_W_data, out_dir, label='CSV run'):
    ep_steps, ep_K = per_episode(dist_K_data)
    _, ep_W = per_episode(dist_W_data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = 100
    for ax, vals, name, color in zip(
        axes,
        [ep_K, ep_W],
        ['KL-Rate Distance', 'Wasserstein Distance'],
        ['teal', 'darkorange'],
    ):
        sm = smooth(vals, w)
        xs = smoothed_xs(ep_steps, sm)
        ax.plot(ep_steps, vals, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(xs, sm, color=color, linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distance (negated)')
        ax.set_title(f'{name}  |  {label}')
        ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'distance_metrics.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_effort(effort_data, out_dir, label='CSV run'):
    ep_steps, ep_effort = per_episode(effort_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_steps, ep_effort, alpha=0.2, color='sienna', linewidth=0.5)
    sm = smooth(ep_effort, 100)
    xs = smoothed_xs(ep_steps, sm)
    ax.plot(xs, sm, color='sienna', linewidth=2, label='Effort (smoothed)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Effort (negated)')
    ax.set_title(f'Attack Effort  |  {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, 'effort.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def fig_summary_dashboard(acc_data, sftmx_data, loss_data, effort_data,
                          dist_K_data, dist_W_data, out_dir, label='CSV run'):
    ep_steps, ep_acc   = per_episode(acc_data)
    _,        ep_sftmx = per_episode(sftmx_data)
    _,        rewards  = episode_rewards(acc_data)
    _,        ep_eff   = per_episode(effort_data)

    loss_eps = loss_data[:, 0].astype(int)
    unique_loss_eps = np.unique(loss_eps)
    critic = np.array([loss_data[loss_eps == ep, 2].mean() for ep in unique_loss_eps])
    actor  = np.array([loss_data[loss_eps == ep, 3].mean() for ep in unique_loss_eps])

    w = 100
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # --- Panel 1: Accuracy ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep_steps, ep_acc, alpha=0.15, color='steelblue', linewidth=0.5)
    sm = smooth(ep_acc, w); xs = smoothed_xs(ep_steps, sm)
    ax1.plot(xs, sm, color='steelblue', linewidth=2, label='@Acc')
    ax1.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('@Acc')
    ax1.set_title('Attack Accuracy'); ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # --- Panel 2: @Acc vs @SoftAcc ---
    ax2 = fig.add_subplot(gs[0, 1])
    sm_acc   = smooth(ep_acc, w);   xs_acc   = smoothed_xs(ep_steps, sm_acc)
    sm_sftmx = smooth(ep_sftmx, w); xs_sftmx = smoothed_xs(ep_steps, sm_sftmx)
    ax2.plot(xs_acc,   sm_acc,   color='steelblue',  linewidth=2, label='@Acc')
    ax2.plot(xs_sftmx, sm_sftmx, color='darkorange', linewidth=2, label='@SoftAcc')
    ax2.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Accuracy')
    ax2.set_title('@Acc vs @SoftAcc'); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # --- Panel 3: Episode Reward ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep_steps, rewards, alpha=0.15, color='forestgreen', linewidth=0.5)
    sm = smooth(rewards, w); xs = smoothed_xs(ep_steps, sm)
    ax3.plot(xs, sm, color='forestgreen', linewidth=2)
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Episode Reward'); ax3.grid(True, alpha=0.3)

    # --- Panel 4: DDPG Loss ---
    ax4 = fig.add_subplot(gs[1, 1])
    for col, name, color in zip([critic, actor], ['Critic', 'Actor'], ['crimson', 'purple']):
        sm = smooth(col, 50); xs = smoothed_xs(unique_loss_eps, sm)
        ax4.plot(xs, sm, color=color, linewidth=1.5, label=name)
    ax4.set_xlabel('Episode'); ax4.set_ylabel('Loss')
    ax4.set_title('DDPG Loss'); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # --- Panel 5: Distance metrics (K and W) ---
    ax5 = fig.add_subplot(gs[2, 0])
    _, ep_K = per_episode(dist_K_data)
    _, ep_W = per_episode(dist_W_data)
    sm_K = smooth(ep_K, w); xs_K = smoothed_xs(ep_steps, sm_K)
    sm_W = smooth(ep_W, w); xs_W = smoothed_xs(ep_steps, sm_W)
    ax5.plot(xs_K, sm_K, color='teal',       linewidth=2, label='KL-Rate')
    ax5.plot(xs_W, sm_W, color='darkorange', linewidth=2, label='Wasserstein')
    ax5.set_xlabel('Episode'); ax5.set_ylabel('Distance (negated)')
    ax5.set_title('Distance to Target'); ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # --- Panel 6: Effort ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(ep_steps, ep_eff, alpha=0.15, color='sienna', linewidth=0.5)
    sm = smooth(ep_eff, w); xs = smoothed_xs(ep_steps, sm)
    ax6.plot(xs, sm, color='sienna', linewidth=2)
    ax6.set_xlabel('Episode'); ax6.set_ylabel('Effort (negated)')
    ax6.set_title('Attack Effort'); ax6.grid(True, alpha=0.3)

    n_ep = int(ep_steps[-1])
    fig.suptitle(
        f'{label}  |  DDPG  |  1 victim  |  {n_ep} episodes  |  max @Acc = {ep_acc.max():.3f}',
        fontsize=13, fontweight='bold'
    )

    path = os.path.join(out_dir, 'dashboard.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def print_stats(acc_data, loss_data, label='CSV run'):
    ep_steps, ep_acc = per_episode(acc_data)
    _, rewards = episode_rewards(acc_data)
    n_ep = int(ep_steps[-1])
    print(f'\n=== {label} Summary ===')
    print(f'  Episodes completed : {n_ep}')
    print(f'  Max @Acc           : {ep_acc.max():.4f}')
    print(f'  Final @Acc         : {ep_acc[-1]:.4f}')
    print(f'  Avg @Acc (last 500): {ep_acc[-500:].mean():.4f}')
    print(f'  Max episode reward : {rewards.max():.4f}')
    print(f'  Avg reward (last 500): {rewards[-500:].mean():.4f}')
    print(f'  First episode to reach 1.0 @Acc: ', end='')
    perfect = ep_steps[ep_acc >= 1.0]
    print(int(perfect[0]) if len(perfect) > 0 else 'never')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out', default=None)
    parser.add_argument('--label', default='CSV run (30k episodes)')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out or os.path.join(
        os.path.dirname(__file__), 'figures', 'csv_run'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading CSV data from {data_dir}/')
    acc_data    = load_csv(os.path.join(data_dir, 'accuracy_buffer.csv'))
    sftmx_data  = load_csv(os.path.join(data_dir, 'accuracy_sftmx_buffer.csv'))
    loss_data   = load_csv(os.path.join(data_dir, 'ddpg_loss.csv'))
    effort_data = load_csv(os.path.join(data_dir, 'effort_buffer.csv'))
    dist_K_data = load_csv(os.path.join(data_dir, 'distance_buffer_K.csv'))
    dist_W_data = load_csv(os.path.join(data_dir, 'distance_buffer_W.csv'))

    if acc_data is None:
        print(f'ERROR: accuracy_buffer.csv not found in {data_dir}/')
        return

    print_stats(acc_data, loss_data, label=args.label)

    print(f'\nGenerating figures → {out_dir}')
    fig_summary_dashboard(acc_data, sftmx_data, loss_data, effort_data,
                          dist_K_data, dist_W_data, out_dir, label=args.label)
    fig_accuracy(acc_data, out_dir, label=args.label)
    if sftmx_data is not None:
        fig_softmax_accuracy(acc_data, sftmx_data, out_dir, label=args.label)
    if loss_data is not None:
        fig_ddpg_loss(loss_data, out_dir, label=args.label)
    fig_episode_reward(acc_data, out_dir, label=args.label)
    if dist_K_data is not None and dist_W_data is not None:
        fig_distance_metrics(dist_K_data, dist_W_data, out_dir, label=args.label)
    if effort_data is not None:
        fig_effort(effort_data, out_dir, label=args.label)

    print('\nDone.')


if __name__ == '__main__':
    main()
