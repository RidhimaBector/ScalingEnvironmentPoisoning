"""Visualize victim policy before and after poisoning.

3-panel figure:
  Panel 1 — Target policy (Mp path)
  Panel 2 — Natural victim (trained on unperturbed env)
  Panel 3 — Attacked victim (DDPG best_model, averaged over seeds)
  Panel 4 — Altitude change (attacked − natural)

Usage:
    python demo_policy.py --run_dirs results/rate0.001_seed0 results/rate0.001_seed1 results/rate0.001_seed2
    python demo_policy.py --run_dirs results/my_run --natural_episodes 12000
    python demo_policy.py --run_dirs results/my_run --out figures/my_run
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ae.observation_encoder import ObservationEncoder
from ae.victim_encoder import VictimEncoder
from agent.victim_system import VictimSystem
from attack.DDPG import DDPG
from envs.env3D_4x4 import Grid3D
from envs.target_def import create_target_policy
from utils.utils_attack import Attack_Done_Identify
from victim.victim_Q import VictimQLearning

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default run dirs — override via --run_dirs
DEFAULT_RUN_DIRS = [
    'results/sacred_run_34_2026-05-09_10-34-22',
]

DDPG_KWARGS = dict(
    seed=0, max_action=1.0,
    hidden1=400, hidden2=300, init_w=0.003,
    prate=1e-4, rate=1e-3,
    ou_theta=0.15, ou_mu=0.0, ou_sigma=0.2,
    bsize=256, tau=0.005, discount=0.9,
    epsilon_divisor=1000, eps_greedy_start_episodes=0,
    is_training=False,
)

# Arrow offsets in imshow coords (row=0 at top, y increases down)
# Actions: NORTH=0, EAST=1, SOUTH=2, WEST=3
ACTION_DX = [0.0,  0.27,  0.0, -0.27]
ACTION_DY = [-0.27, 0.0,  0.27,  0.0]

# ---------------------------------------------------------------------------
# Victim training helpers
# ---------------------------------------------------------------------------

def train_natural_victim(num_episodes: int):
    """Train Q-learning victim on the unperturbed Grid3D."""
    env = Grid3D()
    victim = VictimQLearning(env.nS, env.nA, memory_size=2000)
    print(f'  Training natural victim ({num_episodes} episodes)...')
    victim.train(env, num_episodes)
    target = create_target_policy(env.nS, env.nA)
    _, acc, _, _ = Attack_Done_Identify(target, victim.get_policy_matrix())
    print(f'  Natural victim accuracy = {acc:.4f}')
    return victim.get_policy_matrix(), acc, env.altitude.copy()


def run_attack_episode(run_dir: str, max_timesteps: int, victim_episodes: int):
    """Load saved DDPG best_model, run one attack episode, return Q-table."""
    env = Grid3D()
    victim = VictimQLearning(env.nS, env.nA, memory_size=2000)

    pop = VictimSystem(env=env, algorithm=victim, config=None)
    inner = VictimEncoder.from_population(pop)
    enc = ObservationEncoder(inner)

    ddpg = DDPG(nb_states=enc.embedding_dim, nb_actions=env.nS, **DDPG_KWARGS)
    model_path = os.path.join(SCRIPT_DIR, run_dir, 'best_model')
    if not os.path.exists(model_path + '_actor'):
        raise FileNotFoundError(f'No saved model at {model_path}')
    ddpg.load(model_path)
    ddpg.is_training = False

    obs = enc.get_initial_embedding()
    for _ in range(max_timesteps):
        action = ddpg.select_on_policy_action(obs)
        env.apply_perturbation(action)
        victim.train(env, victim_episodes)
        obs = enc.encode(
            [{'behavior_trace': victim.get_behavior_trace(),
              'q_table': victim.get_policy_matrix(),
              'dynamics': env.get_dynamics()}],
            env_dynamics=env.get_dynamics(),
        )

    target = create_target_policy(env.nS, env.nA)
    _, acc, _, _ = Attack_Done_Identify(target, victim.get_policy_matrix())
    return victim.get_policy_matrix(), acc, env.altitude.copy()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _altitude_to_rgb(altitude: np.ndarray) -> np.ndarray:
    """Convert altitude array [0-10] to RGB (blue=low, yellow=mid, red=high)."""
    norm = altitude / 10.0
    rgb = np.zeros((*altitude.shape, 3), dtype=float)
    rgb[:, :, 0] = np.clip(2 * norm - 1, 0, 1)
    rgb[:, :, 1] = np.clip(1 - np.abs(2 * norm - 1), 0, 1)
    rgb[:, :, 2] = np.clip(1 - 2 * norm, 0, 1)
    return rgb


def draw_panel(ax, greedy, tgt_greedy, altitude, title, accuracy=None):
    """Draw one policy panel: altitude heatmap + correct/wrong overlays + arrows."""
    nrows, ncols = altitude.shape
    nS = nrows * ncols
    rgb = _altitude_to_rgb(altitude)

    extent = [-0.5, ncols - 0.5, nrows - 0.5, -0.5]
    ax.imshow(rgb, extent=extent, zorder=1, aspect='equal')

    # Green = matches target, red = wrong, grey = no target defined
    overlay = np.zeros((nrows, ncols, 4), dtype=float)
    for s in range(nS):
        r, c = divmod(s, ncols)
        tgt = tgt_greedy[s]
        if tgt >= 0:
            overlay[r, c] = [0.0, 0.85, 0.2, 0.45] if greedy[s] == tgt else [0.9, 0.1, 0.1, 0.45]
        else:
            overlay[r, c] = [0.6, 0.6, 0.6, 0.20]
    ax.imshow(overlay, extent=extent, zorder=2, aspect='equal')

    # Grid lines
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1.0, zorder=3)
    for i in range(ncols + 1):
        ax.axvline(i - 0.5, color='black', linewidth=1.0, zorder=3)

    # Altitude numbers (top-right of each cell)
    for r in range(nrows):
        for c in range(ncols):
            ax.text(c + 0.38, r - 0.38, f'{altitude[r, c]:.0f}',
                    fontsize=7, color='white', ha='right', va='top',
                    fontweight='bold', zorder=5)

    # Policy arrows
    for s in range(nS):
        a = greedy[s]
        if a < 0:
            continue
        r, c = divmod(s, ncols)
        ax.annotate(
            '', xy=(c + ACTION_DX[a], r + ACTION_DY[a]),
            xytext=(c, r),
            arrowprops=dict(arrowstyle='->', color='white', lw=2.2),
            zorder=4,
        )

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])

    full_title = title
    if accuracy is not None:
        full_title += f'\naccuracy = {accuracy:.3f}'
    ax.set_title(full_title, fontsize=11, fontweight='bold')


def draw_altitude_diff_panel(ax, alt_natural, alt_attacked, title):
    """Show per-cell altitude change: attacked − natural."""
    diff = alt_attacked - alt_natural
    nrows, ncols = diff.shape
    vmax = max(abs(diff).max(), 0.01)
    extent = [-0.5, ncols - 0.5, nrows - 0.5, -0.5]
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   extent=extent, aspect='equal')
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1.0)
    for i in range(ncols + 1):
        ax.axvline(i - 0.5, color='black', linewidth=1.0)
    for r in range(nrows):
        for c in range(ncols):
            ax.text(c, r, f'{diff[r, c]:+.1f}', ha='center', va='center',
                    fontsize=8, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize poisoned vs natural victim policy')
    parser.add_argument('--run_dirs',          nargs='+', default=DEFAULT_RUN_DIRS,
                        help='Paths to result dirs containing best_model (one per seed)')
    parser.add_argument('--natural_episodes',  type=int,  default=12000,
                        help='Episodes to train the natural (unpoisoned) victim')
    parser.add_argument('--max_timesteps',     type=int,  default=15,
                        help='Attack timesteps (should match training)')
    parser.add_argument('--victim_episodes',   type=int,  default=80,
                        help='Victim training episodes per attack step')
    parser.add_argument('--out',               type=str,  default='figures',
                        help='Output directory for the figure')
    args = parser.parse_args()

    os.makedirs(os.path.join(SCRIPT_DIR, args.out), exist_ok=True)

    env_ref = Grid3D()
    nS, nA = env_ref.nS, env_ref.nA
    natural_alt = env_ref.altitude.copy()

    # Target greedy actions (-1 = no preference at that state)
    tgt_mat = create_target_policy(nS, nA)
    tgt_greedy = np.argmax(tgt_mat, axis=1)
    tgt_greedy[np.max(tgt_mat, axis=1) == 0] = -1

    # --- Natural victim ---
    print('\n[1/2] Natural victim')
    nat_Q, nat_acc, _ = train_natural_victim(args.natural_episodes)
    nat_greedy = np.argmax(nat_Q, axis=1)

    # --- Attacked victim (averaged over seeds) ---
    print(f'\n[2/2] Attacked victim  ({len(args.run_dirs)} seed(s))')
    Q_list, acc_list, alt_list = [], [], []
    for i, run_dir in enumerate(args.run_dirs):
        print(f'  seed {i}: {run_dir}')
        try:
            Q, acc, alt = run_attack_episode(run_dir, args.max_timesteps, args.victim_episodes)
            Q_list.append(Q); acc_list.append(acc); alt_list.append(alt)
            print(f'  → accuracy = {acc:.4f}')
        except FileNotFoundError as e:
            print(f'  SKIPPED: {e}')

    if not Q_list:
        print('\nNo valid run dirs found — showing natural victim only.')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        draw_panel(axes[0], tgt_greedy, tgt_greedy, natural_alt, 'Target Policy (Mp)', accuracy=1.0)
        draw_panel(axes[1], nat_greedy, tgt_greedy, natural_alt,
                   f'Natural Victim ({args.natural_episodes} eps)', accuracy=nat_acc)
    else:
        avg_Q   = np.mean(Q_list, axis=0)
        avg_alt = np.mean(alt_list, axis=0)
        atk_greedy = np.argmax(avg_Q, axis=1)
        _, atk_acc, _, _ = Attack_Done_Identify(tgt_mat, avg_Q)
        print(f'\n  Mean accuracy = {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.subplots_adjust(wspace=0.08)

        draw_panel(axes[0], tgt_greedy, tgt_greedy, natural_alt,
                   'Target Policy (Mp)', accuracy=1.0)
        draw_panel(axes[1], nat_greedy, tgt_greedy, natural_alt,
                   f'Natural Victim\n({args.natural_episodes} eps)', accuracy=nat_acc)
        draw_panel(axes[2], atk_greedy, tgt_greedy, natural_alt,
                   f'Attacked Victim\n({len(Q_list)} seed(s))', accuracy=atk_acc)
        draw_altitude_diff_panel(axes[3], natural_alt, avg_alt,
                                 'Altitude Change\n(attacked − natural)')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0.0, 0.85, 0.2], alpha=0.7, label='Matches target'),
            Patch(facecolor=[0.9, 0.1, 0.1], alpha=0.7, label='Wrong action'),
            Patch(facecolor=[0.6, 0.6, 0.6], alpha=0.5, label='No target (free)'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=10, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(
            'Environment Poisoning — Natural vs Attacked Victim\n'
            f'natural acc={nat_acc:.3f}   attacked acc={atk_acc:.3f}  '
            f'({len(Q_list)} seed(s))',
            fontsize=13, fontweight='bold', y=1.02,
        )

    out_path = os.path.join(SCRIPT_DIR, args.out, 'policy_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
