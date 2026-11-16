#!/bin/bash
# Larger grids: 5x5, 6x6 and 8x8 with three push-search settings, 12,000 episodes,
# 4 seeds each. Needs the autoencoders for those sizes (shipped in ae/models/, or
# train with `python -m ae.train_ae --grid_size N`).
#
#     sbatch --account=<acct> --array=0-35 scripts/slurm/sweep_grid_sizes.sh
#
#SBATCH --job-name=gridsizes
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

SIZES=(5 6 8)
NSEEDS=4
CONFIGS=(
  "floor   explore_push_prob=0.2"
  "push40  explore_push_prob=0.4"
  "slow    explore_push_prob=0.4 cem_batch=32 cem_elite=8 cem_lr=0.3"
)
T=$SLURM_ARRAY_TASK_ID
N=${SIZES[$((T / (3 * NSEEDS)))]}
CFG=(${CONFIGS[$(((T / NSEEDS) % 3))]})
SEED=$((T % NSEEDS))
train_and_score grid${N}_${CFG[0]}_s${SEED} \
    env=grid3d grid_size=$N victim_algo=qlearning encoder_mode=ae reward_mode=trace \
    obs_dynamics=env push_cem=True cem_floor=-1 "${CFG[@]:1}" max_episodes=12000 seed=$SEED
