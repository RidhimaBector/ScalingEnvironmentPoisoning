#!/bin/bash
# One-change-at-a-time ablation on the 4x4 grid (3000 episodes, 3 seeds each).
# The base is plain DDPG with the trace reward and no push exploration.
#
#     sbatch --account=<acct> --array=0-17 scripts/slurm/sweep_ablation_4x4.sh
#
#SBATCH --job-name=ablation4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

BASE="env=grid3d victim_algo=qlearning encoder_mode=ae obs_dynamics=env max_episodes=3000"
NOPUSH="explore_push_prob=0 push_cem=False"
CONFIGS=(
  "base   reward_mode=trace $NOPUSH"
  "td3    reward_mode=trace $NOPUSH td3=True"
  "norm   reward_mode=trace $NOPUSH normalize_obs=True"
  "push   reward_mode=trace explore_push_prob=0.1 push_cem=False"
  "mix    reward_mode=mix   $NOPUSH"
  "noise  reward_mode=trace $NOPUSH expl_noise=0.2"
)
CFG=(${CONFIGS[$((SLURM_ARRAY_TASK_ID / 3))]})
SEED=$((SLURM_ARRAY_TASK_ID % 3))
train_and_score experiment=ablation-4x4 config_name=${CFG[0]} $BASE "${CFG[@]:1}" seed=$SEED
