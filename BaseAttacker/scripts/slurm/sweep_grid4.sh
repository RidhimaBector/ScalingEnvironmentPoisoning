#!/bin/bash
# Main result on the 4x4 grid: Q-learning victim, DDPG attacker with adaptive push,
# 6000 attack episodes. One seed per array task.
#
#     sbatch --account=<acct> --array=0-17 scripts/slurm/sweep_grid4.sh
#
# OBS=env reproduces the first 18 seeds (the attacker also read the altitude map);
# the default OBS=actions is the stricter setting where it sees only its own actions.
#
#SBATCH --job-name=grid4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

SEED=$SLURM_ARRAY_TASK_ID
OBS=${OBS:-actions}
train_and_score experiment=grid4-main config_name=obs-$OBS \
    env=grid3d victim_algo=qlearning encoder_mode=ae reward_mode=trace obs_dynamics=$OBS \
    explore_push_prob=0.2 push_cem=True cem_floor=0.05 max_episodes=6000 seed=$SEED
