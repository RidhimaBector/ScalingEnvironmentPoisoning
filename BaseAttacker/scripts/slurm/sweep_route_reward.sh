#!/bin/bash
# Does a reward for following the target from the start fix the larger grids?
# 6x6 (and 5x5), Q-learning victims, own-actions observation, the tuned push setting,
# 12,000 episodes. Same settings as the `plain` config of sweep_push_base.sh except
# for the reward:
#
#   route      reward_mode=route: share of target states the victim's usual behaviour
#              passes, with the target action, from its start until it first deviates
#   route-mix  reward_mode=route_mix: mean of that and the usual trace reward
#
#     sbatch --account=<acct> --array=0-15 scripts/slurm/sweep_route_reward.sh   # 6x6: 0-7
#
#SBATCH --job-name=routereward
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

SIZES=(6 5)
MODES=(route route_mix)
NSEEDS=4
T=$SLURM_ARRAY_TASK_ID
N=${SIZES[$((T / (2 * NSEEDS)))]}
MODE=${MODES[$(((T / NSEEDS) % 2))]}
SEED=$((T % NSEEDS))
train_and_score experiment=route-reward config_name=${MODE//_/-} \
    env=grid3d grid_size=$N victim_algo=qlearning encoder_mode=ae reward_mode=$MODE \
    obs_dynamics=actions push_cem=True cem_floor=-1 explore_push_prob=0.4 \
    cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 seed=$SEED
