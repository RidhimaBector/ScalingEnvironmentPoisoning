#!/bin/bash
# Does a reward for following the target from the start fix the larger grids?
# 6x6 (and 5x5), Q-learning victims, own-actions observation, the tuned push setting,
# 12,000 episodes, 4 seeds. Same settings as the `plain` config of sweep_push_base.sh
# except for the reward (reward_mode=route: share of target states the victim's usual
# behaviour passes, with the target action, from its start until it first deviates):
#
#   route       the route reward
#   route-hold  the route reward, and the attacker stops perturbing while the hold
#               signal is 1.0 (hold_threshold=1.0). The hold signal is the route measure
#               with one more condition: the walk also ends where the victim is stuck.
#
# experiment=route-reward-v4 (after the seeding fix; v3 is the same setting before it).
# Earlier rounds: route-reward (same reward, no hold, 2
# seeds) and route-reward-v2 (the stricter measure used as the reward too: did not learn).
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
CONFIGS=(
  "route"
  "route-hold  hold_threshold=1.0"
)
NSEEDS=4
T=$SLURM_ARRAY_TASK_ID
N=${SIZES[$((T / (2 * NSEEDS)))]}
CFG=(${CONFIGS[$(((T / NSEEDS) % 2))]})
SEED=$((T % NSEEDS))
train_and_score experiment=route-reward-v4 config_name=${CFG[0]} \
    env=grid3d grid_size=$N victim_algo=qlearning encoder_mode=ae reward_mode=route \
    obs_dynamics=actions push_cem=True cem_floor=-1 explore_push_prob=0.4 \
    cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 "${CFG[@]:1}" seed=$SEED
