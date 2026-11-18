#!/bin/bash
# Does the attacker work when it starts from the push search's result, and does the
# search get there on every seed with more budget?
# 6x6 grid, Q-learning victims, own-actions observation, route reward, 12,000 episodes,
# 4 seeds. In every config the attacker acts as (the search's best directions) + (the
# actor's correction) (push_base=True). Background (route-reward-v5): the search alone
# put 20/20 victims on the route on 2 of 4 seeds and was still improving at the end on
# the other 2; the actor alone scored 0/20 on all.
#
#   base              40% of episodes are push episodes (the v5 search, unchanged)
#   base-search       80% push episodes: twice the search budget
#   base-search-wide  80%, and the search widens when it stalls (cem_patience=5)
#   base-search-fast  80%, and the search updates twice as often (batch 16, elite 4)
#
#     sbatch --account=<acct> --array=0-15 scripts/slurm/sweep_search_base.sh
#
# Each run prints three SCORE lines: the attacker, and the search alone (policy=push,
# policy=push_sample).
#
#SBATCH --job-name=searchbase
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

CONFIGS=(
  "base              explore_push_prob=0.4"
  "base-search       explore_push_prob=0.8"
  "base-search-wide  explore_push_prob=0.8 cem_patience=5"
  "base-search-fast  explore_push_prob=0.8 cem_batch=16 cem_elite=4"
)
NSEEDS=4
T=$SLURM_ARRAY_TASK_ID
CFG=(${CONFIGS[$((T / NSEEDS))]})
SEED=$((T % NSEEDS))
train_and_score experiment=search-base config_name=${CFG[0]} \
    env=grid3d grid_size=6 victim_algo=qlearning encoder_mode=ae reward_mode=route \
    obs_dynamics=actions push_cem=True cem_floor=-1 push_base=True \
    cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 "${CFG[@]:1}" seed=$SEED
