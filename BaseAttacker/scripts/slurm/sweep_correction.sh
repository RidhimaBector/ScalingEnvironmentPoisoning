#!/bin/bash
# Does the attacker keep the push search's result when the actor's correction is small?
# Same setup as sweep_search_base.sh config base-search (6x6 grid, Q-learning victims,
# own-actions observation, route reward, 80% push episodes, 12,000 episodes), where the
# search alone put 20/20 victims on the route on 4 of 4 seeds but search + an unlimited
# correction passed on 2 of 4. Here the actor's output is multiplied by push_correction
# before it is added to the search's directions, and scoring uses the search's final
# directions as the base (push_final_base).
#
#   corr0.2   the actor can move each dimension at most 0.2 from the base
#   corr0.05  at most 0.05
#
#     sbatch --account=<acct> --array=0-7 scripts/slurm/sweep_correction.sh
#
# Each run prints three SCORE lines: the attacker, and the search alone (policy=push,
# policy=push_sample).
#
#SBATCH --job-name=corrlimit
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

CONFIGS=(
  "corr0.2   push_correction=0.2"
  "corr0.05  push_correction=0.05"
)
NSEEDS=4
T=$SLURM_ARRAY_TASK_ID
CFG=(${CONFIGS[$((T / NSEEDS))]})
SEED=$((T % NSEEDS))
train_and_score experiment=correction-limit config_name=${CFG[0]} \
    env=grid3d grid_size=6 victim_algo=qlearning encoder_mode=ae reward_mode=route \
    obs_dynamics=actions push_cem=True cem_floor=-1 push_base=True push_final_base=True \
    explore_push_prob=0.8 cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 \
    "${CFG[@]:1}" seed=$SEED
