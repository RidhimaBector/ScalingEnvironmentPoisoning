#!/bin/bash
# Lander check with victims that have time to learn: DQN victims, 60 episodes per
# attack epoch and 10 epochs (600 victim episodes per attack episode; un-attacked DQN
# victims reach a positive return by then). 150 attack episodes, 3 seeds, with and
# without the push-search base. Slow: about a day per run.
#
#     SCORE_EPISODES=5 TRAIN_TIMEOUT=19h sbatch --account=<acct> --array=0-5 scripts/slurm/verify_lander.sh
#
#SBATCH --job-name=landercheck
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

NSEEDS=3
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % NSEEDS))
if [ $((T / NSEEDS)) -eq 0 ]; then CFG="base push_base=True"; else CFG="plain push_base=False"; fi
CFG=($CFG)
train_and_score lander experiment=lander-check config_name=${CFG[0]} victim_algo=dqn \
    victim_n_episodes=60 max_timesteps=10 max_episodes=150 obs_dynamics=actions \
    explore_push_prob=0.4 push_cem=True cem_floor=-1 cem_batch=8 cem_elite=2 \
    "${CFG[@]:1}" seed=$SEED
