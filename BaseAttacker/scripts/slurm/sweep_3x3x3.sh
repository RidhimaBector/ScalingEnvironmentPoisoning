#!/bin/bash
# 3 environments x 3 variants x 3 victim algorithms, 3 seeds each (81 runs).
#
#   environment   variants                      victims
#   grid3d        4x4, 5x5, 6x6                 qlearning, sarsa, reinforce
#   ring          8, 12, 16 states              qlearning, sarsa, reinforce
#   lunarlander   start kick 500, 1000, 1500    dqn, vpg, a2c
#
# The same attacker everywhere (DDPG + adaptive push, observing a behaviour embedding
# plus the running sum of its own actions). Only the behaviour encoder and reward
# follow the kind of victim: trace autoencoder + trace reward for tabular victims,
# online trajectory LSTM + embedding reward for neural victims.
#
#     sbatch --account=<acct> --array=0-80 scripts/slurm/sweep_3x3x3.sh
#
#SBATCH --job-name=s27
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

NSEEDS=3
T=$SLURM_ARRAY_TASK_ID
CELL=$((T / NSEEDS)); SEED=$((T % NSEEDS))
E=$((CELL / 9)); V=$(((CELL / 3) % 3)); A=$((CELL % 3))
TABULAR=(qlearning sarsa reinforce)
NEURAL=(dqn vpg a2c)
PUSH="explore_push_prob=0.2 push_cem=True cem_floor=-1"

if [ $E -eq 0 ]; then
  SIZES=(4 5 6); EPISODES=(6000 12000 12000)
  NAME=s27_grid${SIZES[$V]}_${TABULAR[$A]}
  ARGS="env=grid3d grid_size=${SIZES[$V]} victim_algo=${TABULAR[$A]} max_episodes=${EPISODES[$V]}"
  # Larger grids: push in 40% of episodes with a larger batch and slower step (tuned in
  # sweep_grid_sizes.sh; the 20% setting never reached the exact route on 6x6).
  [ ${SIZES[$V]} -gt 4 ] && PUSH="explore_push_prob=0.4 push_cem=True cem_floor=-1 cem_batch=32 cem_elite=8 cem_lr=0.3"
elif [ $E -eq 1 ]; then
  SIZES=(8 12 16)
  NAME=s27_ring${SIZES[$V]}_${TABULAR[$A]}
  ARGS="env=ring env_args=n_states:${SIZES[$V]} victim_algo=${TABULAR[$A]} max_episodes=3000"
else
  KICKS=(500 1000 1500)
  NAME=s27_lander${KICKS[$V]}_${NEURAL[$A]}
  ARGS="lander env_args=initial_random:${KICKS[$V]} victim_algo=${NEURAL[$A]}"
fi
train_and_score ${NAME}_s${SEED} $ARGS obs_dynamics=actions $PUSH seed=$SEED
