#!/bin/bash
# Does the attacker that passes on the 6x6 grid with Q-learning victims (round 19,
# config corr0.2-pushed: 8 of 8 runs at 20/20) also work on other grid sizes and other
# victim algorithms, with every setting unchanged?
#
#   cell 0  5x5 grid, Q-learning      (other size)
#   cell 1  4x4 grid, Q-learning      (other size)
#   cell 2  6x6 grid, SARSA           (other victim)
#   cell 3  6x6 grid, REINFORCE       (other victim)
#
# 2 seeds per cell; task = cell * 2 + seed.
#
#     sbatch --account=<acct> --array=0-7 scripts/slurm/sweep_generality.sh
#
# Each run prints three SCORE lines: the attacker, and the search alone (policy=push,
# policy=push_sample).
#
#SBATCH --job-name=generality
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

CELLS=(
  "5 qlearning"
  "4 qlearning"
  "6 sarsa"
  "6 reinforce"
)
NSEEDS=2
T=$SLURM_ARRAY_TASK_ID
CELL=(${CELLS[$((T / NSEEDS))]})
SEED=$((T % NSEEDS))
train_and_score experiment=generality config_name=corr0.2-pushed \
    env=grid3d grid_size=${CELL[0]} victim_algo=${CELL[1]} encoder_mode=ae reward_mode=route \
    obs_dynamics=actions push_cem=True cem_floor=-1 push_base=True push_final_base=True \
    explore_push_prob=0.8 cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 \
    push_correction=0.2 push_correct_held=False seed=$SEED
