#!/bin/bash
# Does the attacker need to see the environment's dynamics on larger grids?
# 5x5 and 6x6, Q-learning victims, the tuned push setting, 12,000 episodes, 4 seeds,
# run twice with the same seeds: once observing the dynamics vector (obs_dynamics=env)
# and once observing only the running sum of its own actions (obs_dynamics=actions).
#
#     sbatch --account=<acct> --array=0-15 scripts/slurm/sweep_obs_comparison.sh
#
#SBATCH --job-name=obscompare
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

SIZES=(5 6)
OBS=(env actions)
NSEEDS=4
T=$SLURM_ARRAY_TASK_ID
N=${SIZES[$((T / (2 * NSEEDS)))]}
O=${OBS[$(((T / NSEEDS) % 2))]}
SEED=$((T % NSEEDS))
train_and_score experiment=obs-comparison config_name=obs-$O \
    env=grid3d grid_size=$N victim_algo=qlearning encoder_mode=ae reward_mode=trace \
    obs_dynamics=$O push_cem=True cem_floor=-1 explore_push_prob=0.4 \
    cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000 seed=$SEED
