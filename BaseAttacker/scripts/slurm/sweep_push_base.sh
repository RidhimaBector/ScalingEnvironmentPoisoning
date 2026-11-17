#!/bin/bash
# Two changes to the attacker, alone and together, on 4x4, 5x5 and 6x6 grids with
# Q-learning victims and the own-actions observation, 4 seeds each:
#
#   plain           neither change (the comparison)
#   base            push_base=True: act as the push search's best directions + the
#                   actor's correction (pre-trained autoencoder)
#   online-ae       ae_model_path=online: autoencoder fitted on the run's own first
#                   500 episodes, no pre-training (plain actor)
#   base-online-ae  both
#
#     sbatch --account=<acct> --array=0-47 scripts/slurm/sweep_push_base.sh   # 6x6 only: 32-47
#
#SBATCH --job-name=pushbase
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out
source scripts/slurm/common.sh

SIZES=(4 5 6)
NSEEDS=4
CONFIGS=(
  "plain           push_base=False"
  "base            push_base=True"
  "online-ae       ae_model_path=online"
  "base-online-ae  push_base=True ae_model_path=online"
)
T=$SLURM_ARRAY_TASK_ID
N=${SIZES[$((T / (4 * NSEEDS)))]}
CFG=(${CONFIGS[$(((T / NSEEDS) % 4))]})
SEED=$((T % NSEEDS))
if [ "$N" -eq 4 ]; then
  SEARCH="explore_push_prob=0.2 max_episodes=6000"
else
  SEARCH="explore_push_prob=0.4 cem_batch=32 cem_elite=8 cem_lr=0.3 max_episodes=12000"
fi
train_and_score experiment=push-base config_name=${CFG[0]} \
    env=grid3d grid_size=$N victim_algo=qlearning encoder_mode=ae reward_mode=trace \
    obs_dynamics=actions push_cem=True cem_floor=-1 $SEARCH "${CFG[@]:1}" seed=$SEED
