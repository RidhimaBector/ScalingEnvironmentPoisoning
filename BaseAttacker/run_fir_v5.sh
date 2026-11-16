#!/bin/bash
# SLURM array on fir, round 5: raise the per-seed success rate of the push config
# (about 1 in 4 in round 4) with adaptive push directions (cross-entropy method) and
# imitation of the highest-reward episodes. 4 configs x 6 seeds, each evaluated on
# 20 fresh victims straight after training. CPU only.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir_v5.sh
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

AE="encoder_mode=ae ae_model_path=ae/models/real_traces_AutoEncoder reward_mode=trace"
CONFIGS=(
  "v5_cem     $AE explore_push_prob=0.2 push_cem=True"
  "v5_bc      $AE explore_push_prob=0.1 bc_weight=5"
  "v5_cembc   $AE explore_push_prob=0.2 push_cem=True bc_weight=5"
  "v5_cembc1  $AE explore_push_prob=0.2 push_cem=True bc_weight=1"
)
CFG=(${CONFIGS[$((SLURM_ARRAY_TASK_ID / 6))]})
NAME=${CFG[0]}
SEED=$((SLURM_ARRAY_TASK_ID % 6))
MAX_EPISODES=${MAX_EPISODES:-3000}

echo "task $SLURM_ARRAY_TASK_ID: $NAME seed=$SEED"
python -u main.py with "${CFG[@]:1}" seed=$SEED max_episodes=$MAX_EPISODES \
    model_dir=results/fir_${NAME}_s${SEED}

echo "### eval ${NAME}_s$SEED"
python eval_attack.py --run_dir results/fir_${NAME}_s${SEED} --encoder_mode ae \
    --ae_model_path ae/models/real_traces_AutoEncoder --episodes 20
