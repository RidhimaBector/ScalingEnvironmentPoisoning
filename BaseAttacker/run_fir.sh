#!/bin/bash
# SLURM array on fir (Alliance): 4 attacker configs x 3 seeds, CPU only.
# Sacred logs to files (sacred_runs/) because there is no MongoDB on the cluster.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir.sh
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --array=0-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(
  "aeold_trace   encoder_mode=ae   reward_mode=trace"
  "aenew_trace   encoder_mode=ae   reward_mode=trace ae_model_path=ae/models/real_traces_AutoEncoder"
  "aenew_traj    encoder_mode=ae   reward_mode=traj  ae_model_path=ae/models/real_traces_AutoEncoder"
  "lstmdyn_trace encoder_mode=lstm reward_mode=trace"
)
CFG=(${CONFIGS[$((SLURM_ARRAY_TASK_ID / 3))]})
SEED=$((SLURM_ARRAY_TASK_ID % 3))
NAME=${CFG[0]}
MAX_EPISODES=${MAX_EPISODES:-3000}

echo "task $SLURM_ARRAY_TASK_ID: $NAME seed=$SEED"
python -u main.py with "${CFG[@]:1}" seed=$SEED max_episodes=$MAX_EPISODES \
    model_dir=results/fir_${NAME}_s${SEED}
