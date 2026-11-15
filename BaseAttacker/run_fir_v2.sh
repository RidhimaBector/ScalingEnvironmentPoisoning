#!/bin/bash
# SLURM array on fir, round 2: stabilised attacker (twin critics, normalised altitude
# input, push exploration) x 4 reward/encoder configs x 3 seeds. CPU only.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir_v2.sh
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

COMMON="td3=True normalize_obs=True explore_push_prob=0.1 eps_greedy_start_episodes=100"
AE="encoder_mode=ae ae_model_path=ae/models/real_traces_AutoEncoder"
CONFIGS=(
  "v2_ae_trace  $AE reward_mode=trace"
  "v2_ae_mix    $AE reward_mode=mix"
  "v2_ae_traj   $AE reward_mode=traj"
  "v2_lstm_mix  encoder_mode=lstm reward_mode=mix"
)
CFG=(${CONFIGS[$((SLURM_ARRAY_TASK_ID / 3))]})
SEED=$((SLURM_ARRAY_TASK_ID % 3))
NAME=${CFG[0]}
MAX_EPISODES=${MAX_EPISODES:-5000}

echo "task $SLURM_ARRAY_TASK_ID: $NAME seed=$SEED"
python -u main.py with "${CFG[@]:1}" $COMMON seed=$SEED max_episodes=$MAX_EPISODES \
    model_dir=results/fir_${NAME}_s${SEED}
