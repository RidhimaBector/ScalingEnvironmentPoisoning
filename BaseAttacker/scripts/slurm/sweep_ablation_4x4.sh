#!/bin/bash
# SLURM array on fir, round 3: one-change-at-a-time ablation around the best round-1
# config (retrained AE + trace reward, plain DDPG). Round 2 changed everything at once
# and got worse, so this isolates which change hurts or helps. CPU only.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir_v3.sh
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --array=0-17
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

BASE="encoder_mode=ae ae_model_path=ae/models/real_traces_AutoEncoder"
# name, first seed, overrides
CONFIGS=(
  "v3_base   3 $BASE reward_mode=trace"
  "v3_td3    0 $BASE reward_mode=trace td3=True"
  "v3_norm   0 $BASE reward_mode=trace normalize_obs=True"
  "v3_push   0 $BASE reward_mode=trace explore_push_prob=0.1"
  "v3_mix    0 $BASE reward_mode=mix"
  "v3_noise  0 $BASE reward_mode=trace expl_noise=0.2"
)
CFG=(${CONFIGS[$((SLURM_ARRAY_TASK_ID / 3))]})
NAME=${CFG[0]}
SEED=$((CFG[1] + SLURM_ARRAY_TASK_ID % 3))
MAX_EPISODES=${MAX_EPISODES:-3000}

echo "task $SLURM_ARRAY_TASK_ID: $NAME seed=$SEED"
python -u main.py with "${CFG[@]:2}" seed=$SEED max_episodes=$MAX_EPISODES \
    model_dir=results/fir_${NAME}_s${SEED}
