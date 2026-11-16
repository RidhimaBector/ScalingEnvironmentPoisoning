#!/bin/bash
# SLURM array on fir, round 4: the config that gave 20/20 exact target path in round 3
# (retrained AE + trace reward + push exploration, plain DDPG) on 12 new seeds, to
# measure how often it succeeds and whether training reward picks the working seeds.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir_v4.sh
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

SEED=$((3 + SLURM_ARRAY_TASK_ID))
MAX_EPISODES=${MAX_EPISODES:-3000}

echo "task $SLURM_ARRAY_TASK_ID: v4_push seed=$SEED"
python -u main.py with encoder_mode=ae ae_model_path=ae/models/real_traces_AutoEncoder \
    reward_mode=trace explore_push_prob=0.1 seed=$SEED max_episodes=$MAX_EPISODES \
    model_dir=results/fir_v4_push_s${SEED}

# Evaluate the best checkpoint on 20 fresh victims straight after training.
echo "### eval v4_push_s$SEED"
python eval_attack.py --run_dir results/fir_v4_push_s${SEED} --encoder_mode ae \
    --ae_model_path ae/models/real_traces_AutoEncoder --episodes 20
