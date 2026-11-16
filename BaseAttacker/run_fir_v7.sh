#!/bin/bash
# SLURM array on fir, round 7: confirm the round-6 winner (retrained AE, trace reward,
# plain DDPG, adaptive push exploration, 6000 episodes; 5 of 6 seeds worked) on 12
# new seeds (6-17). Each run is evaluated on 20 fresh victims after training.
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker && sbatch run_fir_v7.sh
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

SEED=$((6 + SLURM_ARRAY_TASK_ID))
MAX_EPISODES=${MAX_EPISODES:-6000}

echo "task $SLURM_ARRAY_TASK_ID: v6_cem seed=$SEED"
python -u main.py with encoder_mode=ae ae_model_path=ae/models/real_traces_AutoEncoder \
    reward_mode=trace explore_push_prob=0.2 push_cem=True seed=$SEED \
    max_episodes=$MAX_EPISODES model_dir=results/fir_v6_cem_s${SEED}

echo "### eval v6_cem_s$SEED"
python eval_attack.py --run_dir results/fir_v6_cem_s${SEED} --encoder_mode ae \
    --ae_model_path ae/models/real_traces_AutoEncoder --episodes 20
