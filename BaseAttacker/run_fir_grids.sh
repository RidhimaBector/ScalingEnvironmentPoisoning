#!/bin/bash
# Larger grids on fir. Two stages, submitted together:
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
#   AE=$(sbatch --parsable --array=0-1 --export=ALL,STAGE=ae run_fir_grids.sh)
#   sbatch --dependency=afterok:$AE --array=0-11 --export=ALL,STAGE=attack run_fir_grids.sh
#
# STAGE=ae     : train the behaviour-trace autoencoder for each grid size (one task per size)
# STAGE=attack : the round-6 winning attacker (trace reward, plain DDPG, adaptive push)
#                on each grid size x 6 seeds, evaluated on 20 fresh victims afterwards
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SIZES=(5 6)
NSEEDS=6
MAX_EPISODES=${MAX_EPISODES:-8000}

if [ "$STAGE" = "ae" ]; then
  N=${SIZES[$SLURM_ARRAY_TASK_ID]}
  echo "task $SLURM_ARRAY_TASK_ID: train AE for ${N}x${N}"
  python -m ae.train_ae --grid_size $N --epochs 80 --attack_episodes 800
else
  N=${SIZES[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}
  SEED=$((SLURM_ARRAY_TASK_ID % NSEEDS))
  NAME=grid${N}_cem
  echo "task $SLURM_ARRAY_TASK_ID: $NAME seed=$SEED"
  python -u main.py with grid_size=$N encoder_mode=ae reward_mode=trace \
      explore_push_prob=0.2 push_cem=True seed=$SEED max_episodes=$MAX_EPISODES \
      model_dir=results/fir_${NAME}_s${SEED}
  echo "### eval ${NAME}_s$SEED"
  python eval_attack.py --run_dir results/fir_${NAME}_s${SEED} --encoder_mode ae \
      --grid_size $N --episodes 20
fi
