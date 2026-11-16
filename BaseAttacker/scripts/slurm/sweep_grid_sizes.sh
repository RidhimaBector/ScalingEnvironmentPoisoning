#!/bin/bash
# Tuning sweep for larger grids on fir. Two stages, submitted together:
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
#   AE=$(sbatch --parsable --array=0-2 --export=ALL,STAGE=ae run_fir_tune.sh)
#   sbatch --dependency=afterok:$AE --array=0-35 --export=ALL,STAGE=attack run_fir_tune.sh
#
# STAGE=ae     : train the trace autoencoder for each grid size if it is missing
# STAGE=attack : 3 grid sizes x 3 adaptive-push settings x 4 seeds, 12000 episodes,
#                each evaluated on 20 fresh victims afterwards
#
# Round "grids" showed the push search stalls on larger grids (push episodes ~65-75% of
# target states). All three settings scale the direction-probability floor with the
# grid (cem_floor=-1); they differ in push fraction and in batch / step size.
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SIZES=(5 6 8)
NSEEDS=4
MAX_EPISODES=${MAX_EPISODES:-12000}
CONFIGS=(
  "floor   explore_push_prob=0.2 cem_floor=-1"
  "push40  explore_push_prob=0.4 cem_floor=-1"
  "slow    explore_push_prob=0.4 cem_floor=-1 cem_batch=32 cem_elite=8 cem_lr=0.3"
)

if [ "$STAGE" = "ae" ]; then
  N=${SIZES[$SLURM_ARRAY_TASK_ID]}
  if [ -f ae/models/real_traces_${N}x${N}_AutoEncoder ]; then
    echo "AE for ${N}x${N} already exists"
  else
    echo "train AE for ${N}x${N}"
    python -m ae.train_ae --grid_size $N --epochs 80 --attack_episodes 800
  fi
else
  T=$SLURM_ARRAY_TASK_ID
  N=${SIZES[$((T / (3 * NSEEDS)))]}
  CFG=(${CONFIGS[$(((T / NSEEDS) % 3))]})
  SEED=$((T % NSEEDS))
  NAME=tune${N}_${CFG[0]}
  echo "task $T: $NAME seed=$SEED"
  python -u main.py with grid_size=$N encoder_mode=ae reward_mode=trace push_cem=True \
      "${CFG[@]:1}" seed=$SEED max_episodes=$MAX_EPISODES \
      model_dir=results/fir_${NAME}_s${SEED}
  echo "### eval ${NAME}_s$SEED"
  python eval_attack.py --run_dir results/fir_${NAME}_s${SEED} --encoder_mode ae \
      --grid_size $N --episodes 20
fi
