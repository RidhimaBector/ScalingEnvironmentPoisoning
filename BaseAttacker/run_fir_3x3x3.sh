#!/bin/bash
# 3 x 3 x 3 sweep on fir: 3 environments x 3 variants x 3 victim algorithms, 3 seeds each.
#
#   environment   variants                      victims
#   grid3d        4x4, 5x5, 6x6                 qlearning, sarsa, reinforce
#   ring          8, 12, 16 states              qlearning, sarsa, reinforce
#   lunarlander   start kick 500, 1000, 1500    dqn, vpg, a2c
#
# One attacker for all 27 cells: plain DDPG + adaptive push exploration, observing an
# embedding of victim behaviour plus the running sum of its own actions. Only the
# behaviour encoder and reward follow the kind of victim: a per-state trace
# autoencoder and trace reward for tabular victims; an online trajectory LSTM and
# embedding-similarity reward for neural victims.
#
# Every run is scored on 20 fresh victims against a no-attack baseline (score_attack.py).
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
#   AE=$(sbatch --parsable --array=0-1 --export=ALL,STAGE=ae run_fir_3x3x3.sh)
#   sbatch --dependency=afterok:$AE --array=0-80 --export=ALL,STAGE=attack run_fir_3x3x3.sh
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$STAGE" = "ae" ]; then
  N=$((12 + 4 * SLURM_ARRAY_TASK_ID))      # ring sizes 12 and 16 (8 is already trained)
  echo "train AE for ring with $N states"
  python -m ae.train_ae --env ring --env_kwargs "{\"n_states\": $N}" --epochs 80 --attack_episodes 800
  exit
fi

NSEEDS=3
T=$SLURM_ARRAY_TASK_ID
CELL=$((T / NSEEDS)); SEED=$((T % NSEEDS))
E=$((CELL / 9)); V=$(((CELL / 3) % 3)); A=$((CELL % 3))

PUSH="explore_push_prob=0.2 push_cem=True cem_floor=-1"
TAB_ALGOS=(qlearning sarsa reinforce)
NN_ALGOS=(dqn vpg a2c)

if [ $E -eq 0 ]; then
  SIZES=(4 5 6); EPS=(6000 12000 12000)
  ALGO=${TAB_ALGOS[$A]}; NAME=s27_grid${SIZES[$V]}_${ALGO}
  ARGS="env=grid3d grid_size=${SIZES[$V]} victim_algo=$ALGO encoder_mode=ae reward_mode=trace max_episodes=${EPS[$V]}"
elif [ $E -eq 1 ]; then
  SIZES=(8 12 16)
  ALGO=${TAB_ALGOS[$A]}; NAME=s27_ring${SIZES[$V]}_${ALGO}
  ARGS="env=ring env_args=n_states:${SIZES[$V]} victim_algo=$ALGO encoder_mode=ae reward_mode=trace max_episodes=3000"
else
  KICKS=(500 1000 1500)
  ALGO=${NN_ALGOS[$A]}; NAME=s27_lander${KICKS[$V]}_${ALGO}
  ARGS="env=lunarlander env_args=initial_random:${KICKS[$V]} victim_algo=$ALGO encoder_mode=lstm reward_mode=embedding effort_weight=0.1 lstm_seq_len=200 lstm_memory_size=4000 lstm_embedding_dim=100 victim_n_episodes=20 max_timesteps=10 max_episodes=1000"
fi

echo "task $T: $NAME seed=$SEED"
# Training stops itself after 21 h at the latest so that scoring always runs.
timeout 21h python -u main.py with $ARGS obs_dynamics=actions $PUSH seed=$SEED \
    model_dir=results/fir_${NAME}_s${SEED}
echo "### score ${NAME}_s$SEED"
python score_attack.py --run_dir results/fir_${NAME}_s${SEED} --episodes 20
