#!/bin/bash
# Environment- and algorithm-agnostic check on fir. The attacker now observes the
# running sum of its own perturbations instead of the environment's dynamics vector
# (obs_dynamics=actions, the new default). Two stages, submitted together:
#
#   cd ~/scratch/thesis_sep/ScalingEnvironmentPoisoning/BaseAttacker
#   AE=$(sbatch --parsable --array=0 --export=ALL,STAGE=ae run_fir_agnostic.sh)
#   sbatch --dependency=afterok:$AE --array=0-23 --export=ALL,STAGE=attack run_fir_agnostic.sh
#
# Attack tasks (adaptive push, trace reward, plain DDPG; evaluated on 20 fresh victims):
#    0-11  grid 4x4, Q-learning victim, seeds 0-11, 6000 episodes  (compare: 9 of 18 before)
#   12-15  grid 4x4, SARSA victim,      seeds 0-3,  6000 episodes
#   16-19  ring,     Q-learning victim, seeds 0-3,  3000 episodes
#   20-23  ring,     SARSA victim,      seeds 0-3,  3000 episodes
#
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=envpoison
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fir_%A_%a.out

module load python/3.12
source ~/scratch/thesis_sep/venv/bin/activate
export SACRED_FILE_ONLY=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ "$STAGE" = "ae" ]; then
  echo "train AE for the ring environment"
  python -m ae.train_ae --env ring --epochs 80 --attack_episodes 800
  exit
fi

T=$SLURM_ARRAY_TASK_ID
if   [ $T -lt 12 ]; then ENVN=grid3d; ALGO=qlearning; SEED=$T;         EPS=6000
elif [ $T -lt 16 ]; then ENVN=grid3d; ALGO=sarsa;     SEED=$((T - 12)); EPS=6000
elif [ $T -lt 20 ]; then ENVN=ring;   ALGO=qlearning; SEED=$((T - 16)); EPS=3000
else                     ENVN=ring;   ALGO=sarsa;     SEED=$((T - 20)); EPS=3000
fi
NAME=agn_${ENVN}_${ALGO}
echo "task $T: $NAME seed=$SEED"
python -u main.py with env=$ENVN victim_algo=$ALGO encoder_mode=ae reward_mode=trace \
    obs_dynamics=actions explore_push_prob=0.2 push_cem=True seed=$SEED max_episodes=$EPS \
    model_dir=results/fir_${NAME}_s${SEED}
echo "### eval ${NAME}_s$SEED"
python eval_attack.py --run_dir results/fir_${NAME}_s${SEED} --episodes 20
