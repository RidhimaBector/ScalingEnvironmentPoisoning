# Shared setup for the SLURM sweeps. Sourced by every script in this folder.
#
# Run sweeps from BaseAttacker/, passing your own allocation:
#
#     cd BaseAttacker && mkdir -p logs
#     sbatch --account=<your-account> --array=<range> scripts/slurm/<sweep>.sh
#
# Environment variables:
#     VENV            path to the virtual environment (default: ../venv)
#     PYTHON_MODULE   module to load first on clusters that need it, e.g. python/3.12
#     TRAIN_TIMEOUT   training stops itself after this long so scoring still runs (default 21h)
#     HF_PUBLISH=0    do not upload to the Hugging Face Hub (see publish.py)
#
# Every run trains, scores itself on 20 fresh victims, and is published to the Hub.
# Runs are named by naming.py:
#     results/<experiment>/<env>-<variant>__<victim>__<config>__seed<N>
# Collect a sweep into a table with: python scripts/collect_scores.py results/<experiment>/*

[ -n "$PYTHON_MODULE" ] && module load "$PYTHON_MODULE"
source "${VENV:-../venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# train_and_score <main.py settings...>   (must include experiment=<name> and seed=<n>)
train_and_score() {
  local run_dir
  run_dir=$(python main.py print_run_dir with "$@" 2>/dev/null | tail -n 1)
  echo "run: $run_dir  settings: $*"
  timeout "${TRAIN_TIMEOUT:-21h}" python -u main.py with "$@"
  python score_attack.py --run_dir "$run_dir" --episodes 20
  python publish.py --run_dir "$run_dir" --log "logs/${SLURM_JOB_NAME:-run}_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.out"
}
