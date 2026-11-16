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
#
# Every run trains, then scores itself on 20 fresh victims (results/<run>/score.json).
# Collect a sweep into a table with: python scripts/collect_scores.py results/<prefix>*

[ -n "$PYTHON_MODULE" ] && module load "$PYTHON_MODULE"
source "${VENV:-../venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# train_and_score <run name> <main.py settings...>
train_and_score() {
  local name=$1; shift
  echo "run: $name  settings: $*"
  # Training stops itself after 21 h at the latest so that scoring always runs.
  timeout 21h python -u main.py with "$@" model_dir=results/$name
  python score_attack.py --run_dir results/$name --episodes 20
}
