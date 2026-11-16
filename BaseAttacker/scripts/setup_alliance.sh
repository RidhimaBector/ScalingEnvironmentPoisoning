#!/bin/bash
# Create the Python environment on an Alliance Canada cluster (fir, killarney, nibi, ...)
# and check it by running the tests. Run on a login node, from BaseAttacker/:
#
#     bash scripts/setup_alliance.sh <venv path>
#
# Uses the cluster's pre-built wheels where they exist and PyPI for the rest.
set -e
VENV=${1:?usage: bash scripts/setup_alliance.sh <venv path>}
PYTHON_MODULE=${PYTHON_MODULE:-python/3.12}

module load "$PYTHON_MODULE"
[ -d "$VENV" ] || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip

# Prefer the cluster wheelhouse (--no-index); fall back to PyPI package by package.
for pkg in torch numpy scipy gymnasium yacs "sacred>=0.8.4" "setuptools<81" matplotlib \
           pytest huggingface_hub POT Box2D; do
  pip install -q --no-index "$pkg" 2>/dev/null || pip install -q "$pkg" \
    || echo "WARNING: could not install $pkg (Box2D is only needed for LunarLander)"
done

mkdir -p logs
python -m pytest tests -q
echo
echo "Environment ready. Submit sweeps with:"
echo "  PYTHON_MODULE=$PYTHON_MODULE VENV=$VENV sbatch --account=<account> --array=<range> scripts/slurm/<sweep>.sh"
