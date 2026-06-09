#!/usr/bin/env bash
# June 9 experiments: full blackbox LSTM, 1 victim, 3 seeds × {4x4, 6x6}
# Seeds within each grid run in parallel; grids run sequentially.
# After all runs complete, figures are generated automatically.
#
# Usage:
#   ./run_june9.sh
#   ./run_june9.sh 2>&1 | tee logs/june9.log

set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs figures

DATE=$(date +%Y-%m-%d)
MAX_EPS=15000

echo "======================================================"
echo " June 9 experiments — blackbox LSTM, max_episodes=$MAX_EPS"
echo " 4x4 grid: seeds 0,1,2 in parallel"
echo " 6x6 grid: seeds 0,1,2 in parallel (after 4x4 finishes)"
echo "======================================================"

# Record baseline Sacred run ID so we can find new runs after training
BASELINE=$(python - <<'PYEOF'
from pymongo import MongoClient
db = MongoClient('localhost:27017')['env_poisoning']
r = db.runs.find_one(sort=[('_id', -1)])
print(r['_id'] if r else 0)
PYEOF
)
echo "Baseline Sacred run ID: $BASELINE"

# ---------------------------------------------------------------------------
# 4x4 grid — 3 seeds in parallel
# ---------------------------------------------------------------------------
echo ""
echo "[4x4] Starting seeds 0,1,2 in parallel..."
python main.py with seed=0 max_episodes=$MAX_EPS > logs/june9_4x4_seed0.log 2>&1 &
PID0=$!
python main.py with seed=1 max_episodes=$MAX_EPS > logs/june9_4x4_seed1.log 2>&1 &
PID1=$!
python main.py with seed=2 max_episodes=$MAX_EPS > logs/june9_4x4_seed2.log 2>&1 &
PID2=$!

wait $PID0 && echo "[4x4 seed=0] done" || echo "[4x4 seed=0] FAILED"
wait $PID1 && echo "[4x4 seed=1] done" || echo "[4x4 seed=1] FAILED"
wait $PID2 && echo "[4x4 seed=2] done" || echo "[4x4 seed=2] FAILED"
echo "[4x4] All seeds complete."

# ---------------------------------------------------------------------------
# 6x6 grid — 3 seeds in parallel
# ---------------------------------------------------------------------------
echo ""
echo "[6x6] Starting seeds 0,1,2 in parallel..."
python main.py with seed=0 max_episodes=$MAX_EPS grid_size=6 > logs/june9_6x6_seed0.log 2>&1 &
PID3=$!
python main.py with seed=1 max_episodes=$MAX_EPS grid_size=6 > logs/june9_6x6_seed1.log 2>&1 &
PID4=$!
python main.py with seed=2 max_episodes=$MAX_EPS grid_size=6 > logs/june9_6x6_seed2.log 2>&1 &
PID5=$!

wait $PID3 && echo "[6x6 seed=0] done" || echo "[6x6 seed=0] FAILED"
wait $PID4 && echo "[6x6 seed=1] done" || echo "[6x6 seed=1] FAILED"
wait $PID5 && echo "[6x6 seed=2] done" || echo "[6x6 seed=2] FAILED"
echo "[6x6] All seeds complete."

# ---------------------------------------------------------------------------
# Extract run IDs and model dirs from MongoDB
# ---------------------------------------------------------------------------
echo ""
echo "Querying MongoDB for new run IDs..."

eval "$(python - <<PYEOF
from pymongo import MongoClient
db = MongoClient('localhost:27017')['env_poisoning']

runs = list(db.runs.find({'_id': {'\$gt': $BASELINE}}, sort=[('_id', 1)]))

def ids(rs):
    return ' '.join(str(r['_id']) for r in rs)

def dirs(rs):
    return ' '.join(
        r.get('info', {}).get('model_dir', '').replace(' ', '_')
        for r in rs
    )

r4 = [r for r in runs if not r.get('config', {}).get('grid_size')]
r6 = [r for r in runs if str(r.get('config', {}).get('grid_size', '')) == '6']

print(f'RUN_IDS_4X4="{ids(r4)}"')
print(f'RUN_IDS_6X6="{ids(r6)}"')
print(f'MODEL_DIRS_4X4="{dirs(r4)}"')
print(f'MODEL_DIRS_6X6="{dirs(r6)}"')
PYEOF
)"

echo "4x4 run IDs : $RUN_IDS_4X4"
echo "6x6 run IDs : $RUN_IDS_6X6"

# ---------------------------------------------------------------------------
# Figures — per-run dashboards (includes policy heatmaps)
# ---------------------------------------------------------------------------
echo ""
echo "Generating per-run dashboards..."
for RID in $RUN_IDS_4X4 $RUN_IDS_6X6; do
    echo "  analyze_run.py --run_id $RID"
    python analyze_run.py --run_id "$RID" --out "figures/june9_run_${RID}" \
        > "logs/june9_analyze_${RID}.log" 2>&1 \
        && echo "  run $RID: done" || echo "  run $RID: FAILED"
done

# ---------------------------------------------------------------------------
# Figures — multi-seed aggregates
# ---------------------------------------------------------------------------
echo ""
echo "Generating multi-seed aggregate figures..."

OUT_4X4="figures/june9_4x4_${DATE}"
OUT_6X6="figures/june9_6x6_${DATE}"

python analyze_seeds.py --run_ids $RUN_IDS_4X4 --out "$OUT_4X4" \
    > logs/june9_seeds_4x4.log 2>&1 \
    && echo "[4x4] analyze_seeds done" || echo "[4x4] analyze_seeds FAILED"

python analyze_seeds.py --run_ids $RUN_IDS_6X6 --out "$OUT_6X6" \
    > logs/june9_seeds_6x6.log 2>&1 \
    && echo "[6x6] analyze_seeds done" || echo "[6x6] analyze_seeds FAILED"

# ---------------------------------------------------------------------------
# Figures — before/after policy heatmaps averaged across seeds
# ---------------------------------------------------------------------------
echo ""
echo "Generating policy heatmaps (averaged across seeds)..."

python demo_policy.py \
    --run_dirs $MODEL_DIRS_4X4 \
    --out "$OUT_4X4" \
    > logs/june9_demo_4x4.log 2>&1 \
    && echo "[4x4] demo_policy done" || echo "[4x4] demo_policy FAILED"

python demo_policy.py \
    --run_dirs $MODEL_DIRS_6X6 \
    --grid_size 6 \
    --out "$OUT_6X6" \
    > logs/june9_demo_6x6.log 2>&1 \
    && echo "[6x6] demo_policy done" || echo "[6x6] demo_policy FAILED"

# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo " All done. Figures written to:"
echo "   Per-run :  figures/june9_run_<id>/"
echo "   4x4 agg :  $OUT_4X4/"
echo "   6x6 agg :  $OUT_6X6/"
echo " Logs      :  logs/june9_*.log"
echo "======================================================"
