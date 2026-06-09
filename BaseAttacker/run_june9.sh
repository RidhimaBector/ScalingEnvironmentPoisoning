#!/usr/bin/env bash
# June 9 experiments: blackbox LSTM, 1 victim
#   Grids        : 4x4, 6x6
#   Victim eps   : 80, 400, 1200  (victim training episodes per attack step)
#   Seeds        : 0, 1, 2 (run in parallel within each group)
#   Groups run sequentially: 4x4×80 → 4x4×400 → 4x4×1200 → 6x6×80 → …
#
# Usage:
#   ./run_june9.sh
#   ./run_june9.sh 2>&1 | tee logs/june9_main.log

set -e
cd "$(dirname "$0")"
source venv/bin/activate
mkdir -p logs figures

DATE=$(date +%Y-%m-%d)
MAX_EPS=15000

echo "======================================================"
echo " June 9 — blackbox LSTM, max_episodes=$MAX_EPS"
echo " Grids: 4x4, 6x6   Victim eps: 80, 400, 1200   Seeds: 0-2"
echo " Total runs: 18"
echo "======================================================"

# Baseline Sacred ID — used to find only this session's runs in MongoDB
BASELINE=$(python - <<'PYEOF'
from pymongo import MongoClient
db = MongoClient('localhost:27017')['env_poisoning']
r = db.runs.find_one(sort=[('_id', -1)])
print(r['_id'] if r else 0)
PYEOF
)
echo "Baseline Sacred run ID: $BASELINE"

# ---------------------------------------------------------------------------
# Helper: run 3 seeds in parallel for one (grid, victim_eps) group
#   run_group  LABEL  VICTIM_EPS  [EXTRA_SACRED_ARGS...]
# ---------------------------------------------------------------------------
run_group() {
    local LABEL=$1
    local VEPS=$2
    shift 2
    local EXTRA="$*"

    echo ""
    echo "[$LABEL] Starting seeds 0,1,2 (victim_n_episodes=$VEPS)..."
    python main.py with seed=0 max_episodes=$MAX_EPS victim_n_episodes=$VEPS $EXTRA \
        > "logs/june9_${LABEL}_seed0.log" 2>&1 &
    local P0=$!
    python main.py with seed=1 max_episodes=$MAX_EPS victim_n_episodes=$VEPS $EXTRA \
        > "logs/june9_${LABEL}_seed1.log" 2>&1 &
    local P1=$!
    python main.py with seed=2 max_episodes=$MAX_EPS victim_n_episodes=$VEPS $EXTRA \
        > "logs/june9_${LABEL}_seed2.log" 2>&1 &
    local P2=$!

    wait $P0 && echo "[$LABEL seed=0] done" || echo "[$LABEL seed=0] FAILED"
    wait $P1 && echo "[$LABEL seed=1] done" || echo "[$LABEL seed=1] FAILED"
    wait $P2 && echo "[$LABEL seed=2] done" || echo "[$LABEL seed=2] FAILED"
    echo "[$LABEL] complete."
}

# ---------------------------------------------------------------------------
# Training — 4x4
# ---------------------------------------------------------------------------
run_group  4x4_80eps    80
run_group  4x4_400eps   400
run_group  4x4_1200eps  1200

# ---------------------------------------------------------------------------
# Training — 6x6
# ---------------------------------------------------------------------------
run_group  6x6_80eps    80    grid_size=6
run_group  6x6_400eps   400   grid_size=6
run_group  6x6_1200eps  1200  grid_size=6

echo ""
echo "All 18 training runs complete."

# ---------------------------------------------------------------------------
# Extract run IDs and model dirs from MongoDB, grouped by (grid, victim_eps)
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
    return ' '.join(r.get('info', {}).get('model_dir', '') for r in rs)

def match(runs, grid_size, veps):
    gs = str(grid_size) if grid_size else ''
    return [
        r for r in runs
        if str(r.get('config', {}).get('grid_size', '')) == gs
        and r.get('config', {}).get('victim_n_episodes') == veps
    ]

groups = [
    ('4X4_80',   None, 80),
    ('4X4_400',  None, 400),
    ('4X4_1200', None, 1200),
    ('6X6_80',   6,    80),
    ('6X6_400',  6,    400),
    ('6X6_1200', 6,    1200),
]

for name, gs, veps in groups:
    rs = match(runs, gs, veps)
    print(f'RUN_IDS_{name}="{ids(rs)}"')
    print(f'MODEL_DIRS_{name}="{dirs(rs)}"')
PYEOF
)"

# Report what was found
for GRP in 4X4_80 4X4_400 4X4_1200 6X6_80 6X6_400 6X6_1200; do
    eval "echo \"  $GRP run IDs : \$RUN_IDS_$GRP\""
done

# ---------------------------------------------------------------------------
# Figures — per-run dashboards (training curves + policy heatmap)
# ---------------------------------------------------------------------------
echo ""
echo "Generating per-run dashboards..."
ALL_IDS="$RUN_IDS_4X4_80 $RUN_IDS_4X4_400 $RUN_IDS_4X4_1200 \
         $RUN_IDS_6X6_80 $RUN_IDS_6X6_400 $RUN_IDS_6X6_1200"
for RID in $ALL_IDS; do
    python analyze_run.py --run_id "$RID" --out "figures/june9_run_${RID}" \
        > "logs/june9_analyze_${RID}.log" 2>&1 \
        && echo "  run $RID: done" || echo "  run $RID: FAILED"
done

# ---------------------------------------------------------------------------
# Figures — per-group aggregate (analyze_seeds + demo_policy)
# ---------------------------------------------------------------------------
echo ""
echo "Generating per-group aggregate figures..."

gen_group_figs() {
    local GRP=$1       # e.g. 4X4_80
    local VEPS=$2      # e.g. 80
    local GSIZE=$3     # e.g. "" or "6"
    local OUT=$4       # output dir

    eval "local IDS=\$RUN_IDS_${GRP}"
    eval "local DIRS=\$MODEL_DIRS_${GRP}"

    [ -z "$IDS" ] && { echo "  [$GRP] no run IDs found, skipping"; return; }

    mkdir -p "$OUT"

    python analyze_seeds.py --run_ids $IDS --out "$OUT" \
        > "logs/june9_seeds_${GRP}.log" 2>&1 \
        && echo "  [$GRP] analyze_seeds done" || echo "  [$GRP] analyze_seeds FAILED"

    local GSARG=""
    [ -n "$GSIZE" ] && GSARG="--grid_size $GSIZE"

    python demo_policy.py --run_dirs $DIRS $GSARG \
        --victim_episodes "$VEPS" \
        --out "$OUT" \
        > "logs/june9_demo_${GRP}.log" 2>&1 \
        && echo "  [$GRP] demo_policy done" || echo "  [$GRP] demo_policy FAILED"
}

gen_group_figs  4X4_80    80    ""  "figures/june9_4x4_80eps_${DATE}"
gen_group_figs  4X4_400   400   ""  "figures/june9_4x4_400eps_${DATE}"
gen_group_figs  4X4_1200  1200  ""  "figures/june9_4x4_1200eps_${DATE}"
gen_group_figs  6X6_80    80    6   "figures/june9_6x6_80eps_${DATE}"
gen_group_figs  6X6_400   400   6   "figures/june9_6x6_400eps_${DATE}"
gen_group_figs  6X6_1200  1200  6   "figures/june9_6x6_1200eps_${DATE}"

# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo " All done."
echo " Per-run figures  :  figures/june9_run_<id>/"
echo " Group aggregates :  figures/june9_{4x4,6x6}_{80,400,1200}eps_${DATE}/"
echo " Logs             :  logs/june9_*.log"
echo "======================================================"
