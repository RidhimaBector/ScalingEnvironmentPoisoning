#!/usr/bin/env bash
# June 8 experiments: full blackbox LSTM, 1 victim, 3 seeds × {4x4, 6x6}
# Seeds within each grid run in parallel; grids run sequentially.
#
# Usage:
#   ./run_june8.sh
#   ./run_june8.sh 2>&1 | tee logs/june8.log

set -e
cd "$(dirname "$0")"
source ../venv/bin/activate
mkdir -p logs

echo "======================================================"
echo " June 8 experiments — blackbox LSTM, max_episodes=15000"
echo " 4x4 grid: seeds 0,1,2 in parallel"
echo " 6x6 grid: seeds 0,1,2 in parallel (after 4x4 finishes)"
echo "======================================================"

# --- 4x4 grid: 3 seeds in parallel ---
echo ""
echo "[4x4] Starting seeds 0,1,2 in parallel..."
python main.py with seed=0 max_episodes=15000 > logs/june8_4x4_seed0.log 2>&1 &
PID0=$!
python main.py with seed=1 max_episodes=15000 > logs/june8_4x4_seed1.log 2>&1 &
PID1=$!
python main.py with seed=2 max_episodes=15000 > logs/june8_4x4_seed2.log 2>&1 &
PID2=$!

wait $PID0 && echo "[4x4 seed=0] done" || echo "[4x4 seed=0] FAILED"
wait $PID1 && echo "[4x4 seed=1] done" || echo "[4x4 seed=1] FAILED"
wait $PID2 && echo "[4x4 seed=2] done" || echo "[4x4 seed=2] FAILED"
echo "[4x4] All seeds complete."

# --- 6x6 grid: 3 seeds in parallel ---
echo ""
echo "[6x6] Starting seeds 0,1,2 in parallel..."
python main.py with seed=0 max_episodes=15000 grid_size=6 > logs/june8_6x6_seed0.log 2>&1 &
PID3=$!
python main.py with seed=1 max_episodes=15000 grid_size=6 > logs/june8_6x6_seed1.log 2>&1 &
PID4=$!
python main.py with seed=2 max_episodes=15000 grid_size=6 > logs/june8_6x6_seed2.log 2>&1 &
PID5=$!

wait $PID3 && echo "[6x6 seed=0] done" || echo "[6x6 seed=0] FAILED"
wait $PID4 && echo "[6x6 seed=1] done" || echo "[6x6 seed=1] FAILED"
wait $PID5 && echo "[6x6 seed=2] done" || echo "[6x6 seed=2] FAILED"
echo "[6x6] All seeds complete."

echo ""
echo "All 6 runs complete. Logs in logs/june8_*.log"
echo "Check Sacred IDs then plot with:"
echo "  python analyze_seeds.py --run_ids <4x4 ids> --out figures/june8_4x4 --encoder_mode lstm"
echo "  python analyze_seeds.py --run_ids <6x6 ids> --out figures/june8_6x6 --encoder_mode lstm"
