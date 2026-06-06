#!/usr/bin/env bash
mkdir -p results

for rate in 0.0001 0.001 0.01; do
  for seed in 0 1 2; do
    python -u main.py with encoder_mode=lstm max_episodes=8000 attack_rate=$rate seed=$seed \
      num_victims=1 model_dir=results/rate${rate}_seed${seed} \
      > results/rate${rate}_seed${seed}.log 2>&1 &
  done
  wait
done
echo "All done"
