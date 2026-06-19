#!/bin/bash
for algo in ppo sac; do
  for seed in 0 1 2; do
    echo "=============== eval ${algo} seed ${seed} ==============="
    python src/evaluate.py --run-dir experiments/dq1/results/${algo}_seed${seed} --episodes 100 --eval-seed 0
  done
done