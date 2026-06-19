#!/bin/bash
for sev in 1 2 3 4 5; do
  for algo in ppo sac; do
    for seed in 0 1 2; do
      echo "=== ${algo} seed ${seed} severity ${sev} ==="
      python src/evaluate.py --run-dir experiments/dq1/results/${algo}_seed${seed} \
        --episodes 100 --eval-seed 0 --severity ${sev} --tag _sev${sev}
    done
  done
done