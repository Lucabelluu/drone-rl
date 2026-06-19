#!/bin/bash
# DQ1: 6 run (PPO e SAC, seed 0/1/2) allo stesso budget.
BUDGET=500000
for algo in ppo sac; do
  for seed in 0 1 2; do
    echo "=============== $algo seed $seed ==============="
    python src/train.py --algo "$algo" --seed "$seed" --timesteps "$BUDGET" --overwrite
  done
done
echo "=== Tutti i run completati ==="