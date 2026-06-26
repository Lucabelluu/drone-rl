#!/bin/bash
# run_eval_sweep_dq1.sh — Sweep di robustezza DQ1: Crash Rate di PPO e SAC a severità crescente.
# Condizioni iniziali disturbate CON calcio di velocità (--kick), seeded e identiche per tutti i
# modelli (eval-seed 0). 100 episodi per (modello, severità). 5 severità x 2 algo x 3 seed = 30 valutazioni.
for sev in 1 2 3 4 5; do
  for algo in ppo sac; do
    for seed in 0 1 2; do
      echo "=== ${algo} seed ${seed} severity ${sev} ==="
      python src/evaluate.py --run-dir experiments/dq1/results/${algo}_seed${seed} \
        --episodes 100 --eval-seed 0 --severity ${sev} --kick --tag _sev${sev}
    done
  done
done