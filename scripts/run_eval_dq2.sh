#!/usr/bin/env bash
# run_eval_dq2.sh — Valutazione DQ2 dei 9 modelli validi (3 seed per lambda).
# Condizioni iniziali disturbate, seeded e identiche per tutti (eval-seed 0), severità 1, niente
# calcio (lo stressore di velocità è specifico della DQ1). 100 episodi per modello.
# Produce eval_episodes_dq2.csv e eval_summary_dq2.json in ogni cartella di run.
# Lancio:  bash scripts/run_eval_dq2.sh
# Durata:  ~15-25 min (9 modelli x 100 episodi).
set -euo pipefail

declare -a RUNS=(
  "sac_lam0.1_seed0" "sac_lam0.1_seed1" "sac_lam0.1_seed2"
  "sac_lam0.5_seed1" "sac_lam0.5_seed3" "sac_lam0.5_seed4"
  "sac_lam0.8_seed0" "sac_lam0.8_seed1" "sac_lam0.8_seed3"
)

for run in "${RUNS[@]}"; do
  echo "=== [EVAL-DQ2] ${run} ==="
  python src/evaluate.py --run-dir "experiments/dq2/results/${run}" \
    --episodes 100 --severity 1 --eval-seed 0 --tag _dq2
done

echo "=== [EVAL-DQ2] Completato: 9 summary in experiments/dq2/results/*/eval_summary_dq2.json ==="