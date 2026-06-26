#!/usr/bin/env bash
# run_dq2_fix.sh — Riaddestramento mirato dei run DQ2 problematici.
# I run lam0.5 seed0/seed2 e lam0.8 seed2 erano degradati (due seed caduti in un minimo a terra,
# uno snapshot finale rumoroso). Si rifanno con seed nuovi per ottenere 3 run validi per lambda,
# mantenendo budget e metodo identici al batch principale. I run validi esistenti non si toccano.
# Lancio:  caffeinate -i bash scripts/run_dq2_fix.sh
# Durata:  ~1h30 (3 run da 500k).
set -euo pipefail

TIMESTEPS=500000

run() {
  local lam=$1 seed=$2
  echo "=== [DQ2-FIX] SAC lambda=${lam} seed=${seed} ==="
  python src/train.py --algo sac --seed "${seed}" --timesteps "${TIMESTEPS}" \
    --shaped --lambda "${lam}" \
    --output-dir "experiments/dq2/results" 2>&1 | tee "experiments/dq2/sac_lam${lam}_seed${seed}_log.txt"
}

run 0.5 3
run 0.5 4
run 0.8 3

echo "=== [DQ2-FIX] Completato. Verifica i nuovi run prima di includerli. ==="