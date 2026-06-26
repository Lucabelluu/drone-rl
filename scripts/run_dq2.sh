#!/usr/bin/env bash
# run_dq2.sh — Batch di addestramento DQ2 (reward shaping su SAC).
# Addestra le condizioni shaped lambda in {0.01, 0.1, 0.5}, seed in {0,1,2}: 9 run da 500k.
# La baseline lambda=0 NON e' qui: e' la SAC della DQ1 (gia' addestrata), riusata come termine
# di confronto. Ogni run scrive in experiments/dq2/results/sac_lam{lambda}_seed{seed}/.
# Lancio:  caffeinate -i bash scripts/run_dq2.sh
# Durata:  ~5 h totali (~33 min/run x 9). Nessun early stopping: budget pieno per tutti.
set -euo pipefail

TIMESTEPS=500000
LAMBDAS=(0.1 0.5 0.8)
SEEDS=(0 1 2)

for lam in "${LAMBDAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="experiments/dq2/results/sac_lam${lam}_seed${seed}"
    echo "=== [DQ2] SAC lambda=${lam} seed=${seed} -> ${out} ==="
    python src/train.py --algo sac --seed "${seed}" --timesteps "${TIMESTEPS}" \
      --shaped --lambda "${lam}" \
      --output-dir "experiments/dq2/results" 2>&1 | tee "experiments/dq2/sac_lam${lam}_seed${seed}_log.txt"
  done
done

echo "=== [DQ2] Batch completato: 9 run in experiments/dq2/results/ ==="