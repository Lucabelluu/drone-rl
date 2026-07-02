set -e
A_DIR=experiments/dq2/results
B_DIR=experiments/dq3/results
WINDS="0.02 0.05 0.08 0.12 0.15"
for S in 0 1 2
do
  for W in $WINDS
  do
    python src/evaluate.py --run-dir $A_DIR/sac_lam0.1_seed$S --episodes 100 --severity 1 --wind-mag $W --tag _dq3_A_w$W
    python src/evaluate.py --run-dir $B_DIR/sac_lam0.1_wind0.08_seed$S --episodes 100 --severity 1 --wind-mag $W --tag _dq3_B_w$W
  done
done
