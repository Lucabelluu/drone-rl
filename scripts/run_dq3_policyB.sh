set -e
for S in 1 2
do
  python src/train.py --algo sac --seed $S --timesteps 500000 --shaped --lambda 0.1 --wind-mag 0.08 --output-dir experiments/dq3/results
done
