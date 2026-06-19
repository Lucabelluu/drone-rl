import sys
import numpy as np
import matplotlib.pyplot as plt

run_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/dq1/results/ppo_seed0"
data = np.load(f"{run_dir}/evaluations.npz")
ts = data["timesteps"]
res = data["results"]            # (n_valutazioni, n_episodi_per_valutazione)
mean = res.mean(axis=1)
std = res.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(ts, mean, label="eval reward (media)")
plt.fill_between(ts, mean - std, mean + std, alpha=0.2)
plt.xlabel("Timesteps")
plt.ylabel("Eval reward")
plt.title(f"Curva di apprendimento — {run_dir}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("assets/calibrazione/ppo_seed0_curva_1M.png", dpi=120, bbox_inches="tight")
plt.show()