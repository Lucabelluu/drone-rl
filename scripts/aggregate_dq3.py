"""Raccoglie i riepiloghi della valutazione DQ3 (A e B, 3 seed, 5 venti) in un unico CSV.
A = SAC lam0.1 allenata in aria calma (experiments/dq2); B = SAC lam0.1 + vento 0.08 in training
(experiments/dq3). Sorgente per la curva di robustezza Crash Rate vs wind_mag."""
import json
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
WINDS = ["0.02", "0.05", "0.08", "0.12", "0.15"]
SEEDS = [0, 1, 2]
rows = []

for seed in SEEDS:
    a_dir = ROOT / "experiments/dq2/results" / f"sac_lam0.1_seed{seed}"
    for w in WINDS:
        f = a_dir / f"eval_summary_dq3_A_w{w}.json"
        d = json.loads(f.read_text())
        rows.append({"policy": "A", "seed": seed, "wind_mag": float(w),
                     "crash_pct": d["crash_rate_pct"], "spin_pct": d["spin_rate_pct"],
                     "stable_pct": d["stable_rate_pct"], "mean_return": d["mean_return"]})

for seed in SEEDS:
    b_dir = ROOT / "experiments/dq3/results" / f"sac_lam0.1_wind0.08_seed{seed}"
    for w in WINDS:
        f = b_dir / f"eval_summary_dq3_B_w{w}.json"
        d = json.loads(f.read_text())
        rows.append({"policy": "B", "seed": seed, "wind_mag": float(w),
                     "crash_pct": d["crash_rate_pct"], "spin_pct": d["spin_rate_pct"],
                     "stable_pct": d["stable_rate_pct"], "mean_return": d["mean_return"]})

out = ROOT / "results/tables/dq3_robustness.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fp:
    wr = csv.DictWriter(fp, fieldnames=["policy", "seed", "wind_mag", "crash_pct",
                                        "spin_pct", "stable_pct", "mean_return"])
    wr.writeheader()
    wr.writerows(rows)
print(f"[OK] Scritte {len(rows)} righe in {out}")
