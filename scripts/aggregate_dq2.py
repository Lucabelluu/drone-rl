"""
aggregate_dq2.py — Raccoglie i summary di valutazione DQ2 dei run shaped in un unico CSV.
Legge experiments/dq2/results/<run>/eval_summary_dq2.json per i run elencati e produce
results/tables/dq2_summary.csv, una riga per run con lambda, seed e tutte le metriche.
Lancio:  python scripts/aggregate_dq2.py
Durata:  immediato.
"""
import csv
import json
from pathlib import Path

# Run validi della DQ2 (3 seed per lambda; i seed sostitutivi sono dichiarati nel diario).
RUNS = [
    ("0.1", 0), ("0.1", 1), ("0.1", 2),
    ("0.5", 1), ("0.5", 3), ("0.5", 4),
    ("0.8", 0), ("0.8", 1), ("0.8", 3),
]

RESULTS_DIR = Path("experiments/dq2/results")
OUT = Path("results/tables/dq2_summary.csv")

FIELDS = ["lambda", "seed", "stable_rate_pct", "spin_rate_pct", "crash_rate_pct",
          "settle_dist_median_m", "settle_time_median_s", "fluidity_omega_mean",
          "mech_dact_mean", "episodes"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lam, seed in RUNS:
        f = RESULTS_DIR / f"sac_lam{lam}_seed{seed}" / "eval_summary_dq2.json"
        if not f.exists():
            print(f"[WARN] manca: {f}")
            continue
        d = json.loads(f.read_text())
        rows.append({
            "lambda": lam, "seed": seed,
            "stable_rate_pct": d["stable_rate_pct"],
            "spin_rate_pct": d["spin_rate_pct"],
            "crash_rate_pct": d["crash_rate_pct"],
            "settle_dist_median_m": d["settle_dist_median_m"],
            "settle_time_median_s": d["settle_time_median_s"],
            "fluidity_omega_mean": d["fluidity_omega_mean"],
            "mech_dact_mean": d["mech_dact_mean"],
            "episodes": d["episodes"],
        })
    with open(OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    print(f"[OK] {len(rows)} run -> {OUT}")


if __name__ == "__main__":
    main()