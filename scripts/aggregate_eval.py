import json, csv, glob, re, os
from pathlib import Path

rows = []
for f in sorted(glob.glob("experiments/dq1/results/*/eval_summary_sev*.json")):
    d = json.loads(Path(f).read_text())
    m = re.match(r"(\w+)_seed(\d+)", Path(f).parent.name)
    rows.append({"algo": m.group(1), "seed": int(m.group(2)), "severity": d["severity"],
                 "crash_rate_pct": d["crash_rate_pct"], "mean_return": d["mean_return"],
                 "mean_length": d["mean_length"]})
rows.sort(key=lambda r: (r["algo"], r["seed"], r["severity"]))
os.makedirs("experiments/dq1", exist_ok=True)
with open("experiments/dq1/eval_sweep_results.csv", "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["algo","seed","severity","crash_rate_pct","mean_return","mean_length"])
    w.writeheader(); w.writerows(rows)
print(f"OK: {len(rows)} righe -> experiments/dq1/eval_sweep_results.csv")