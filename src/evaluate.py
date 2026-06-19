"""
evaluate.py — Valuta una policy addestrata su N episodi di hovering e calcola il Crash Rate.
Per ogni episodio randomizza le condizioni iniziali (seeded, identiche per tutti i modelli),
poi classifica la fine come crash / timeout / target riapplicando le soglie native di
HoverAviary. Salva un log per-episodio (CSV) e un riepilogo (JSON) nella cartella del run.
"""
import os
# Stesso conflitto OpenMP di train.py: va disinnescato PRIMA di importare torch/SB3.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

ALGOS = {"ppo": PPO, "sac": SAC}
TARGET_POS = np.array([0.0, 0.0, 1.0])

# Ampiezza della randomizzazione iniziale (dentro l'inviluppo di sicurezza).
INIT_XY = 0.25                          # offset orizzontale max in m
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25    # quota iniziale in m
INIT_TILT = 0.1                         # roll/pitch iniziale max in rad (crash a 0.4)


def is_crash(state):
    """Riapplica le 5 soglie native di HoverAviary._computeTruncated sullo stato grezzo."""
    x, y, z = state[0], state[1], state[2]
    roll, pitch = state[7], state[8]
    return bool(abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0
                or abs(roll) > 0.4 or abs(pitch) > 0.4)


def parse_args():
    p = argparse.ArgumentParser(description="Valutazione di una policy su HoverAviary (Crash Rate).")
    p.add_argument("--run-dir", required=True,
                   help="Cartella del run da valutare (es. experiments/dq1/results/ppo_seed0).")
    p.add_argument("--episodes", type=int, default=100,
                   help="Numero di episodi di valutazione (default 100).")
    p.add_argument("--eval-seed", type=int, default=0,
                   help="Seed che fissa le condizioni iniziali (uguale per tutti i modelli).")
    p.add_argument("--model", choices=["final", "best"], default="final",
                   help="Quale modello valutare (default: final).")
    return p.parse_args()


def load_model(run_dir, which):
    """Carica il modello giusto leggendo l'algoritmo da config.json."""
    cfg = json.loads((run_dir / "config.json").read_text())
    algo = cfg["algo"]
    model_path = run_dir / ("final_model.zip" if which == "final" else "best_model.zip")
    if not model_path.exists():
        raise SystemExit(f"[STOP] Modello non trovato: {model_path}")
    return ALGOS[algo].load(str(model_path)), algo


def sample_init(rng):
    """Una condizione iniziale (posizione + assetto), forma (1,3) richiesta da HoverAviary."""
    xyz = np.array([[rng.uniform(-INIT_XY, INIT_XY),
                     rng.uniform(-INIT_XY, INIT_XY),
                     rng.uniform(INIT_Z_LOW, INIT_Z_HIGH)]])
    rpy = np.array([[rng.uniform(-INIT_TILT, INIT_TILT),
                     rng.uniform(-INIT_TILT, INIT_TILT),
                     0.0]])
    return xyz, rpy


def run_episode(model, xyz, rpy, seed):
    """Un episodio deterministico; ritorna (motivo, passi, ritorno, distanza finale)."""
    env = HoverAviary(obs=ObservationType("kin"), act=ActionType("rpm"),
                      initial_xyzs=xyz, initial_rpys=rpy, gui=False)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    ep_return, ep_len = 0.0, 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        ep_len += 1
    state = env._getDroneStateVector(0)
    if terminated:
        reason = "target"
    elif is_crash(state):
        reason = "crash"
    else:
        reason = "timeout"
    dist = float(np.linalg.norm(TARGET_POS - state[0:3]))
    env.close()
    return reason, ep_len, ep_return, dist


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    model, algo = load_model(run_dir, args.model)

    # RNG dedicato: le condizioni iniziali sono identiche per ogni modello valutato.
    rng = np.random.default_rng(args.eval_seed)
    inits = [sample_init(rng) for _ in range(args.episodes)]

    rows = []
    counts = {"crash": 0, "timeout": 0, "target": 0}
    for i, (xyz, rpy) in enumerate(inits):
        reason, ep_len, ep_return, dist = run_episode(model, xyz, rpy, args.eval_seed + i)
        counts[reason] += 1
        rows.append({"episode": i, "reason": reason, "length": ep_len,
                     "return": round(ep_return, 4), "dist_to_target": round(dist, 4)})
        print(f"  ep {i:3d}: {reason:8s} len={ep_len:3d} return={ep_return:8.2f} dist={dist:.3f}")

    crash_rate = 100.0 * counts["crash"] / args.episodes

    with open(run_dir / "eval_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reason", "length", "return", "dist_to_target"])
        w.writeheader()
        w.writerows(rows)

    summary = {
        "algo": algo,
        "model": args.model,
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "crash_rate_pct": round(crash_rate, 2),
        "counts": counts,
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 4),
        "mean_length": round(float(np.mean([r["length"] for r in rows])), 2),
        "init_randomization": {"xy": INIT_XY, "z": [INIT_Z_LOW, INIT_Z_HIGH], "tilt": INIT_TILT},
    }
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[RESULT] {algo} ({args.model}) — Crash Rate: {crash_rate:.2f}%  "
          f"(crash={counts['crash']}, timeout={counts['timeout']}, target={counts['target']})")
    print(f"[INFO] Scritti eval_episodes.csv e eval_summary.json in {run_dir}")


if __name__ == "__main__":
    main()