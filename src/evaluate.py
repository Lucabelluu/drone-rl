"""
evaluate.py — Valuta una policy addestrata su N episodi di hovering e calcola il Crash Rate.
Randomizza le condizioni iniziali (seeded, identiche per tutti i modelli), normalizza le
osservazioni con le statistiche salvate dal training (VecNormalize), e classifica la fine
come crash / timeout riapplicando le soglie native di HoverAviary sullo stato finale.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from envs.hover_terminal import HoverAviaryTerminal, is_crash

ALGOS = {"ppo": PPO, "sac": SAC}
TARGET_POS = np.array([0.0, 0.0, 1.0])

INIT_XY = 0.25
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25
INIT_TILT = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Valutazione di una policy su HoverAviary (Crash Rate).")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--model", choices=["final", "best"], default="final")
    return p.parse_args()


def load_model(run_dir, which):
    cfg = json.loads((run_dir / "config.json").read_text())
    algo = cfg["algo"]
    model_path = run_dir / ("final_model.zip" if which == "final" else "best_model.zip")
    if not model_path.exists():
        raise SystemExit(f"[STOP] Modello non trovato: {model_path}")
    return ALGOS[algo].load(str(model_path)), algo


def load_normalizer(run_dir):
    """Carica le statistiche di normalizzazione delle osservazioni salvate dal training."""
    stats = run_dir / "vecnormalize.pkl"
    if not stats.exists():
        raise SystemExit(f"[STOP] vecnormalize.pkl non trovato in {run_dir}")
    dummy = DummyVecEnv([lambda: HoverAviaryTerminal(obs=ObservationType("kin"), act=ActionType("rpm"))])
    vec = VecNormalize.load(str(stats), dummy)
    dummy.close()
    rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon
    def normalize(obs):
        return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    return normalize


def sample_init(rng):
    xyz = np.array([[rng.uniform(-INIT_XY, INIT_XY),
                     rng.uniform(-INIT_XY, INIT_XY),
                     rng.uniform(INIT_Z_LOW, INIT_Z_HIGH)]])
    rpy = np.array([[rng.uniform(-INIT_TILT, INIT_TILT),
                     rng.uniform(-INIT_TILT, INIT_TILT),
                     0.0]])
    return xyz, rpy


def run_episode(model, normalize, xyz, rpy, seed):
    env = HoverAviaryTerminal(obs=ObservationType("kin"), act=ActionType("rpm"),
                              initial_xyzs=xyz, initial_rpys=rpy, gui=False)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    ep_return, ep_len = 0.0, 0
    while not (terminated or truncated):
        action, _ = model.predict(normalize(obs), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        ep_len += 1
    state = env._getDroneStateVector(0)
    reason = "crash" if is_crash(state) else "timeout"
    dist = float(np.linalg.norm(TARGET_POS - state[0:3]))
    env.close()
    return reason, ep_len, ep_return, dist


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    model, algo = load_model(run_dir, args.model)
    normalize = load_normalizer(run_dir)

    rng = np.random.default_rng(args.eval_seed)
    inits = [sample_init(rng) for _ in range(args.episodes)]

    rows = []
    counts = {"crash": 0, "timeout": 0}
    for i, (xyz, rpy) in enumerate(inits):
        reason, ep_len, ep_return, dist = run_episode(model, normalize, xyz, rpy, args.eval_seed + i)
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
        "algo": algo, "model": args.model, "episodes": args.episodes,
        "eval_seed": args.eval_seed, "crash_rate_pct": round(crash_rate, 2),
        "counts": counts,
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 4),
        "mean_length": round(float(np.mean([r["length"] for r in rows])), 2),
        "init_randomization": {"xy": INIT_XY, "z": [INIT_Z_LOW, INIT_Z_HIGH], "tilt": INIT_TILT},
    }
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[RESULT] {algo} ({args.model}) — Crash Rate: {crash_rate:.2f}%  "
          f"(crash={counts['crash']}, timeout={counts['timeout']})")
    print(f"[INFO] Scritti eval_episodes.csv e eval_summary.json in {run_dir}")


if __name__ == "__main__":
    main()