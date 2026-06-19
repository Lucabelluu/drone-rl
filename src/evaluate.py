"""
evaluate.py — Valuta una policy su N episodi di hovering e calcola il Crash Rate.
La SEVERITÀ delle condizioni iniziali è regolabile con --severity: scala inclinazione,
offset e un "calcio" di velocità lineare/angolare a t=0. Serve a misurare l'inviluppo
di robustezza (Crash Rate vs severità). Condizioni seeded, identiche per tutti i modelli.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import io
import contextlib
import json
from pathlib import Path

import numpy as np
import pybullet as pb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from envs.hover_terminal import HoverAviaryTerminal, is_crash

ALGOS = {"ppo": PPO, "sac": SAC}
TARGET_POS = np.array([0.0, 0.0, 1.0])

# Magnitudini BASE (a severity = 1). La severità le scala linearmente.
INIT_XY = 0.25          # offset orizzontale [m]
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25   # quota iniziale [m] (non scalata)
INIT_TILT = 0.10        # inclinazione roll/pitch [rad]
INIT_LINVEL = 0.30      # calcio velocità lineare [m/s]
INIT_ANGVEL = 0.50      # calcio velocità angolare [rad/s]  <-- il vero stressore

# Tetti per non partire GIÀ oltre le soglie di schianto (1.5 m, 0.4 rad).
XY_CAP = 1.40
TILT_CAP = 0.38


def parse_args():
    ap = argparse.ArgumentParser(description="Valutazione policy su HoverAviary (Crash Rate).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--model", choices=["final", "best"], default="final")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="Scala la severità delle condizioni iniziali (1 = base).")
    ap.add_argument("--tag", default="",
                    help="Suffisso file di output (es. _sev3) per non sovrascrivere.")
    return ap.parse_args()


def load_model(run_dir, which):
    cfg = json.loads((run_dir / "config.json").read_text())
    algo = cfg["algo"]
    model_path = run_dir / ("final_model.zip" if which == "final" else "best_model.zip")
    if not model_path.exists():
        raise SystemExit(f"[STOP] Modello non trovato: {model_path}")
    return ALGOS[algo].load(str(model_path)), algo


def load_normalizer(run_dir):
    stats = run_dir / "vecnormalize.pkl"
    if not stats.exists():
        raise SystemExit(f"[STOP] vecnormalize.pkl non trovato in {run_dir}")
    with contextlib.redirect_stdout(io.StringIO()):
        dummy = DummyVecEnv([lambda: HoverAviaryTerminal(obs=ObservationType("kin"), act=ActionType("rpm"))])
        vec = VecNormalize.load(str(stats), dummy)
        dummy.close()
    rms, clip, eps = vec.obs_rms, vec.clip_obs, vec.epsilon
    def normalize(obs):
        return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    return normalize


def sample_init(rng, severity):
    tilt = min(INIT_TILT * severity, TILT_CAP)
    xy = min(INIT_XY * severity, XY_CAP)
    linvel = INIT_LINVEL * severity
    angvel = INIT_ANGVEL * severity
    xyz = np.array([[rng.uniform(-xy, xy),
                     rng.uniform(-xy, xy),
                     rng.uniform(INIT_Z_LOW, INIT_Z_HIGH)]])
    rpy = np.array([[rng.uniform(-tilt, tilt),
                     rng.uniform(-tilt, tilt),
                     0.0]])
    lin = rng.uniform(-linvel, linvel, size=3)
    ang = rng.uniform(-angvel, angvel, size=3)
    return xyz, rpy, lin, ang


def run_episode(model, normalize, xyz, rpy, lin, ang, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        env = HoverAviaryTerminal(obs=ObservationType("kin"), act=ActionType("rpm"),
                                  initial_xyzs=xyz, initial_rpys=rpy, gui=False)
        obs, _ = env.reset(seed=seed)
    # Calcio di velocità iniziale (disturbo a t=0)
    pb.resetBaseVelocity(int(env.DRONE_IDS[0]),
                         linearVelocity=lin.tolist(),
                         angularVelocity=ang.tolist(),
                         physicsClientId=env.CLIENT)
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
    inits = [sample_init(rng, args.severity) for _ in range(args.episodes)]

    rows = []
    counts = {"crash": 0, "timeout": 0}
    for i, (xyz, rpy, lin, ang) in enumerate(inits):
        reason, ep_len, ep_return, dist = run_episode(model, normalize, xyz, rpy, lin, ang, args.eval_seed + i)
        counts[reason] += 1
        rows.append({"episode": i, "reason": reason, "length": ep_len,
                     "return": round(ep_return, 4), "dist_to_target": round(dist, 4)})

    crash_rate = 100.0 * counts["crash"] / args.episodes

    with open(run_dir / f"eval_episodes{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "reason", "length", "return", "dist_to_target"])
        w.writeheader(); w.writerows(rows)

    summary = {
        "algo": algo, "model": args.model, "episodes": args.episodes,
        "eval_seed": args.eval_seed, "severity": args.severity,
        "crash_rate_pct": round(crash_rate, 2), "counts": counts,
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 4),
        "mean_length": round(float(np.mean([r["length"] for r in rows])), 2),
        "init_randomization": {
            "xy": round(min(INIT_XY * args.severity, XY_CAP), 3),
            "tilt": round(min(INIT_TILT * args.severity, TILT_CAP), 3),
            "linvel": round(INIT_LINVEL * args.severity, 3),
            "angvel": round(INIT_ANGVEL * args.severity, 3),
        },
    }
    with open(run_dir / f"eval_summary{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[RESULT] {algo} sev={args.severity} — Crash Rate: {crash_rate:.2f}%  "
          f"(crash={counts['crash']}, timeout={counts['timeout']}, mean_ret={summary['mean_return']:.1f})")


if __name__ == "__main__":
    main()