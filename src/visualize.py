"""
visualize.py — Riproduce visivamente una policy su un episodio di hovering e (opzionale) ne
registra un video MP4. Strumento di DIMOSTRAZIONE, separato da evaluate.py (che resta headless
e batch): qui si apre la GUI di PyBullet per vedere il drone in volo.

La logica di gioco è identica a evaluate.py — stessa normalizzazione manuale da vecnormalize.pkl,
stesso "calcio" di velocità iniziale seeded — così la clip mostra lo stesso comportamento che le
metriche quantificano. La partenza è disturbata e riproducibile (--episode-seed), quindi modelli
diversi possono essere confrontati sulla STESSA condizione iniziale.

Input:  --run-dir (final_model.zip/best_model.zip, config.json, vecnormalize.pkl).
Output: a schermo la simulazione; con --record, un MP4 nel percorso indicato da --out.
Lancio: python src/visualize.py --run-dir <dir> --record --out results/assets/clip.mp4
Durata: ~10-15 s per episodio (gira circa in tempo reale).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
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

# Stessa partenza disturbata di evaluate.py a severity=1 (riusata per coerenza visiva/metrica).
INIT_XY = 0.25
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25
INIT_TILT = 0.10
INIT_LINVEL = 0.30
INIT_ANGVEL = 0.50
XY_CAP = 1.40
TILT_CAP = 0.38


def parse_args():
    ap = argparse.ArgumentParser(description="Visualizza/registra una policy su HoverAviary.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", choices=["final", "best"], default="final")
    ap.add_argument("--episode-seed", type=int, default=0,
                    help="Seed della partenza disturbata: stesso valore = stessa condizione iniziale.")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--record", action="store_true",
                    help="Registra un MP4 dell'episodio (richiede --out).")
    ap.add_argument("--out", default="results/assets/clip.mp4",
                    help="Percorso del file MP4 di output (con --record).")
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
    xyz = np.array([[rng.uniform(-xy, xy), rng.uniform(-xy, xy),
                     rng.uniform(INIT_Z_LOW, INIT_Z_HIGH)]])
    rpy = np.array([[rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), 0.0]])
    lin = rng.uniform(-INIT_LINVEL * severity, INIT_LINVEL * severity, size=3)
    ang = rng.uniform(-INIT_ANGVEL * severity, INIT_ANGVEL * severity, size=3)
    return xyz, rpy, lin, ang


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    model, algo = load_model(run_dir, args.model)
    normalize = load_normalizer(run_dir)

    rng = np.random.default_rng(args.episode_seed)
    xyz, rpy, lin, ang = sample_init(rng, args.severity)

    # Ambiente con GUI: la finestra di PyBullet mostra il drone in volo.
    env = HoverAviaryTerminal(obs=ObservationType("kin"), act=ActionType("rpm"),
                              initial_xyzs=xyz, initial_rpys=rpy, gui=True)

    # Registrazione MP4 nativa di PyBullet: va avviata dopo la creazione del client e prima del volo.
    log_id = None
    if args.record:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        log_id = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, str(out),
                                      physicsClientId=env.CLIENT)

    obs, _ = env.reset(seed=args.episode_seed)
    pb.resetBaseVelocity(int(env.DRONE_IDS[0]),
                         linearVelocity=lin.tolist(), angularVelocity=ang.tolist(),
                         physicsClientId=env.CLIENT)

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(normalize(obs), deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

    reason = "crash" if is_crash(env._getDroneStateVector(0)) else "timeout"
    if log_id is not None:
        pb.stopStateLogging(log_id, physicsClientId=env.CLIENT)
        print(f"[VIDEO] Salvato: {args.out}")
    env.close()
    print(f"[VISUAL] {algo}  esito={reason}  seed={args.episode_seed}")


if __name__ == "__main__":
    main()