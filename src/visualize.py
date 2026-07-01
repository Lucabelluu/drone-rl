"""
visualize.py — Riproduce una policy su un episodio di hovering e ne registra un video MP4 alla
velocità reale. Strumento di DIMOSTRAZIONE, separato da evaluate.py (headless e batch).

La logica di gioco è identica a evaluate.py: il modello gioca dentro il VecNormalize ricaricato dal
training (stessa scala di osservazioni con cui ha imparato) e la partenza è disturbata e riproducibile
(--episode-seed), così modelli diversi si confrontano sulla STESSA condizione iniziale. Il calcio di
velocità (stressore DQ1) è opzionale via --kick; per le clip DQ2 si lascia disattivato.

Il video è costruito catturando un fotogramma a ogni passo di controllo con la telecamera di PyBullet
e scrivendolo a CTRL_FREQ fotogrammi al secondo: la clip scorre quindi alla stessa velocità del volo
reale (~8 s), a differenza della registrazione nativa di PyBullet che accelera la simulazione.

Input:  --run-dir (final_model.zip/best_model.zip, config.json, vecnormalize.pkl).
Output: con --record, un MP4 nel percorso indicato da --out.
Lancio: python src/visualize.py --run-dir <dir> --record --out results/videos/clip.mp4
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
import imageio
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from envs.hover_terminal import HoverAviaryTerminal, is_crash

ALGOS = {"ppo": PPO, "sac": SAC}
TARGET_POS = np.array([0.0, 0.0, 1.0])

INIT_XY = 0.25
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25
INIT_TILT = 0.10
INIT_LINVEL = 0.30
INIT_ANGVEL = 0.50
XY_CAP = 1.40
TILT_CAP = 0.38

# Telecamera fissa che inquadra la zona di volo attorno al target [0,0,1].
CAM_EYE = [1.3, -1.3, 1.4]
CAM_TARGET = [0.0, 0.0, 0.8]
CAM_W, CAM_H = 640, 480


def parse_args():
    ap = argparse.ArgumentParser(description="Visualizza/registra una policy su HoverAviary.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model", choices=["final", "best"], default="final")
    ap.add_argument("--episode-seed", type=int, default=0,
                    help="Seed della partenza disturbata: stesso valore = stessa condizione iniziale.")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--kick", action="store_true",
                    help="Aggiunge il calcio di velocità iniziale (stressore DQ1). Off per le clip DQ2.")
    ap.add_argument("--record", action="store_true", help="Registra un MP4 (richiede --out).")
    ap.add_argument("--out", default="results/videos/clip.mp4", help="Percorso MP4 di output.")
    return ap.parse_args()


def read_algo(run_dir):
    return json.loads((run_dir / "config.json").read_text())["algo"]


def load_model(run_dir, which, algo):
    model_path = run_dir / ("final_model.zip" if which == "final" else "best_model.zip")
    if not model_path.exists():
        raise SystemExit(f"[STOP] Modello non trovato: {model_path}")
    return ALGOS[algo].load(str(model_path))


def sample_init(rng, severity):
    tilt = min(INIT_TILT * severity, TILT_CAP)
    xy = min(INIT_XY * severity, XY_CAP)
    xyz = np.array([[rng.uniform(-xy, xy), rng.uniform(-xy, xy),
                     rng.uniform(INIT_Z_LOW, INIT_Z_HIGH)]])
    rpy = np.array([[rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), 0.0]])
    lin = rng.uniform(-INIT_LINVEL * severity, INIT_LINVEL * severity, size=3)
    ang = rng.uniform(-INIT_ANGVEL * severity, INIT_ANGVEL * severity, size=3)
    return xyz, rpy, lin, ang


def grab_frame(client):
    """Cattura un fotogramma RGB dalla telecamera fissa sulla scena."""
    view = pb.computeViewMatrix(cameraEyePosition=CAM_EYE, cameraTargetPosition=CAM_TARGET,
                                cameraUpVector=[0, 0, 1])
    proj = pb.computeProjectionMatrixFOV(fov=60, aspect=CAM_W / CAM_H, nearVal=0.1, farVal=10)
    try:
        _, _, rgb, _, _ = pb.getCameraImage(CAM_W, CAM_H, view, proj,
                                            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                                            physicsClientId=client)
    except pb.error:
        _, _, rgb, _, _ = pb.getCameraImage(CAM_W, CAM_H, view, proj,
                                            renderer=pb.ER_TINY_RENDERER,
                                            physicsClientId=client)
    return np.reshape(rgb, (CAM_H, CAM_W, 4))[:, :, :3].astype(np.uint8)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    algo = read_algo(run_dir)
    model = load_model(run_dir, args.model, algo)

    rng = np.random.default_rng(args.episode_seed)
    xyz, rpy, lin, ang = sample_init(rng, args.severity)

    # Ambiente headless (gui=False): i fotogrammi si catturano via getCameraImage, non serve la GUI.
    stats = run_dir / "vecnormalize.pkl"
    if not stats.exists():
        raise SystemExit(f"[STOP] vecnormalize.pkl non trovato in {run_dir}")
    with contextlib.redirect_stdout(io.StringIO()):
        venv = DummyVecEnv([lambda: HoverAviaryTerminal(
            obs=ObservationType("kin"), act=ActionType("rpm"),
            initial_xyzs=xyz, initial_rpys=rpy, gui=False)])
        venv = VecNormalize.load(str(stats), venv)
    venv.training = False
    venv.norm_reward = False
    inner = venv.envs[0]
    ctrl_freq = inner.CTRL_FREQ
    

    obs = venv.reset()
    # Marcatore visivo del punto di hover target [0,0,1]: una sfera rossa semitrasparente fissa,
    # così nel video si vede dove il drone dovrebbe stabilizzarsi.
    vis = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 0.6],
                               physicsClientId=inner.CLIENT)
    pb.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                       basePosition=TARGET_POS.tolist(), physicsClientId=inner.CLIENT)
    if args.kick:
        pb.resetBaseVelocity(int(inner.DRONE_IDS[0]),
                             linearVelocity=lin.tolist(), angularVelocity=ang.tolist(),
                             physicsClientId=inner.CLIENT)

    frames = []
    crashed = False
    done = [False]
    while not done[0]:
        if args.record:
            frames.append(grab_frame(inner.CLIENT))
        # stato PRIMA dello step: cattura la condizione di schianto prima del reset del VecEnv
        if is_crash(inner._getDroneStateVector(0)):
            crashed = True
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)

    reason = "crash" if crashed else "timeout"
    venv.close()

    if args.record and frames:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), frames, fps=int(ctrl_freq))
        print(f"[VIDEO] Salvato: {out}  ({len(frames)} fotogrammi, {len(frames)/ctrl_freq:.1f} s)")
    print(f"[VISUAL] {algo}  esito={reason}  seed={args.episode_seed}")


if __name__ == "__main__":
    main()