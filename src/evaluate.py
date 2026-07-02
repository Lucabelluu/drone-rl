"""
evaluate.py — Valuta una policy su N episodi di hovering e calcola le metriche di robustezza (DQ1)
e di qualità del volo (DQ2). Le condizioni iniziali sono disturbate e seeded (identiche per tutti i
modelli), con intensità regolabile da --severity. Il "calcio" di velocità a t=0 (stressore della
DQ1) è opzionale via --kick.

Normalizzazione: il modello gioca DENTRO un VecNormalize ricaricato da vecnormalize.pkl, lo stesso
oggetto usato in training; la policy vede le osservazioni nella scala con cui ha imparato. Le
metriche di stato (distanza dal target, ‖ω‖) sono lette dall'ambiente interno NON normalizzato, in
unità fisiche reali (m, rad/s), e registrate PRIMA di ogni step per non leggere lo stato del reset
automatico del VecEnv a fine episodio.

Metriche DQ2 (continue, senza soglia di "arrivato sì/no"): su ogni episodio che non schianta si
misura, sulla FINESTRA FINALE di WINDOW_SEC secondi:
  - settle_dist  = mediana della distanza dal target nella finestra [m]  -> quanto vicino si assesta;
  - omega_final  = media di ‖ω‖ nella finestra [rad/s]                    -> fluidità (bassa = liscio);
  - dact_final   = media di ‖aₜ − aₜ₋₁‖ nella finestra                    -> meccanismo (la grandezza
                   penalizzata in training; prova che lo shaping agisce sui comandi, non misura di
                   fluidità);
  - settle_time  = istante da cui la distanza resta entro SETTLE_TOL dal suo valore finale [s]
                   -> tempo di assestamento (definito sul comportamento del modello, non su δ assoluto).
Un episodio è classificato "spin" (avvitamento) se omega_final >= OMEGA_SPIN: il drone resta in volo
ma ruota su sé stesso invece di stabilizzarsi. Le metriche continue sono aggregate SOLO sugli episodi
stabili (non crash, non spin); distanza e tempo come MEDIANA. I tassi (crash, spin, stable) descrivono
la ripartizione dei comportamenti.

Input:  --run-dir (final_model.zip/best_model.zip, config.json, vecnormalize.pkl).
Output: eval_episodes{tag}.csv (per-episodio) e eval_summary{tag}.json (aggregato) nella run-dir.
Lancio (DQ2): python src/evaluate.py --run-dir <dir> --episodes 100 --severity 1 --tag _dq2
Lancio (DQ1): python src/evaluate.py --run-dir <dir> --episodes 100 --severity 3 --kick --tag _sev3
Durata: ~1-2 min per 100 episodi.
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
from envs.hover_wind import HoverAviaryWind

ALGOS = {"ppo": PPO, "sac": SAC}
TARGET_POS = np.array([0.0, 0.0, 1.0])

# Magnitudini BASE (a severity = 1). La severità le scala linearmente.
INIT_XY = 0.25          # offset orizzontale [m]
INIT_Z_LOW, INIT_Z_HIGH = 0.75, 1.25   # quota iniziale [m] (non scalata)
INIT_TILT = 0.10        # inclinazione roll/pitch [rad]
INIT_LINVEL = 0.30      # calcio velocità lineare [m/s]   (solo con --kick)
INIT_ANGVEL = 0.50      # calcio velocità angolare [rad/s] (solo con --kick)

# Tetti per non partire GIÀ oltre le soglie di schianto (1.5 m, 0.4 rad).
XY_CAP = 1.40
TILT_CAP = 0.38

# DQ2 — finestra finale e classificazione del comportamento.
WINDOW_SEC = 1.0        # durata della finestra finale su cui si misurano le metriche continue [s]
OMEGA_SPIN = 1.0        # soglia di ‖ω‖ oltre cui l'episodio è "spin" (avvitamento) [rad/s]
SETTLE_TOL = 0.10       # tolleranza per il tempo di assestamento: la distanza è "ferma" entro ±SETTLE_TOL
                        # dal suo valore finale [m]


def parse_args():
    ap = argparse.ArgumentParser(description="Valutazione policy su HoverAviary (DQ1 Crash Rate + DQ2).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--model", choices=["final", "best"], default="final")
    ap.add_argument("--severity", type=float, default=1.0,
                    help="Scala la severità delle condizioni iniziali (1 = base).")
    ap.add_argument("--kick", action="store_true",
                    help="Aggiunge il calcio di velocità lineare/angolare a t=0 (stressore DQ1).")
    ap.add_argument("--tag", default="",
                    help="Suffisso file di output (es. _dq2) per non sovrascrivere.")
    ap.add_argument("--wind-mag", type=float, default=0.0,
                    help="Intensità del vento (frazione del peso). >0 attiva HoverAviaryWind (DQ3).")
    ap.add_argument("--wind-tau", type=float, default=0.8,
                    help="Tempo di correlazione della raffica [s] (DQ3).")
    return ap.parse_args()


def read_algo(run_dir):
    return json.loads((run_dir / "config.json").read_text())["algo"]


def load_model(run_dir, which, algo):
    model_path = run_dir / ("final_model.zip" if which == "final" else "best_model.zip")
    if not model_path.exists():
        raise SystemExit(f"[STOP] Modello non trovato: {model_path}")
    return ALGOS[algo].load(str(model_path))


def make_norm_env(run_dir, xyz, rpy, wind_mag=0.0, wind_tau=0.8, wind_seed=0):
    """Ambiente di valutazione avvolto nel VecNormalize del training (statistiche congelate).
    wind_mag=0 -> HoverAviaryTerminal (DQ1/DQ2, comportamento invariato).
    wind_mag>0 -> HoverAviaryWind con vento stocastico non osservato dalla policy (DQ3).
    Il vento è calcolato sullo stato fisico, quindi la penalità di shaping (ereditata da Shaped)
    è irrilevante in valutazione: shaping_lambda resta al default 0."""
    stats = run_dir / "vecnormalize.pkl"
    if not stats.exists():
        raise SystemExit(f"[STOP] vecnormalize.pkl non trovato in {run_dir}")
    with contextlib.redirect_stdout(io.StringIO()):
        if wind_mag > 0.0:
            venv = DummyVecEnv([lambda: HoverAviaryWind(
                obs=ObservationType("kin"), act=ActionType("rpm"),
                initial_xyzs=xyz, initial_rpys=rpy, gui=False,
                wind_mag=wind_mag, wind_tau=wind_tau, wind_seed=wind_seed)])
        else:
            venv = DummyVecEnv([lambda: HoverAviaryTerminal(
                obs=ObservationType("kin"), act=ActionType("rpm"),
                initial_xyzs=xyz, initial_rpys=rpy, gui=False)])
        venv = VecNormalize.load(str(stats), venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def sample_init(rng, severity):
    """Condizione iniziale disturbata: offset di posizione, inclinazione roll/pitch e (se richiesto)
    calcio di velocità. I tetti XY_CAP/TILT_CAP evitano partenze già oltre le soglie di schianto."""
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


def settle_time_of(dists, dist_final, ctrl_freq):
    """Tempo di assestamento: primo istante [s] dopo il quale la distanza resta entro ±SETTLE_TOL dal
    suo valore finale fino a fine episodio. Definito sul comportamento del modello, non su una soglia
    di distanza assoluta dal target."""
    d = np.asarray(dists)
    for t in range(len(d)):
        if np.all(np.abs(d[t:] - dist_final) < SETTLE_TOL):
            return t / ctrl_freq
    return None


def run_episode(model, run_dir, xyz, rpy, lin, ang, kick, wind_mag=0.0, wind_tau=0.8, wind_seed=0):
    """Gioca un episodio deterministico dalla partenza (xyz, rpy), con calcio (lin, ang) se kick=True.
    Lo stato è registrato PRIMA di ogni step, così l'ultimo campione è l'ultimo stato reale e non lo
    stato del reset automatico del VecEnv. Calcola le metriche sulla finestra finale di WINDOW_SEC."""
    venv = make_norm_env(run_dir, xyz, rpy, wind_mag=wind_mag, wind_tau=wind_tau, wind_seed=wind_seed)
    obs = venv.reset()
    inner = venv.envs[0]   # ambiente interno: stato fisico reale, non normalizzato

    if kick:
        pb.resetBaseVelocity(int(inner.DRONE_IDS[0]),
                             linearVelocity=lin.tolist(),
                             angularVelocity=ang.tolist(),
                             physicsClientId=inner.CLIENT)

    ep_return, ep_len = 0.0, 0
    dists, omegas, actions = [], [], []
    done = [False]
    while not done[0]:
        state = inner._getDroneStateVector(0)   # registrato PRIMA dello step (evita lo stato post-reset)
        dists.append(float(np.linalg.norm(TARGET_POS - state[0:3])))
        omegas.append(float(np.linalg.norm(state[13:16])))
        action, _ = model.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=float).flatten())
        obs, reward, done, info = venv.step(action)
        ep_return += float(reward[0])
        ep_len += 1

    ctrl_freq = inner.CTRL_FREQ
    venv.close()

    d = np.asarray(dists)
    w = np.asarray(omegas)
    dist_final = float(d[-1])

    # Schianto: l'ultimo stato registrato (pre-step finale) è fuori dall'inviluppo di volo sicuro.
    # is_crash lavora sul vettore di stato; lo ricostruiamo implicitamente dai criteri su dist e tilt
    # non è affidabile, quindi usiamo la lunghezza episodio: un episodio interrotto presto = schianto.
    crashed = ep_len < int(0.95 * (inner.EPISODE_LEN_SEC * ctrl_freq))

    w_steps = int(round(WINDOW_SEC * ctrl_freq))
    win_d = d[-w_steps:]
    win_w = w[-w_steps:]
    win_settle_dist = float(np.median(win_d))
    win_omega = float(np.mean(win_w))
    spin = (not crashed) and (win_omega >= OMEGA_SPIN)
    stable = (not crashed) and (not spin)

    base = {"length": ep_len, "return": round(ep_return, 4),
            "dist_final": round(dist_final, 4),
            "crashed": crashed, "spin": spin, "stable": stable}

    if stable:
        # ‖Δa‖ sulla finestra finale (differenza definita da t>=1 nella finestra)
        win_actions = actions[-w_steps:]
        diffs = [float(np.linalg.norm(win_actions[i] - win_actions[i - 1]))
                 for i in range(1, len(win_actions))]
        dact = float(np.mean(diffs)) if diffs else 0.0
        st = settle_time_of(dists, dist_final, ctrl_freq)
        base.update(settle_dist=round(win_settle_dist, 4),
                    omega_final=round(win_omega, 6),
                    dact_final=round(dact, 6),
                    settle_time_s=round(st, 4) if st is not None else None)
    else:
        base.update(settle_dist=None, omega_final=round(win_omega, 6),
                    dact_final=None, settle_time_s=None)
    return base


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    algo = read_algo(run_dir)
    model = load_model(run_dir, args.model, algo)

    # Condizioni iniziali generate una volta dal solo eval-seed: identiche per ogni modello valutato.
    rng = np.random.default_rng(args.eval_seed)
    inits = [sample_init(rng, args.severity) for _ in range(args.episodes)]

    rows = []
    for i, (xyz, rpy, lin, ang) in enumerate(inits):
        # wind_seed distinto per episodio (ma deterministico dato eval-seed): episodi con venti
        # diversi, IDENTICI però tra modelli valutati con lo stesso eval-seed -> confronto A-vs-B equo.
        r = run_episode(model, run_dir, xyz, rpy, lin, ang, args.kick,
                        wind_mag=args.wind_mag, wind_tau=args.wind_tau,
                        wind_seed=1000 * args.eval_seed + i)
        r["episode"] = i
        rows.append(r)

    n = args.episodes
    n_crash = sum(r["crashed"] for r in rows)
    n_spin = sum(r["spin"] for r in rows)
    stable_rows = [r for r in rows if r["stable"]]
    n_stable = len(stable_rows)

    def med(key):
        vals = [r[key] for r in stable_rows if r[key] is not None]
        return float(np.median(vals)) if vals else None

    def mean(key):
        vals = [r[key] for r in stable_rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "algo": algo, "model": args.model, "episodes": n,
        "eval_seed": args.eval_seed, "severity": args.severity, "kick": args.kick,
        "wind_mag": args.wind_mag, "wind_tau": args.wind_tau,
        "crash_rate_pct": round(100.0 * n_crash / n, 2),
        "spin_rate_pct": round(100.0 * n_spin / n, 2),
        "stable_rate_pct": round(100.0 * n_stable / n, 2),
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 4),
        "mean_length": round(float(np.mean([r["length"] for r in rows])), 2),
        "settle_dist_median_m": round(med("settle_dist"), 4) if med("settle_dist") is not None else None,
        "settle_time_median_s": round(med("settle_time_s"), 4) if med("settle_time_s") is not None else None,
        "fluidity_omega_mean": round(mean("omega_final"), 6) if mean("omega_final") is not None else None,
        "mech_dact_mean": round(mean("dact_final"), 6) if mean("dact_final") is not None else None,
        "window_sec": WINDOW_SEC, "omega_spin_rad_s": OMEGA_SPIN, "settle_tol_m": SETTLE_TOL,
        "init_randomization": {
            "xy": round(min(INIT_XY * args.severity, XY_CAP), 3),
            "tilt": round(min(INIT_TILT * args.severity, TILT_CAP), 3),
            "linvel": round(INIT_LINVEL * args.severity, 3) if args.kick else 0.0,
            "angvel": round(INIT_ANGVEL * args.severity, 3) if args.kick else 0.0,
        },
    }
    with open(run_dir / f"eval_summary{args.tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = ["episode", "length", "return", "dist_final", "crashed", "spin", "stable",
                  "settle_dist", "omega_final", "dact_final", "settle_time_s"]
    with open(run_dir / f"eval_episodes{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})

    sd = f"{summary['settle_dist_median_m']:.2f}m" if summary["settle_dist_median_m"] is not None else "n/d"
    om = f"{summary['fluidity_omega_mean']:.3f}" if summary["fluidity_omega_mean"] is not None else "n/d"
    print(f"[RESULT] {algo} λ-run sev={args.severity} kick={args.kick} — "
          f"crash {summary['crash_rate_pct']:.0f}%  spin {summary['spin_rate_pct']:.0f}%  "
          f"stable {summary['stable_rate_pct']:.0f}%  | settle_dist {sd}  ‖ω‖ {om}")


if __name__ == "__main__":
    main()