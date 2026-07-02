"""
train.py — Addestra UNA policy (un algoritmo, un seed) sull'hovering di HoverAviary,
per un budget fisso di timestep, e salva gli artefatti del run.
DQ1: confronto PPO (on-policy) vs SAC (off-policy). Niente early stopping: budget pieno.
"""
import os
# macOS + conda: PyTorch e lo stack NumPy/SciPy linkano due copie di OpenMP (libomp),
# che vanno in conflitto e abortiscono il processo (OMP Error #15). Va impostato PRIMA
# di importare torch/SB3. Mitigazione nota e innocua per carichi NumPy/PyTorch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from envs.hover_terminal import HoverAviaryTerminal
from envs.hover_shaped import HoverAviaryShaped
from envs.hover_wind import HoverAviaryWind

# Punti voluti sulla curva di apprendimento, a prescindere dal budget.
N_EVAL_POINTS = 50
# Episodi per ogni valutazione DURANTE il training (risoluzione della curva, non il
# Crash Rate finale, che calcoleremo su molti più episodi in fase di valutazione).
N_EVAL_EPISODES = 5

ALGOS = {"ppo": PPO, "sac": SAC}


def parse_args():
    p = argparse.ArgumentParser(description="Training di una policy su HoverAviary (DQ1).")
    p.add_argument("--algo", required=True, choices=["ppo", "sac"],
                   help="Algoritmo: ppo (on-policy) o sac (off-policy).")
    p.add_argument("--seed", required=True, type=int,
                   help="Seed per la riproducibilità.")
    p.add_argument("--timesteps", required=True, type=int,
                   help="Budget di addestramento in timestep (fisso per i confronti).")
    p.add_argument("--output-dir", default="experiments/dq1/results",
                   help="Cartella base dei risultati.")
    p.add_argument("--overwrite", action="store_true",
                   help="Sovrascrive il run se la cartella esiste già.")
    # DQ2: reward shaping. Senza --shaped il comportamento è identico alla DQ1 (HoverAviaryTerminal).
    p.add_argument("--shaped", action="store_true",
                   help="Usa HoverAviaryShaped (reward − lambda*||a_t − a_{t-1}||^2).")
    p.add_argument("--lambda", dest="shaping_lambda", type=float, default=0.0,
                   help="Peso lambda della penalità di shaping (usato solo con --shaped).")
    p.add_argument("--wind-mag", type=float, default=0.0,
                   help="Vento in training (frazione del peso). >0 → domain randomization (DQ3).")
    p.add_argument("--wind-tau", type=float, default=0.8,
                   help="Tempo di correlazione della raffica in training [s] (DQ3).")
    return p.parse_args()


def get_git_commit():
    """Hash del commit corrente, per legare ogni risultato al codice che l'ha prodotto."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def extract_hyperparams(model):
    """Iperparametri principali effettivamente usati (default di SB3), per tracciabilità."""
    keys = ["learning_rate", "gamma", "batch_size", "n_steps", "n_epochs",
            "buffer_size", "tau", "train_freq", "gradient_steps", "ent_coef",
            "learning_starts"]
    hp = {"policy": "MlpPolicy"}
    for k in keys:
        if hasattr(model, k):
            v = getattr(model, k)
            try:
                json.dumps(v)       # tieni il valore se serializzabile in JSON...
                hp[k] = v
            except TypeError:
                hp[k] = str(v)      # ...altrimenti salvane la forma testuale.
    return hp


def main():
    args = parse_args()

    # 1) Riproducibilità globale (Python, NumPy, PyTorch).
    set_random_seed(args.seed)

    # 2) Cartella del run, nome deterministico. Per i run shaped (DQ2) il lambda entra nel nome,
    #    cosi' condizioni con lambda diverso non collidono e il path e' auto-descrittivo.
    if args.wind_mag > 0.0:
        run_name = f"{args.algo}_lam{args.shaping_lambda}_wind{args.wind_mag}_seed{args.seed}"
    elif args.shaped:
        run_name = f"{args.algo}_lam{args.shaping_lambda}_seed{args.seed}"
    else:
        run_name = f"{args.algo}_seed{args.seed}"
    run_dir = Path(args.output_dir) / run_name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"[STOP] {run_dir} esiste già. Usa --overwrite per rifarlo.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3) Carta d'identità del run (scritta subito: c'è anche se il training fallisce).
    config = {
        "algo": args.algo,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "obs": "kin",
        "shaped": args.shaped,
        "shaping_lambda": args.shaping_lambda,
        "wind_mag": args.wind_mag,
        "wind_tau": args.wind_tau,
        "sb3_version": stable_baselines3.__version__,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Config:\n{json.dumps(config, indent=2)}")

    # 4) Ambienti: hovering z=1, osservazione cinematica, azione sui 4 motori (rpm).
    env_kwargs = dict(obs=ObservationType("kin"), act=ActionType("rpm"))
    # L'osservazione KIN è grezza e con limiti infiniti (verificato in BaseRLAviary): SAC diverge
    # senza normalizzazione. VecNormalize su ENTRAMBI = stesso pre-processing, confronto equo.
    # kin = stato cinematico (pos/assetto/velocità); rpm = azione = giri dei 4 motori
    # DQ2: l'ambiente di TRAINING applica lo shaping (se --shaped); quello di VALUTAZIONE resta
    # nativo (vedi sotto), così la curva di controllo riflette l'obiettivo vero ed è confrontabile
    # tra i diversi lambda.
    train_kwargs = dict(env_kwargs)
    if args.wind_mag > 0.0:
        # Policy B (DQ3): vento in training = domain randomization. HoverAviaryWind eredita da
        # HoverAviaryShaped, quindi porta con sé anche la penalità di shaping (stessa reward della DQ2).
        train_kwargs["shaping_lambda"] = args.shaping_lambda
        train_kwargs["wind_mag"] = args.wind_mag
        train_kwargs["wind_tau"] = args.wind_tau
        train_kwargs["wind_seed"] = args.seed
        train_cls = HoverAviaryWind
    elif args.shaped:
        train_kwargs["shaping_lambda"] = args.shaping_lambda
        train_cls = HoverAviaryShaped
    else:
        train_cls = HoverAviaryTerminal
    train_env = make_vec_env(train_cls, env_kwargs=train_kwargs, n_envs=1, seed=args.seed)
    # norm_reward=False: tengo il reward grezzo (0-2) per ritorni interpretabili e comparabili (~480 max)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    eval_env = make_vec_env(HoverAviaryTerminal, env_kwargs=env_kwargs, n_envs=1, seed=args.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # 5) Modello. Default di SB3, con UNA eccezione per SAC: ent_coef fisso.
    #    Su questo ambiente l'auto-tuning dell'entropia di SAC diverge (ent_coef esplode);
    #    lo fissiamo per stabilità. PPO resta sui default (non ne ha bisogno).
    algo_kwargs = dict(seed=args.seed, verbose=1)
    if args.algo == "sac":
        algo_kwargs["ent_coef"] = 0.1
    model = ALGOS[args.algo]("MlpPolicy", train_env, **algo_kwargs)

    # 6) Valutazione SENZA early stopping: logga la curva (evaluations.npz) e salva best_model.
    #    eval_freq derivato dal budget per avere ~N_EVAL_POINTS punti sulla curva.
    eval_freq = max(1000, args.timesteps // N_EVAL_POINTS)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=eval_freq,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # 7) Addestramento per il budget pieno.
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    # 8) Salvataggio del modello finale e chiusura.
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vecnormalize.pkl"))   # statistiche di normalizzazione (servono in valutazione)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()