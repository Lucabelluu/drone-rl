"""
HoverAviaryShaped — HoverAviaryTerminal con reward shaping sulla variazione dell'azione.

Estende HoverAviaryTerminal (DQ1) aggiungendo alla reward nativa una penalità quadratica sulla
variazione del comando ai motori tra due passi consecutivi:

    reward_shaped = reward_nativa - lambda * ||a_t - a_{t-1}||^2

Con shaping_lambda=0 il comportamento coincide con HoverAviaryTerminal: la baseline della DQ2
(SAC lambda=0) e' la SAC gia' addestrata in DQ1 e NON va riaddestrata. Con shaping_lambda>0
l'agente e' incentivato a comandi piu' dolci (volo piu' fluido). La penalita' e' calcolata
sull'azione grezza in [-1,1]^4 (act=rpm), la stessa emessa dalla policy.

Scelta progettuale (a_{t-1}): l'ultima azione e' presente nello stato in [16:20], ma al momento
del calcolo della reward quella e' GIA' a_t; a_{t-1} non e' quindi leggibile dallo stato e va
conservato in self._prev_action (aggiornato a fine step, azzerato al reset). Al primo passo di
ogni episodio non esiste un'azione precedente: la penalita' e' 0.

Input  (env_kwargs): obs=ObservationType("kin"), act=ActionType("rpm"), shaping_lambda=float.
Output (per step):   reward shaped; info["shaping_penalty"] = lambda*||a_t - a_{t-1}||^2 (diagnostica).
Uso:   istanziata da src/train.py con --shaped --lambda L. Non eseguibile come script.
"""
import numpy as np

from envs.hover_terminal import HoverAviaryTerminal


class HoverAviaryShaped(HoverAviaryTerminal):
    def __init__(self, shaping_lambda=0.0, **kwargs):
        # Gli argomenti dell'ambiente (drone_model, obs, act, frequenze, ...) sono inoltrati
        # invariati a HoverAviaryTerminal: con shaping_lambda=0 l'ambiente coincide con la DQ1.
        super().__init__(**kwargs)
        self.shaping_lambda = float(shaping_lambda)
        self._prev_action = None    # azione del passo precedente; None = inizio episodio

    def reset(self, seed=None, options=None):
        # Azzeramento della memoria dell'azione precedente: nessuno "scatto" ereditato tra episodi.
        self._prev_action = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # La reward nativa (con la semantica terminated/truncated della DQ1) e' prodotta da super();
        # lo shaping e' un post-processing che non altera la logica nativa dell'ambiente.
        obs, reward, terminated, truncated, info = super().step(action)

        if self._prev_action is None:
            penalty = 0.0   # primo passo dell'episodio: nessuna azione precedente da confrontare
        else:
            delta = np.asarray(action, dtype=float) - self._prev_action
            penalty = self.shaping_lambda * float(np.sum(delta ** 2))   # lambda * ||a_t - a_{t-1}||^2

        self._prev_action = np.asarray(action, dtype=float).copy()
        info["shaping_penalty"] = penalty   # tracciata per la diagnosi (verifica dello smoke test)
        return obs, reward - penalty, terminated, truncated, info