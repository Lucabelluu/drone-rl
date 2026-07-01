"""
HoverAviaryWind — HoverAviaryShaped con vento orizzontale stocastico (Ornstein-Uhlenbeck 2D).

Estende HoverAviaryShaped (DQ2): mantiene il reward shaping (penalità -lambda*||a_t - a_{t-1}||^2)
e aggiunge una forza esterna orizzontale [Fx, Fy, 0] applicata al baricentro del drone, con
intensità e direzione che variano nel tempo secondo due processi di Ornstein-Uhlenbeck (OU)
indipendenti su Fx e Fy. La forza è in WORLD_FRAME (direzione fissa nel mondo, indipendente
dall'orientamento del drone) ed è applicata al baricentro (nessuna coppia parassita): il drone
deve inclinarsi per generare spinta orizzontale e mantenere la posizione.

Il vento NON è osservato dalla policy: _computeObs non è toccato, quindi la policy vede solo lo
stato del proprio corpo e reagisce agli effetti del vento (robustezza reattiva).

Modello del vento (per ciascuna componente, media 0):
    dF = -(1/tau) * F * dt + sigma * sqrt(dt) * N(0,1)
parametrizzato con due grandezze interpretabili:
    wind_mag : deviazione standard stazionaria della forza, in FRAZIONE del peso del drone
               (peso = self.GRAVITY = G*M, in N). wind_mag=0.15 -> raffiche tipiche ~15% del peso.
    wind_tau : tempo di correlazione della raffica [s].
Da cui sigma = (wind_mag * self.GRAVITY) * sqrt(2/tau). L'integrazione è a passo di fisica
(dt = 1/PYB_FREQ), coerente con la frequenza a cui _physics applica le forze.

Riproducibilità: il vento di ogni episodio è generato da un RNG seedato con (wind_seed + indice
episodio). Episodi successivi hanno venti diversi (domain randomization in training), ma la
sequenza è riproducibile dato wind_seed; due modelli valutati con lo stesso wind_seed e lo stesso
ordine di episodi vedono lo STESSO vento (confronto A-vs-B equo).

Uso:
  - Policy B (domain randomization): training con shaping_lambda=0.1 e wind_mag>0.
  - Stress di Policy A (allenata in aria calma): valutazione con wind_mag variato (curva di robustezza).
"""
import numpy as np
import pybullet as p

from envs.hover_shaped import HoverAviaryShaped


class HoverAviaryWind(HoverAviaryShaped):
    def __init__(self, shaping_lambda=0.0, wind_mag=0.0, wind_tau=0.5, wind_seed=0, **kwargs):
        # Attributi del vento impostati PRIMA di super().__init__: se la catena di init invocasse
        # reset(), gli attributi esistono già.
        self._wind_mag_frac = float(wind_mag)   # std stazionaria della forza, in frazione del peso
        self._wind_tau = float(wind_tau)        # tempo di correlazione della raffica [s]
        self._wind_base_seed = int(wind_seed)
        self._wind_episode = -1                 # incrementato a ogni reset -> seed del vento per episodio
        self._wind_fx = 0.0
        self._wind_fy = 0.0
        self._wind_rng = np.random.default_rng(wind_seed)
        super().__init__(shaping_lambda=shaping_lambda, **kwargs)

    def reset(self, seed=None, options=None):
        # Nuovo episodio: vento riseedato (deterministico dato wind_base_seed) e forza azzerata,
        # così nessuna raffica è ereditata dall'episodio precedente.
        self._wind_episode += 1
        self._wind_rng = np.random.default_rng(self._wind_base_seed + self._wind_episode)
        self._wind_fx = 0.0
        self._wind_fy = 0.0
        return super().reset(seed=seed, options=options)

    def _physics(self, rpm, nth_drone):
        # Spinte dei motori (comportamento nativo), poi la forza del vento come forza esterna extra,
        # nello stesso punto del loop di fisica in cui l'ambiente aggiunge drag/downwash/ground effect.
        super()._physics(rpm, nth_drone)
        if self._wind_mag_frac <= 0.0:
            return

        # Avanzamento OU delle due componenti orizzontali (dt = passo di fisica).
        dt = 1.0 / self.PYB_FREQ
        std_force = self._wind_mag_frac * self.GRAVITY          # std stazionaria [N]
        sigma = std_force * np.sqrt(2.0 / self._wind_tau)
        theta = 1.0 / self._wind_tau
        noise = self._wind_rng.standard_normal(2)
        self._wind_fx += -theta * self._wind_fx * dt + sigma * np.sqrt(dt) * float(noise[0])
        self._wind_fy += -theta * self._wind_fy * dt + sigma * np.sqrt(dt) * float(noise[1])

        # Forza al baricentro in WORLD_FRAME: posObj = posizione MONDO del baricentro -> braccio nullo
        # -> nessuna coppia parassita (solo spinta orizzontale pulita).
        base_pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                                       physicsClientId=self.CLIENT)
        p.applyExternalForce(self.DRONE_IDS[nth_drone], -1,
                             forceObj=[float(self._wind_fx), float(self._wind_fy), 0.0],
                             posObj=list(base_pos),
                             flags=p.WORLD_FRAME,
                             physicsClientId=self.CLIENT)