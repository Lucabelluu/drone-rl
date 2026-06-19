"""
HoverAviaryTerminal — HoverAviary con semantica di fine episodio corretta per l'RL off-policy.
Lo SCHIANTO diventa `terminated` (valore futuro = 0), non `truncated`: così SAC non stima più
il valore di stati già schiantati, e i valori Q smettono di divergere. Solo il tempo scaduto
resta `truncated`. Usata sia in training che in valutazione.
"""
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


def is_crash(state):
    """Le 5 soglie native di schianto di HoverAviary (posizione e assetto fuori limite)."""
    x, y, z = state[0], state[1], state[2]
    roll, pitch = state[7], state[8]
    return bool(abs(x) > 1.5 or abs(y) > 1.5 or z > 2.0
                or abs(roll) > 0.4 or abs(pitch) > 0.4)


class HoverAviaryTerminal(HoverAviary):
    def _computeTerminated(self):
        return is_crash(self._getDroneStateVector(0))

    def _computeTruncated(self):
        return bool(self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC)