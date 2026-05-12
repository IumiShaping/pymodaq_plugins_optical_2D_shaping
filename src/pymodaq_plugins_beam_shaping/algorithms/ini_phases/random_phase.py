import numpy as np

from pymodaq_plugins_beam_shaping.algorithms.ini_phase import PhaseFactory, PhaseBase


@PhaseFactory.register_phase()
class RandomPhase(PhaseBase):

    def compute_phase(self) -> np.ndarray:
        return np.random.random_sample(self.algo.shape) * 2 *np.pi