import numpy as np

from pymodaq.utils.data import DataActuator

from pymodaq_plugins_optical_2D_shaping.utilities.sizing import get_slm_size


class ShaperCamera:
    def __init__(self):
        self._slm_values: np.ndarray = np.zeros(get_slm_size())

    def get_slm(self):
        return self._slm_values

    def apply_grey_scale(self, value: np.ndarray):
        self._slm_values = value

    def get_camera(self) -> np.ndarray:
        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self._slm_values,
                                                                  axes=1),
                                                  axes=(1,)),
                                      axes=1)) ** 2