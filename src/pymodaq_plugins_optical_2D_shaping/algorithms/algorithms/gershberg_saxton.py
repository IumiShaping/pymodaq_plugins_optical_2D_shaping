# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Any

import numpy as np

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq_utils import math_utils as mutils
from pymodaq_data import Q_

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig


if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


@AlgorithmFactory.register_algorithm()
class GbSax(AlgoBase):
    """ Implementation of the Gerchberg-Saxton iterative algorithm to create amplitude modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'Gerchberg-Saxton'
    ITERATIVE = True

    params = [
        {'title': 'Wavelength (nm)', 'name': 'wavelength', 'type': 'float',
         'value': plugin_config('wavelength_nm',)},
        {'title': 'Focal length (mm)', 'name': 'focal_length', 'type': 'float',
         'value': plugin_config('algo', 'gbsax', 'focal_length_mm')},
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        if phase.shape == self._object_field.shape:
            self._object_field.phase = phase.copy()
            self._object_field.amplitude = (
                    self._input_field.amplitude.copy() *
                    (induced_amplitude if induced_amplitude is not None else 1))
        else:
            raise ValueError('The phase shape is incoherent with the parameters')


    def get_target_pixels_size(self, slm_size: Tuple[Q_, Q_] = None) -> list[Q_]:
        """ Get the expected physical size of the pixels in the target plane given
        the chosen algorithm and physical parameters: focal length, wavelength..."""
        if slm_size is None:
            slm_size = [self._input_field.shape[ind] * self._input_field.pixels_sizes[ind]
                         for ind in range(2)]

        return [Q_(self.settings['wavelength'], 'nm') *
                Q_(self.settings['focal_length'], 'mm') /
                size for size in slm_size]

    def propagate_field(self):
        self._image_field = self._object_field.fft2()
        self._image_field = self.scale_target_with_geometry(self._image_field)

    def evolve_field(self):
        field_image_corrected = Field(amplitude=self._target_field.amplitude,
                                      phase=self._image_field.phase,
                                      pixel_sizes=self._image_field.pixels_sizes)
        field_object_corrected = field_image_corrected.ifft2()

        self.set_phase_in_object_plane(field_object_corrected.phase)

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return 100 * np.sum(
            np.abs(np.sqrt(self._target_field.intensity) - self._image_field.intensity)) ** 2 \
            / np.prod(self._image_field.shape) / np.sum(self._target_field.intensity)

    def compute_phase(self):
        self.propagate_field()
        self.evolve_field()






