# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq.utils import math_utils as mutils

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field

logger = set_logger(get_module_name(__file__))


@AlgorithmFactory.register_algorithm()
class GbSax(AlgoBase):

    ALGO_NAME = 'Gerchberg–Saxton'

    params = [

    ]

    def __init__(self):
        super().__init__()

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        self._object_field.phase = phase
        if phase.shape == self._object_field.shape:
            self._object_field.phase = phase
            self._object_field.amplitude = (
                    self._input_field.amplitude *
                    (induced_amplitude if induced_amplitude is not None else 1))
            self.propagate_field()
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    def propagate_field(self):
        field_object_padded = self._object_field.pad(self.get_npad_between_image_object(),
                                                     constant_values=(0, 0))
        self._image_field = field_object_padded.fft2()

    def evolve_field(self):
        npad = self.get_npad_between_image_object()

        field_image_corrected = Field(amplitude=self._target_field.amplitude,
                                      phase=self._image_field.phase,
                                      pixel_sizes=self._image_field.pixels_sizes)
        field_object_corrected = field_image_corrected.ifft2().unpad(
            npad, ini_shape=self.object_field.shape)
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

    @property
    def intensity_image(self):
        return self._image_field.intensity





