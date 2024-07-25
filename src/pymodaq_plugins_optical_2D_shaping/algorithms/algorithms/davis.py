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
class Davis(AlgoBase):

    ALGO_NAME = 'Davis1999'
    ITERATIVE = False
    params = [

    ]

    def __init__(self):
        super().__init__()

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        if phase.shape == self._object_field.shape:
            self._object_field.phase = phase.copy()
            self._object_field.amplitude = (
                    self._input_field.amplitude.copy() *
                    (induced_amplitude if induced_amplitude is not None else 1))
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return 100 * np.sum(
            np.abs(np.sqrt(self._target_field.intensity) - self._image_field.intensity)) ** 2 \
            / np.prod(self._image_field.shape) / np.sum(self._target_field.intensity)

    def wrap_phase(self, phase: np.ndarray):
        """ Wrap phase between -pi and +pi"""
        return np.angle(np.exp(1j * phase))

    def compute_phase(self):
        _, wedge_phase = np.mgrid[0: self._target_field.shape[0], 0: self._target_field.shape[1]]
        wedge_phase = wedge_phase.astype(float)
        wedge_phase *= 100 * np.pi / self._target_field.shape[1]

        calculated_field = self._target_field.ifft2()
        self.set_phase_in_object_plane(mutils.normalize(calculated_field.amplitude) *
                                       self.wrap_phase(calculated_field.phase + wedge_phase),
                                       )
        self._image_field = self.object_field.fft2()

    def compute_Tn(self, field, order_n=1):
        Tn = (np.sinc(np.pi * (order_n - field.amplitude)))
        return Tn



