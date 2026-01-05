# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from dataclasses import field
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
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithms.gershberg_saxton import GbSax
from pymodaq_utils.math_utils import gauss2D


if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


@AlgorithmFactory.register_algorithm()
class Oe23001052(GbSax):
    """ Implementation of an amplitude-phase algorithm based on the publication in scientific report
    DOI:10.1364/OE.23.001052

    There are mixed constraints in phase and in amplitude in the target plane (supposed to be in the Fourier Plane
    of a lens)
    """

    ALGO_NAME = 'oe23001052 Mixed Constraints'
    ITERATIVE = True

    params = [
    ]

    def set_target_field(self, field: Field):
        self._target_field = field
        self._image_field = Field.init_from_field(self._target_field)

        self._target_mask = self.generate_square_mask()

    def generate_random_target_mask(self):
        return np.random.randint(0, 2, self._target_field.shape)

    def generate_mask_from_target(self):
        mask = np.ones_like(self._target_field.amplitude)
        mask[self._target_field.amplitude <= 0.7] = 0
        return mask

    def generate_square_mask(self):
        shape = self._target_field.shape
        x = np.linspace(0, shape[1], shape[1])
        y = np.linspace(0, shape[0], shape[0])
        top_hat = gauss2D(x, np.mean(x), int(np.max(x)/2.5),
                       y, np.mean(y), int(np.max(y)/2.5),
                       4)
        mask = np.ones(shape)
        mask[top_hat <= 0.5] = 0
        return mask

    def evolve_field(self):

        amplitude = (self._target_field.amplitude * self._target_mask +
                           self._image_field.amplitude * (1 - self._target_mask))
        phase = (self._target_field.phase * (1- self._target_mask) +
                 self._image_field.phase * self._target_mask)

        field = Field(amplitude=amplitude,
                      phase=phase,
                      pixel_sizes=self._image_field.pixels_sizes)

        field_alpha_object = field.ifft2()


        self.set_phase_in_object_plane(field_alpha_object.phase)




