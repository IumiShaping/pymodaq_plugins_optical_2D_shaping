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
class Srep15426(GbSax):
    """ Implementation of an amplitude-phase algorithm based on the publication in scientific report
    DOI: 10.1038/srep15426

    There are mixed constraints in phase and in amplitude in the target plane (supposed to be in the Fourier Plane
    of a lens)
    """

    ALGO_NAME = 'srep15426'
    ITERATIVE = True

    params = [
        {'title': 'Wavelength (nm)', 'name': 'wavelength', 'type': 'float',
         'value': plugin_config('wavelength_nm',)},
        {'title': 'Focal length (mm)', 'name': 'focal_length', 'type': 'float',
         'value': plugin_config('algo', 'gbsax', 'focal_length_mm')},
    ]

    def set_target_field(self, field: Field):
        self._target_field = field
        self._image_field = Field.init_from_field(self._target_field)

        self._target_mask = self.generate_random_target_mask()

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
        top_hat = gauss2D(x, np.mean(x), int(np.max(x)/3),
                       y, np.mean(y), int(np.max(y)/3),
                       4)
        mask = np.ones(shape)
        mask[top_hat <= 0.5] = 0
        return mask

    def evolve_field(self):

        amplitude_alpha = (self._target_field.amplitude * self._target_mask +
                           self._image_field.amplitude * (1 - self._target_mask))
        phase_alpha = (self._target_field.phase * self._target_mask +
                       self._image_field.phase * (1 - self._target_mask))

        field_alpha = Field(amplitude=amplitude_alpha,
                            phase=phase_alpha,
                            pixel_sizes=self._image_field.pixels_sizes)

        amplitude_beta = (self._target_field.amplitude * (1 - self._target_mask) +
                           self._image_field.amplitude * self._target_mask)
        phase_beta = (self._target_field.phase * (1 - self._target_mask) +
                       self._image_field.phase * self._target_mask)

        field_beta = Field(amplitude=amplitude_beta,
                            phase=phase_beta,
                            pixel_sizes=self._image_field.pixels_sizes)

        field_alpha_object = field_alpha.ifft2()
        field_beta_object = field_beta.ifft2()

        phase_corrected = np.angle(np.exp(1j * field_alpha_object.phase) +
                                   np.exp(1j * field_beta_object.phase))


        self.set_phase_in_object_plane(phase_corrected)





