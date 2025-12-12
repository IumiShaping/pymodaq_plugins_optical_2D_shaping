# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Any

import numpy as np
from pymodaq_utils.enums import StrEnum
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq_utils import math_utils as mutils
from pymodaq_data import Q_
from pymodaq_gui.parameter import Parameter

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

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()

    def do_things_after_set_input(self):
        """ Apply the initial phase to the object field """

        pass

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

        amplitude = self._target_field.amplitude

        field_image_corrected = Field(amplitude=amplitude,
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



@AlgorithmFactory.register_algorithm()
class GbSaxAdaptiveWeighted(GbSax):
    """ Implementation of the Weighted Gerchberg-Saxton iterative algorithm to create amplitude modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens

    The algorithm is based on paper https://doi.org/10.1364/OE.413723

    """

    ALGO_NAME = 'Weighted Gerchberg-Saxton'
    ITERATIVE = True


    params = GbSax.params

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

        self.amplitude_mask: Field = None

    def evolve_field(self):
        if self.mask is not None and (
                self.amplitude_mask is None or self._target_field.shape != self.amplitude_mask.shape):
            self.amplitude_mask = self.mask_from_slices()

        if self.mask is not None and self.amplitude_mask is not None:
            mask_target = self.amplitude_mask.amplitude
            mask_noise = np.ones_like(mask_target) - mask_target
            amplitude = (self._target_field.amplitude * mask_target  *
                         np.exp(self._target_field.amplitude - self.image_field.amplitude) +
                         self._image_field.amplitude * mask_noise)
        else:
            amplitude = self._target_field.amplitude

        field_image_corrected = Field(amplitude=amplitude,
                                      phase=self._image_field.phase,
                                      pixel_sizes=self._image_field.pixels_sizes)
        field_object_corrected = field_image_corrected.ifft2()

        self.set_phase_in_object_plane(field_object_corrected.phase)

    def mask_from_slices(self) -> Field:
        mask = Field.init_from_field(self._target_field).amplitude * 0
        mask[*self.mask] = 1
        return Field(amplitude=mask)


    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        if self.amplitude_mask is not None:
            return 100 * np.sum(
                np.abs(np.sqrt(self._target_field.intensity)
                       - self._image_field.intensity) * self.amplitude_mask.amplitude) ** 2 \
                / np.prod(self._image_field.shape) / np.sum(self._target_field.intensity * self.amplitude_mask.amplitude)
        else:
            return super().fitness





