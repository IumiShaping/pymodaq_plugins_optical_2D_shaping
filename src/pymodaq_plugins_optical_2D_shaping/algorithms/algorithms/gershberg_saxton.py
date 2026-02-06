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
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo, AlgoType
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
    ALGOTYPE = AlgoType.AMPLITUDE
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = True

    params = []

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)
        self._algo_init = False
        self.amplitude_mask: Field = None

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()

    def do_things_after_set_target(self):
        if self._algo_init:  #make sure target and object have same shape

            #normalize target_intensity wrt input intensity
            self._target_field = self.normalize_wrt(self._target_field, self._input_field)


    def do_things_after_init(self):
        if not self._algo_init:
            self._algo_init = True
            self.do_things_after_set_target()

            self.define_input_phase()

    def do_things_after_set_input(self):
        """ Apply the initial phase to the object field """

        pass

    def propagate_field(self):
        self.compute_forward_fft(update_plots=False)

    def evolve_field(self):

        if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
            if self.update_mask:
                self.amplitude_mask = self.get_mask_field(ApplyMaskTo.TARGET, inner_value=1, outer_value=0)
                self.update_mask = False

            mask_target = self.amplitude_mask.amplitude
            mask_noise = np.ones_like(mask_target) - mask_target
            amplitude = (self._target_field.amplitude * mask_target +
                         self._image_field.amplitude * mask_noise)
        else:
            amplitude = self._target_field.amplitude

        field_image_corrected = Field(amplitude=amplitude,
                                      phase=self._image_field.phase,
                                      pixel_sizes=self._image_field.pixels_sizes)

        field_object_corrected = self.compute_backward_fft(field_image_corrected)
        self.set_phase_in_object_plane(field_object_corrected.phase)

    def compute_phase(self, do_step=True, **kwargs):
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


    def evolve_field(self):
        if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
            if self.update_mask:
                self.amplitude_mask = self.get_mask_field(ApplyMaskTo.TARGET, inner_value=1, outer_value=0)
                self.update_mask = False

            mask_target = self.amplitude_mask.amplitude
            mask_noise = np.ones_like(mask_target) - mask_target
            amplitude = (self._target_field.amplitude * mask_target  *
                         np.sum((np.exp(self._target_field.amplitude - self.image_field.amplitude)) * mask_target) +
                         self._image_field.amplitude * mask_noise)
        else:
            amplitude = self._target_field.amplitude

        field_image_corrected = Field(amplitude=amplitude,
                                      phase=self._image_field.phase,
                                      pixel_sizes=self._image_field.pixels_sizes)
        field_object_corrected = self.compute_backward_fft(field_image_corrected)

        self.set_phase_in_object_plane(field_object_corrected.phase)





