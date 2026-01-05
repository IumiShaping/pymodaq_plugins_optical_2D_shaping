# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
from copy import deepcopy
from qtpy import QtWidgets
from zernpy.calculations.calc_psfs_check import pixel_size

from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq_utils import math_utils as mutils
from pymodaq_gui import utils as gutils
from pymodaq_data import Q_, DataToExport
from pymodaq_gui.plotting.data_viewers import ViewerDispatcher, Viewer2D

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


@AlgorithmFactory.register_algorithm()
class OE2016(AlgoBase):
    """ Implementation of the algorithms discussed in OPTICS EXPRESS DOI:10.1364/OE.24.006249
    The algorithms create amplitude image with phase only spatial light modulators
    in a 4f line with two lens and a circular filter in the Fourier Plane to filter out undiffracted light

    The corresponding experimental setup should define a working light wavelength and focal lengths
    of the used lens
    """

    ALGO_NAME = 'OE2016 - Amplitude diffractive'
    SETUP_TYPE = LensSetup.FourF
    ITERATIVE = False

    params = [
        {'title': 'Gamma', 'name': 'gamma', 'type': 'int', 'value': 4,
         'tip': 'Grating period of the modulation in pixels'},
        {'title': 'Circular Aperture', 'name': 'circ_aperture', 'type': 'group', 'children': [
            {'title': 'Position X', 'name': 'posx', 'type': 'float', 'value': -1380, 'suffix': 'um'},
            {'title': 'Position Y', 'name': 'posy', 'type': 'float', 'value': 0, 'suffix': 'um'},
            {'title': 'Diameter', 'name': 'diameter', 'type': 'float', 'value': 200, 'suffix': 'um'},
            ]},
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

        self.intermediate_field: Field = None

    def get_fields_to_plot(self) -> DataToExport:
        dte = super().get_fields_to_plot()
        if self.intermediate_field is not None:
            dte.append(self.intermediate_field.intensity_as_dwa(name='Intensity', origin_name='Intermediate'))
        return dte

    def generate_grating_phase(self) -> np.ndarray:
        x_array = np.linspace(0, self._input_field.shape[1], self._input_field.shape[1])
        return (2 * np.pi * x_array / self.settings['gamma']) % 2*np.pi

    def compute_phase(self):

        relative_amplitude = ((self._target_field.amplitude / self._target_field.amplitude.max()) /
                              (self._input_field.amplitude / self._input_field.amplitude.max()))
        relative_amplitude = relative_amplitude / relative_amplitude.max()


        self.set_phase_in_object_plane(relative_amplitude * self.generate_grating_phase())



        circ_aperture = self.create_aperture(self.intermediate_pixel_sizes)
        self.intermediate_field = self.object_field.fft2() * circ_aperture
        self.intermediate_field.calibrate_axes(self.intermediate_pixel_sizes)
        self.intermediate_field.axes = self.intermediate_field.get_axes()

        self._image_field = self.intermediate_field.ifft2()

        self.image_field.calibrate_axes(self._target_field.pixels_sizes)
        self.image_field.axes = self.image_field.get_axes()

    def create_aperture(self, intermediate_pixel_size: Q_) -> np.ndarray:
        x = np.arange(0, self._target_field.shape[1], 1) * intermediate_pixel_size[1]
        y = np.arange(0, self._target_field.shape[0], 1) * intermediate_pixel_size[0]

        xx, yy = np.meshgrid(x, y)
        mask_field = np.zeros(self._target_field.shape)
        mask_field[
            np.sqrt(
                (xx - np.mean(x) - Q_(self.settings['circ_aperture', 'posx'], 'um')) ** 2 +
                (yy -np.mean(y) - Q_(self.settings['circ_aperture', 'posy'], 'um')) ** 2)
            <=
            Q_(self.settings['circ_aperture', 'diameter'], 'um') / 2] = 1
        return mask_field

    def value_changed(self, param):
        self.parent_app.compute_phase()





