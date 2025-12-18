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
class OL2014(AlgoBase):
    """ Implementation of the OPTICS LETTERS / Vol. 39, No. 7 / April 1, 2014
    algorithm to create amplitude and phase modulated image with phase only spatial light modulators
    in a 4f line with two lens and a circular filter in the Fourier Plane

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'OL2014'
    SETUP_TYPE = LensSetup.FourF
    ITERATIVE = False

    params = [
        {'title': 'Mask block size (pxls)', 'name': 'block_size', 'type': 'int', 'value': 5, 'min': 1},
        {'title': 'Circular Aperture (um)', 'name': 'circ_aperture', 'type': 'float',
         'value': 500},
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

        self.intermediate_field: Field = None

    def get_fields_to_plot(self) -> DataToExport:
        dte = super().get_fields_to_plot()
        if self.intermediate_field is not None:
            dte.append(self.intermediate_field.intensity_as_dwa(name='Intensity', origin_name='Intermediate'))
        return dte

    def get_target_pixels_size(self, slm_size: Tuple[Q_, Q_] = None) -> list[Q_]:
        """ Get the expected physical size of the pixels in the target plane given
        the chosen algorithm and physical parameters: focal length, wavelength...

        Here we use a 4f setup, so the image field has the same size as the SLM with a ratio given by the focal
        length ratio

        """
        pixels_size = self._input_field.pixels_sizes
        focal_ratio = (plugin_config('setup', self.SETUP_TYPE.value, 'focals')[1] /
                       plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0])
        return [size * focal_ratio for size in pixels_size]

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

    def compute_phase(self):
        odd_mask = self.create_checker_board()
        even_mask = 1 - odd_mask

        calculated_field: Field = deepcopy(self._target_field)

        beta = np.arccos(mutils.normalize(calculated_field.amplitude) / 2)
        theta_field = Field('theta', phase=calculated_field.phase + beta)
        alpha_field = Field('alpha', phase=calculated_field.phase - beta)

        theta_field.phase *= odd_mask
        alpha_field.phase *= even_mask

        slm_pixel_sizes = self._input_field.pixels_sizes

        intermediate_pixel_sizes = [((Q_(plugin_config('setup', 'wavelength_nm',), 'nm') *
                                     Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0], 'mm')) /
                                    (slm_pixel_sizes[ind] * theta_field.shape[ind])
                                    ).to('um') for ind in range(2)]

        self.set_phase_in_object_plane(theta_field.phase + alpha_field.phase)



        circ_aperture = self.create_aperture(intermediate_pixel_sizes)
        self.intermediate_field = self.object_field.fft2() * circ_aperture
        self.intermediate_field.calibrate_axes(intermediate_pixel_sizes)
        self.intermediate_field.axes = self.intermediate_field.get_axes()

        self._image_field = self.intermediate_field.ifft2()

        target_pixel_sizes = [((Q_(plugin_config('setup', 'wavelength_nm',), 'nm') *
                               Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[1], 'mm')) /
                               (intermediate_pixel_sizes[ind] * theta_field.shape[ind])
                              ).to('um') for ind in range(2)]
        self.image_field.calibrate_axes(target_pixel_sizes)
        self.image_field.axes = self.image_field.get_axes()

    def create_aperture(self, intermediate_pixel_size: Q_) -> np.ndarray:
        x = np.arange(0, self._target_field.shape[1], 1) * intermediate_pixel_size[1]
        y = np.arange(0, self._target_field.shape[0], 1) * intermediate_pixel_size[0]

        xx, yy = np.meshgrid(x, y)
        mask_field = np.zeros(self._target_field.shape)
        mask_field[
            np.sqrt((xx - np.mean(x)) ** 2 + (yy - np.mean(y)) ** 2)
            <=
            Q_(self.settings['circ_aperture'], 'um') / 2] = 1
        return mask_field

    def create_checker_board(self) -> np.ndarray:

        block_size = self.settings['block_size']
        y, x = np.indices(self._image_field.shape)
        mask = ((x // block_size) + (y // block_size)) % 2
        return mask

    def value_changed(self, param):
        self.parent_app.compute_phase()





