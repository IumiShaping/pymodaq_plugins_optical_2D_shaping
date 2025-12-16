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

from pymodaq_utils.logger import set_logger, get_module_name

from pymodaq_utils import math_utils as mutils
from pymodaq_gui import utils as gutils
from pymodaq_data import Q_
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
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = False

    params = [
        {'title': 'Mask block size (pxls)', 'name': 'block_size', 'type': 'int', 'value': 5, 'min': 1},
        {'title': 'Circular Aperture', 'name': 'circ_aperture', 'type': 'str',
         'value': ''},
        {'title': 'Intermediate plane:', 'name': 'show_inter_plane', 'type': 'bool_push',
         'label': 'Show Intermediate Plane', 'value': False, },

    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

        self._is_roi_init = False

        self.intermediate_widget = QtWidgets.QWidget()
        self.intermediate_widget.closeEvent = \
            lambda event: self.settings.child('show_inter_plane').setValue(False)
        self.intermediate_viewer = Viewer2D(self.intermediate_widget)
        self.intermediate_viewer.roi_manager.add_roi_programmatically('CircularROI')

    def init_roi(self):
        roi = self.intermediate_viewer.roi_manager.get_roi_from_index(0)
        roi.set_center(np.array(self._input_field.shape)[::-1] / 2)
        size = self.intermediate_viewer.view.unscale_axis(
            Q_(self.settings['circ_aperture'], 'um').m_as('m'),
            Q_(self.settings['circ_aperture'], 'um').m_as('m'))
        roi.setSize(size)
        self.intermediate_viewer.roi_manager.roi_changed.connect(self.update_circular_aperture)

    def get_target_pixels_size(self, slm_size: Tuple[Q_, Q_] = None) -> list[Q_]:
        """ Get the expected physical size of the pixels in the target plane given
        the chosen algorithm and physical parameters: focal length, wavelength...

        Here we use a 4f setup, so the image field has the same size as the SLM with a ratio given by the focal
        length ratio

        """
        if slm_size is None:
            slm_size = [self._input_field.shape[ind] * self._input_field.pixels_sizes[ind]
                        for ind in range(2)]

        return [Q_(plugin_config('setup', 'wavelength_nm', ), 'nm') *
                Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0], 'mm') /
                size for size in slm_size]

    def quit(self):
        """ to reimplement if neccessary"""
        self.intermediate_widget.close()

    def update_circular_aperture(self):
        diameter = Q_(max(self.intermediate_viewer.view.scale_axis(
            *self.intermediate_viewer.roi_manager.get_roi_from_index(0).size())),
            self.intermediate_viewer.view.get_axis('top').axis_units)
        self.settings.child('circ_aperture').setValue(diameter.m_as('um'))

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        if phase.shape == self._object_field.shape:
            self._object_field.phase = phase.copy()
            self._object_field.amplitude = (
                    self._input_field.amplitude.copy() *
                    (induced_amplitude if induced_amplitude is not None else 1))
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    def scale_target_with_geometry(self, field: Field):
        """ Apply an axis scaling to have the target and its axes in correct units with respect to
        a given algorithm implementation and experimental setup

        to be reimplemented if needed
        """
        return field

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

        beta = np.arccos(mutils.normalize(calculated_field.amplitude))
        theta_field = Field('theta', phase=calculated_field.phase + beta)
        alpha_field = Field('alpha', phase=calculated_field.phase - beta)

        theta_field.phase *= odd_mask
        alpha_field.phase *= even_mask

        slm_pixel_sizes = self._input_field.pixels_sizes

        intermediate_pixel_sizes = ((Q_(plugin_config('wavelength_nm',), 'nm') *
                                     Q_(self.settings['focal_length_1'], 'mm')) /
                                    slm_pixel_sizes /
                                    np.array(theta_field.shape)
                                    ).to('um')

        self.set_phase_in_object_plane(theta_field.phase + alpha_field.phase)

        intermediate_field = self.object_field.fft2()

        intermediate_field.calibrate_axes(intermediate_pixel_sizes)
        intermediate_field.axes = intermediate_field.get_axes()

        circ_aperture = self.create_aperture(intermediate_pixel_sizes)

        self.intermediate_viewer.show_data(intermediate_field.intensity_as_dwa().to_dB() *
                                           circ_aperture)

        if not self._is_roi_init:
            self.init_roi()
            self._is_roi_init = True

        self._image_field = (intermediate_field * circ_aperture).ifft2()

        target_pixel_sizes = ((Q_(plugin_config('wavelength_nm',), 'nm') *
                               Q_(self.settings['focal_length_2'], 'mm')) /
                              intermediate_pixel_sizes /
                              np.array(theta_field.shape)
                              ).to('um')

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

        if param.name() != 'show_inter_plane':
            self.parent_app.compute_phase()
        else:
            self.intermediate_widget.setVisible(param.value())





