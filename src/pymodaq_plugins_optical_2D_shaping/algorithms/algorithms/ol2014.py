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
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field
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
    ITERATIVE = False

    params = [
        {'title': 'Focal length 1 (mm)', 'name': 'focal_length_1', 'type': 'float', 'value': 300.,},
        {'title': 'Focal length 2 (mm)', 'name': 'focal_length_2', 'type': 'float',
         'value': 300., },
        {'title': 'Mask period', 'name': 'period', 'type': 'int', 'value': 1, 'min': 1},
        {'title': 'Circular Aperture (um)', 'name': 'circ_aperture', 'type': 'float',
         'value': 3000., },
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
        odd_mask = self.create_mask_odd(period=self.settings['period'])
        even_mask = self.create_mask_odd(False, period=self.settings['period'])

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

    def create_cell(self, odd=True, period=1):
        zeros = np.zeros((period, period))
        ones = np.ones((period, period))
        if odd:
            cell = np.concatenate((ones, zeros))
        else:
            cell = np.concatenate((zeros, ones))
        cell = np.hstack((cell, cell[::-1, :])).astype(int)
        return cell

    def create_mask_odd(self, odd=True, period=1) -> np.ndarray:

        cell = self.create_cell(odd, period)
        mask = np.tile(cell, ((self._target_field.shape[0] // period) + 1,
                              (self._target_field.shape[1] // period) + 1,))
        mask = mask[0:self._target_field.shape[0], 0:self._target_field.shape[1]]

        return mask

    def value_changed(self, param):

        if param.name() != 'show_inter_plane':
            self.parent_app.compute_phase()
        else:
            self.intermediate_widget.setVisible(param.value())





