# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
from copy import deepcopy

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq.utils import math_utils as mutils
from pymodaq.utils import gui_utils as gutils
from pymodaq import Q_
from pymodaq.utils.plotting.data_viewers import ViewerDispatcher

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field

logger = set_logger(get_module_name(__file__))


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
        {'title': 'Wavelength (nm)', 'name': 'wavelength', 'type': 'float', 'value': 515.,},
        {'title': 'Focal length 1 (mm)', 'name': 'focal_length_1', 'type': 'float', 'value': 300.,},
        {'title': 'Focal length 2 (mm)', 'name': 'focal_length_2', 'type': 'float',
         'value': 300., },
        {'title': 'Circular Aperture (um)', 'name': 'circ_aperture', 'type': 'float',
         'value': 300., },

    ]

    def __init__(self):
        super().__init__()
        self.intermediate_dockarea = gutils.DockArea()
        self.intermediate_viewer_dispatcher = ViewerDispatcher(self.intermediate_dockarea)
        self.intermediate_dockarea.show()

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
        odd_mask = self.create_mask_odd()
        even_mask = self.create_mask_odd(False)

        calculated_field: Field = deepcopy(self._target_field)

        beta = np.arccos(mutils.normalize(calculated_field.amplitude))
        theta_field = Field('theta', phase=calculated_field.phase + beta)
        alpha_field = Field('alpha', phase=calculated_field.phase - beta)

        theta_field.phase *= odd_mask
        alpha_field.phase *= even_mask

        slm_pixel_sizes = self._input_field.pixels_sizes

        intermediate_pixel_sizes = ((Q_(self.settings['wavelength'], 'nm') *
                                     Q_(self.settings['focal_length_1'], 'mm')) /
                                    slm_pixel_sizes /
                                    np.array(theta_field.shape)
                                    ).to('um')

        self.set_phase_in_object_plane(theta_field.phase + alpha_field.phase)

        intermediate_field = self.object_field.fft2()

        intermediate_field.calibrate_axes(intermediate_pixel_sizes)
        intermediate_field.axes = intermediate_field.get_axes()

        circ_aperture = self.create_aperture(intermediate_pixel_sizes)

        self.intermediate_viewer_dispatcher.show_data(
            DataToExport('intermediate', data=[
                intermediate_field.amplitude_as_dwa(),
                intermediate_field.phase_as_dwa(),
                intermediate_field.amplitude_as_dwa(name='aperture') * circ_aperture
            ])
        )

        self._image_field = (intermediate_field * circ_aperture
                             ).ifft2()

        target_pixel_sizes = ((Q_(self.settings['wavelength'], 'nm') *
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
        mask_field[np.sqrt(xx**2 + yy**2) <= Q_(self.settings['circ_aperture'], 'um')] = 1
        return mask_field

    def create_mask_odd(self, odd=True) -> np.ndarray:
        ix, iy = np.mgrid[0:self._target_field.shape[0], 0:self._target_field.shape[1]]

        if odd:
            mask_field = np.zeros(self._target_field.shape)
            mask_field[mutils.odd_even(ix + iy)] = 1
        else:
            mask_field = np.ones(self._target_field.shape)
            mask_field[mutils.odd_even(ix + iy)] = 0
        return mask_field






