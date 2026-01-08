# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Any
from qtpy import QtWidgets, QtCore

import numpy as np

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

import torch
import torch.optim as optim
import ncg_optimizer as optim_ncg
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


@AlgorithmFactory.register_algorithm()
class ConjugateGradient(AlgoBase):
    """ Implementation of the ConjugateGradient iterative algorithm to create amplitude and phase modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    Based on Vol. 25, No. 10 | 15 May 2017 | OPTICS EXPRESS 11695 and implemented here with pytorch

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'ConjugateGradient'
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = True

    params = []

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)
        self._slices: tuple[slice, slice] = None

        self.phase_distribution: torch.nn.Parameter = None
        self._calculated_fitness = 0.
        self._loss: torch.Tensor = None
        self.target_torch: torch.Tensor = None
        self.image_torch: torch.Tensor = None

        self._loss_function:  nn.MSELoss = None

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()

    def do_things_after_init(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)

        # Initialize phase distribution as trainable parameter
        self.phase_distribution = torch.nn.Parameter(torch.asarray(self.object_field.phase))
        self.target_torch = torch.asarray(self._target_field.field)
        self._loss_function = nn.MSELoss()
        self.optimizer = optim_ncg.BASIC([self.phase_distribution], method='LS',
                                         line_search='Armijo', c1=1e-4, c2=0.9, lr=1,
                                         rho=0.5)  # OK
        QtWidgets.QApplication.restoreOverrideCursor()

    @property
    def object_torch(self) -> torch.Tensor:
        return torch.Tensor(torch.asarray(self.object_field.amplitude)) * torch.exp(1j * self.phase_distribution)

    def propagate_field(self):
        self.image_torch = torch.fft.fft2(self.object_torch)
        img_array = self.image_torch.numpy(force=True)
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))


    def closure(self):
        self.optimizer.zero_grad()

        if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
            if self.update_mask:

                self._slices = self.get_mask_slices(ApplyMaskTo.TARGET)
                self.update_mask = False
        else:
            self._slices = (Ellipsis, Ellipsis)

        d=2
        # max_amplitude = torch.max(torch.abs(self.image_torch[*self._slices])**2 *
        #                           torch.abs(self.target_torch[*self._slices])**2)

        # loss = 10 ** d * (
        #         1 - torch.sum(
        #     torch.sqrt(
        #         1/ max_amplitude *
        #         torch.abs(self.image_torch[*self._slices])**2 *
        #         torch.abs(self.target_torch[*self._slices])**2) *
        #     torch.cos(torch.angle(self.image_torch[*self._slices]) -
        #               torch.angle(self.target_torch[*self._slices]))))**2

        loss = self._loss_function(torch.abs(self.image_torch[*self._slices]),
                                   torch.abs(self.target_torch[*self._slices]))
        loss.backward(retain_graph=True)
        self._calculated_fitness = loss.numpy(force=True)

        return loss

    def evolve_field(self):

        self.optimizer.step(self.closure)

        self.set_phase_in_object_plane(self.phase_distribution.data.numpy(force=True))

    def compute_phase(self):
        self.propagate_field()
        self.evolve_field()

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return float(self._calculated_fitness)





