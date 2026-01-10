# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

see https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Any
from qtpy import QtWidgets, QtCore

import numpy as np

from pymodaq_gui.utils import DockArea
from pymodaq_gui.utils.utils import mkQApp
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

import torch
from torch.nn import MSELoss
from torchmin import minimize


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)


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

        self._phase_tensor: torch.tensor = None

        self._calculated_fitness: float = 0.

        self._loss_function = MSELoss()

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()

    def do_things_after_init(self):
        # Initialize phase distribution as trainable parameter
        self.phase_distribution = self.object_field.phase
        self._amplitude_tensor = torch.tensor(self.object_field.amplitude)
        #self.compute_loss()

    @property
    def phase_distribution(self):
        return self._phase_tensor.detach().numpy()

    @phase_distribution.setter
    def phase_distribution(self, value: np.ndarray):
        self._phase_tensor = torch.tensor(value, requires_grad=True)

    def compute_loss(self, phase_input: torch.tensor) -> torch.tensor:
        self.fft2_field = torch.fft.fftshift(
            torch.fft.fft(
                torch.fft.fftshift(
                    self._amplitude_tensor * torch.exp(1j * phase_input)
                )
            )
        )

        loss = self._loss_function(torch.abs(self.fft2_field), torch.tensor(self._target_field.amplitude))

        self._calculated_fitness = loss.item()
        return loss

    def evolve_field(self):
        pass

    def compute_phase(self):
        result = minimize(self.compute_loss, self.phase_distribution, method='newton-cg')

        print(result)
        # loss = self.compute_loss()
        # loss.backward()
        #
        # self._grad = self._phase_tensor.grad
        #
        # with torch.no_grad():
        #     self._phase_tensor = None
        #
        #     # Manually zero the gradients after updating weights
        #     self._phase_tensor.grad = None


        self.set_phase_in_object_plane(result.x.detach().numpy())
        img_array = self.fft2_field.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))


    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return self._calculated_fitness





