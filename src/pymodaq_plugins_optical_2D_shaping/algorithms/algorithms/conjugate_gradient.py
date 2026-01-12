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
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo, TargetPhase
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

    params = [
        {'title': 'Max Iterations', 'name': 'max_iter', 'type': 'int', 'value': 20, 'min': 1},
        {'title': 'Loss exponent', 'name': 'exponent', 'type': 'int', 'value': 4, 'min': 2},
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)
        self._slices: tuple[slice, slice] = None
        self.iter = 0
        self._algo_init = False
        self._phase_tensor: torch.Tensor = None
        self.image_tensor: torch.Tensor = None
        self._module: Union[torch, np] = torch
        self.module = torch

        self._calculated_fitness: float = 0.

        self._loss_function = MSELoss()

    @property
    def module(self):
        return self.module

    @module.setter
    def module(self, mod: Union[torch, np]) -> None:
        self._module = mod
        self.abs = getattr(mod, 'abs')
        self.sum = getattr(mod, 'sum')
        self.exp = getattr(mod, 'exp')
        self.prod = getattr(mod, 'prod')

        if mod is torch:
            self.fft2 = lambda x: torch.fft.fft2(x, norm='forward')
            self.fftshift = torch.fft.fftshift
        else:
            self.fft2 = lambda x: np.fft.fft2(x, norm='forward')
            self.fftshift = np.fft.fftshift


    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()

    def do_things_after_set_target(self):
        if self._algo_init:  #make sure target and object have same shape
            self._target_tensor = torch.tensor(self._target_field.field)

            #normalize target_intensity wrt input amplitude
            self._target_tensor = self._target_tensor / self.abs(self._target_tensor).max()
            self._target_tensor *= (self.sum(self._amplitude_tensor ** 2) /
                                    self.sum(self.abs(self._target_tensor)**2))


    def do_things_after_init(self):
        # Initialize phase distribution as trainable parameter
        self.phase_distribution = self.define_input_phase()
        self._amplitude_tensor = torch.tensor(self.object_field.amplitude)

        if not self._algo_init:
            self._algo_init = True
            self.do_things_after_set_target()

        self.compute_image_field(self._phase_tensor)
        self.update_plots(self._phase_tensor.reshape(np.prod(self._phase_tensor.shape)))

        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()
    @property
    def phase_distribution(self):
        return self._phase_tensor.detach().numpy()

    @phase_distribution.setter
    def phase_distribution(self, value: np.ndarray):
        self._phase_tensor = torch.tensor(value, requires_grad=True)

    def compute_image_field(self, phase_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        image_tensor = self.fftshift(
            self.fft2(
                self.fftshift(
                    self._amplitude_tensor * self.exp(1j * phase_input)
                )
            )
        )
        self.image_tensor = image_tensor

    def loss_function(self,
                      field_tested: Union[torch.Tensor, np.ndarray],
                      field_target: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
            slices = self.get_mask_slices(ApplyMaskTo.TARGET)
        else:
            slices = (Ellipsis, Ellipsis)

        field_tested = field_tested[*slices]

        return ((self.sum(
                    (self.abs(field_tested) ** 2 -
                     self.abs(field_target[*slices]) ** 2)
                    ** self.settings['exponent'])))

    def compute_loss(self, phase) -> torch.Tensor:
        self.compute_image_field(phase)
        loss = self.loss_function(self.image_tensor, self._target_tensor)
        self._calculated_fitness = loss.item()
        return loss

    def evolve_field(self):
        pass

    def update_plots(self, phase):
        self.iter += 1

        image_array = self.image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(image_array), np.angle(image_array)))
        phase = phase.detach().numpy().reshape(self.object_field.shape)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        self.set_phase_in_object_plane(phase)
        print(f'{self.iter}')
        self.parent_app.fields_to_plot.emit(self.get_fields_to_plot())


    def compute_phase(self):
        self.iter = 0
        self.phase_distribution = self.define_input_phase()

        result = minimize(self.compute_loss, self._phase_tensor, method='cg', max_iter=self.settings['max_iter'],
                          callback=self.update_plots)

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
        img_array = self.image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))


    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return self._calculated_fitness





