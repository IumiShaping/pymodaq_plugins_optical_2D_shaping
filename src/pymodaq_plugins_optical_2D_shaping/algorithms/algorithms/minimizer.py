# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

see https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/

@author: Sebastien Weber
"""

from typing import TYPE_CHECKING
from qtpy import QtWidgets

import numpy as np

from pymodaq_plugins_optical_2D_shaping.algorithms.loss import LossFactory
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

import torch
from torchmin import minimize



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)


if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


methods = ['bfgs',
           'l-bfgs',
           'cg',
           'newton-cg',
           'newton-exact',
           'dogleg',
           'trust-ncg',
           'trust-exact',
           'trust-krylov']

loss_factory = LossFactory()


@AlgorithmFactory.register_algorithm()
class Minimize(AlgoBase):
    """ Implementation of minimization iterative algorithms to create amplitude and phase modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    Based on Vol. 25, No. 10 | 15 May 2017 | OPTICS EXPRESS 11695 and implemented here with pytorch
    and the pytorch-minimize package

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'Minimize'
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = True
    MANUAL_LOOP = False

    params = [
        {'title': 'Method', 'name': 'method', 'type': 'list', 'value': 'cg', 'limits': methods},
        {'title': 'Max Iterations', 'name': 'max_iter', 'type': 'int', 'value': 100, 'min': 1},
        {'title': 'Refresh time (s)', 'name': 'refresh_time', 'type': 'float', 'value': 1, 'min': 0.2},
        {'title': 'Tolerance', 'name': 'tolerance', 'type': 'float', 'value': 1e-9},
        {'title': 'Loss', 'name': 'loss', 'type': 'list', 'value': loss_factory.losses[0],
         'limits': loss_factory.losses},
        {'title': 'Loss Parameters', 'name': 'loss_params', 'type': 'group', 'children': []}
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)
        self._slices: tuple[slice, slice] = None
        self.iter = 0
        self._algo_init = False
        self._phase_tensor: torch.Tensor = None
        self.image_field_array: np.ndarray = None

        self._calculated_fitness: float = 0.

        for loss in loss_factory.losses:
            self.settings.child('loss_params').addChild({'title': loss, 'name': loss, 'type': 'group',
                                                         'visible': self.settings['loss'] == loss,
                                                         'children': loss_factory.get_loss(loss).params})

        self._loss = loss_factory.get_loss(self.settings['loss'])(self.settings.child('loss_params',
                                                                                      self.settings['loss']))

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()
        if param.name() == 'loss':
            for param_child in self.settings.child('loss_params').children():
                param_child.show(param.value() == param_child.name())
            self._loss = loss_factory.get_loss(param.value())(self.settings.child('loss_params', param.value()))

    def do_things_after_set_target(self):
        if self._algo_init:  #make sure target and object have same shape
            self._target_field.phase = self._target_field.phase / np.max(self._target_field.phase) * np.pi
            target_tensor = torch.from_numpy(self._target_field.field)

            #normalize target_intensity wrt input amplitude
            self._target_tensor = self.normalize_intensity_wrt(target_tensor, self._amplitude_tensor)

    @staticmethod
    def compute_intensity_ratio(tensor: torch.Tensor, tensor_ref: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((torch.sum(torch.abs(tensor_ref)**2) /
                           torch.sum(torch.abs(tensor)**2)))

    def normalize_intensity_wrt(self, tensor: torch.Tensor, tensor_ref: torch.Tensor) -> torch.Tensor:
        return tensor * self.compute_intensity_ratio(tensor, tensor_ref)

    def do_things_after_init(self):
        # Initialize phase distribution as trainable parameter
        self.phase_distribution = self.define_input_phase()
        self._amplitude_tensor = torch.tensor(self.object_field.amplitude.copy(), requires_grad=True)

        if not self._algo_init:
            self._algo_init = True
            self.do_things_after_set_target()

        self.compute_image_field(self._phase_tensor)
        self.callback(self._phase_tensor.reshape(np.prod(self._phase_tensor.shape)))

        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

    @property
    def phase_distribution(self):
        return self._phase_tensor.detach().numpy()

    @phase_distribution.setter
    def phase_distribution(self, value: np.ndarray):
        self._phase_tensor = torch.tensor(value.copy(), requires_grad=True)

    def compute_image_field(self, phase_input: torch.Tensor) -> torch.Tensor:
        image_tensor = torch.fft.fftshift(
            torch.fft.fft2(self._amplitude_tensor * torch.exp(1j * phase_input), norm='forward'
            )
        )
        self.image_field_array = image_tensor.detach().numpy()
        return image_tensor


    def compute_loss(self, phase) -> torch.Tensor:
        image_tensor = self.ratio * self.compute_image_field(phase)

        loss = self._loss.compute_loss(image_tensor * self.mask,
                                       self._target_tensor * self.mask)

        self._calculated_fitness = loss.item()
        print(self._calculated_fitness)
        return loss


    def callback(self, phase):
        self.iter += 1
        self.update_data(phase)

        if not self._running:
            ### todo could use that call to stop the inner minimize loop
            # PR in pytorch-minimize in that direction submitted
            return True

    def update_data(self, phase):
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(self.image_field_array), np.angle(self.image_field_array)))
        phase = phase.detach().numpy().reshape(self.object_field.shape)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        self.set_phase_in_object_plane(phase)
        print(f'{self.iter}')
        self.parent_app.fields_to_plot.emit(self.get_fields_to_plot())
        QtWidgets.QApplication.processEvents()

    def compute_phase(self, do_step=True, ini_phase=None, **kwargs):
        self.iter = 0

        if ini_phase is not None:
            self.phase_distribution = ini_phase

        if do_step:
            max_iter = 1
        else:
            max_iter = self.settings['max_iter']
        result = minimize(self.compute_loss, self._phase_tensor,
                          method=self.settings['method'],
                          max_iter=max_iter,
                          callback=self.callback,
                          tol=self.settings['tolerance'])

        self.set_phase_in_object_plane(result.x.detach().numpy())
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(self.image_field_array), np.angle(self.image_field_array)))

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return self._calculated_fitness





