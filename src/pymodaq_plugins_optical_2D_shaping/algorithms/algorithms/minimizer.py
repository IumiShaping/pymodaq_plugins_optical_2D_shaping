# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

see https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/

@author: Sebastien Weber
"""
from abc import ABC, abstractmethod

from pathlib import Path
from typing import Union, Tuple, List, TYPE_CHECKING, Any, Callable
from qtpy import QtWidgets, QtCore

import numpy as np

from pymodaq_gui.utils import DockArea
from pymodaq_gui.qt_utils import mkQApp
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo, TargetPhase
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

import torch
from torch.nn import MSELoss as TorchMSELoss, SmoothL1Loss as TorchSmoothL1Loss
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


class LossBase(ABC):

    params: list[dict[str, str]] = []  # definition of the specific parameters needed to compute the loss

    def __init__(self, settings: Parameter):
        self.settings = settings  # attribute used to access specific parameters changed by the user

    @abstractmethod
    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm

        To be reimplemented
        """
        ...


class LossFactory:
    """The factory class for creating Algorithm"""

    _registry = {}

    @classmethod
    def register_loss(cls) -> Callable:
        """Class decorator method to register Loss class to the internal registry. Must be used as
        decorator above the definition of a LossBase inherited class.

        The Loss class must implement specific class attributes and methods
        """

        def inner_wrapper(wrapped_class: type[LossBase]) -> type[LossBase]:
            name = wrapped_class.__name__
            
            if name not in cls._registry:
                cls._registry[name] = wrapped_class
            # Return wrapped_class
            return wrapped_class

        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_loss(cls, name: str) -> type[LossBase]:
        """Factory command to get registered loss class
        .
        This method gets the appropriate executor class from the registry

        Parameters
        ----------
        name: str
            The name of the class as specified during registration

        Returns
        -------
        an instance of the executor created
        """

        if name not in cls._registry:
            raise ValueError(f".{name} is not a supported Loss.")

        return cls._registry[name]

    @property
    def losses(self):
        return list(self._registry.keys())


@LossFactory.register_loss()
class LeastSquares(LossBase):
    params = []

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return ((torch.sum(
                    (torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2)
                    ** 2)))

@LossFactory.register_loss()
class MSELoss(LossBase):
    params = []

    def __init__(self, settings: Parameter):
        super().__init__(settings)

        self._loss = TorchMSELoss()

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return self._loss(torch.abs(field_tested), torch.abs(field_target))


@LossFactory.register_loss()
class SmoothL1Loss(LossBase):
    params = [
        {'title': 'Beta', 'name': 'beta', 'type': 'float', 'value': 0.5},
    ]


    def __init__(self, settings: Parameter):
        super().__init__(settings)

        self._loss = TorchSmoothL1Loss(beta=self.settings['beta'])

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return self._loss(torch.abs(field_tested), torch.abs(field_target))


@LossFactory.register_loss()
class LeastExponent(LossBase):
    params = [
        {'title': 'Loss exponent', 'name': 'exponent', 'type': 'int', 'value': 4, 'min': 2},
    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor,) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """
        return ((torch.sum(
                    (torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2)
                    ** self.settings['exponent'])))


@LossFactory.register_loss()
class Chicken(LossBase):
    params = [
        {'title': 'Power exponent', 'name': 'power_exponent', 'type': 'int', 'value': 9, 'min': 2},
        {'title': 'Sum exponent', 'name': 'sum_exponent', 'type': 'int', 'value': 4, 'min': 2},

    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        amplitude_normalized = torch.sum(torch.abs(field_tested) * torch.abs(field_target))

        return 10 ** self.settings['power_exponent'] * (
                1 - torch.sum((torch.abs(field_tested) * torch.abs(field_target) / amplitude_normalized) *
                              torch.cos(torch.angle(field_target) - torch.angle(field_tested))))**self.settings['sum_exponent']


@LossFactory.register_loss()
class LSQAmplitudePhase(LossBase):
    params = [
        {'title': 'Amplitude exponent', 'name': 'amplitude_exponent', 'type': 'int', 'value': 1, 'min': 1},
        {'title': 'Phase exponent', 'name': 'phase_exponent', 'type': 'int', 'value': 1, 'min': 1},

    ]

    def compute_loss(self,
                     field_tested: torch.Tensor,
                     field_target: torch.Tensor, ) -> torch.Tensor:
        """ Compute the loss by returning a 0D Tensor that will be minimized using minimization algorithm
        """

        amplitude_normalized = torch.sum(torch.abs(field_tested) * torch.abs(field_target))

        return ((torch.sum(
                    (torch.abs(field_tested) ** 2 -
                     torch.abs(field_target) ** 2)
                    ** self.settings['amplitude_exponent'])) *
                ((torch.sum(
                    (torch.angle(field_tested) ** 2 -
                     torch.angle(field_target) ** 2)
                    ** self.settings['phase_exponent'])))
                )

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
        {'title': 'Max Iterations', 'name': 'max_iter', 'type': 'int', 'value': 400, 'min': 1},
        {'title': 'Tolerance', 'name': 'tolerance', 'type': 'float', 'value': 1e-20},
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
        self.image_tensor: torch.Tensor = None

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
            self._target_tensor = torch.tensor(self._target_field.field)

            #normalize target_intensity wrt input amplitude
            self._target_tensor = self._target_tensor / torch.abs(self._target_tensor).max()
            self._target_tensor *= (torch.sum(self._amplitude_tensor ** 2) /
                                    torch.sum(torch.abs(self._target_tensor)**2))

    def do_things_after_init(self):
        # Initialize phase distribution as trainable parameter
        self.phase_distribution = self.define_input_phase()
        self._amplitude_tensor = torch.tensor(self.object_field.amplitude)

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
        self._phase_tensor = torch.tensor(value, requires_grad=True)

    def compute_image_field(self, phase_input: torch.Tensor) -> torch.Tensor:
        image_tensor = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.fftshift(
                    self._amplitude_tensor * torch.exp(1j * phase_input)
                ), norm='forward'
            )
        )
        self.image_tensor = image_tensor
        return image_tensor


    def compute_loss(self, phase) -> torch.Tensor:
        self.compute_image_field(phase)

        if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
            slices = self.get_mask_slices(ApplyMaskTo.TARGET)
        else:
            slices = (Ellipsis, Ellipsis)

        loss = self._loss.compute_loss(self.image_tensor[*slices],
                                       self._target_tensor[*slices])

        self._calculated_fitness = loss.item()
        return loss


    def callback(self, phase):
        self.iter += 1

        image_array = self.image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(image_array), np.angle(image_array)))
        phase = phase.detach().numpy().reshape(self.object_field.shape)
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        self.set_phase_in_object_plane(phase)
        print(f'{self.iter}')
        self.parent_app.fields_to_plot.emit(self.get_fields_to_plot())
        QtWidgets.QApplication.processEvents()
        if not self._running:
            ### todo could use that call to stop the inner minimize loop
            # PR in pytorch-minimize in that direction submitted
            return True

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
        img_array = self.image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return self._calculated_fitness





