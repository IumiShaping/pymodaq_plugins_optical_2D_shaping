# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

see https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/

@author: Sebastien Weber
"""
from typing import Optional, Union

from typing import Union, Tuple, List, TYPE_CHECKING, Any


import numpy as np
from torch import nn

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo, TargetPhase
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig

import torch
from torch.nn import MSELoss, Module
from torchmin import minimize

from torchmin.optim import Minimizer


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


class Model(Module):
    def __init__(self, object_field: Field, target_field: Field, exponent: int = 4):
        super().__init__()

        self.object_phase_tensor = nn.Parameter(torch.tensor(object_field.phase, requires_grad=True))

        self.object_amplitude_tensor = torch.tensor(object_field.amplitude)
        self.exponent = exponent

        self._image_tensor: torch.Tensor = None
        self._target_tensor: torch.Tensor = None

        self._slices = (Ellipsis, Ellipsis)

        self.set_target_from_field(target_field)

    def set_target_from_field(self, target: Field):
        self._target_tensor = torch.Tensor(target.field)

        # normalize target_intensity wrt input amplitude
        self._target_tensor = self._target_tensor / torch.abs(self._target_tensor).max()
        self._target_tensor *= (torch.sum(torch.abs(self.object_amplitude_tensor) ** 2) /
                                torch.sum(torch.abs(self._target_tensor) ** 2))

    @property
    def slices(self) -> tuple[Union[slice, Ellipsis], Union[slice, Ellipsis]]:
        return self._slices

    @slices.setter
    def slices(self, slices: tuple[slice, slice] = None):
        if slices is None:
            self._slices = (Ellipsis, Ellipsis)
        else:
            self._slices = slices

    @property
    def image_field(self) -> Field:
        field_array = self._image_tensor.detach().numpy()
        return Field('image', amplitude=np.abs(field_array), phase=np.angle(field_array),
                     pixel_sizes=self._target_field.pixels_sizes)

    def forward(self) -> torch.Tensor:
        return self.compute_image_tensor(self.object_phase_tensor)

    def compute_image_tensor(self, phase_input: torch.Tensor) -> torch.Tensor:
        image_tensor = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.fftshift(
                    self.object_amplitude_tensor * torch.exp(1j * phase_input)
                ), norm='forward'
            )
        )
        self._image_tensor = image_tensor
        return image_tensor

    def loss_function(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return ((torch.sum(
            (torch.abs(image_tensor[*self.slices]) ** 2 -
             torch.abs(self._target_tensor[*self.slices]) ** 2)
            ** self.exponent)))

    def compute_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        loss = self.loss_function(image_tensor)
        return loss


@AlgorithmFactory.register_algorithm()
class NCGConjugateGradient(AlgoBase):
    """ Implementation of the ConjugateGradient iterative algorithm to create amplitude and phase modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    Based on Vol. 25, No. 10 | 15 May 2017 | OPTICS EXPRESS 11695 and implemented here with pytorch

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'NCGConjugateGradient'
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = True

    params = [
        {'title': 'Method', 'name': 'method', 'type': 'list', 'value': 'cg', 'limits': methods},
        {'title': 'Max Iterations', 'name': 'max_iter', 'type': 'int', 'value': 20, 'min': 1},
        {'title': 'Loss exponent', 'name': 'exponent', 'type': 'int', 'value': 4, 'min': 2},

    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)
        self._fitness = 1.0
        self.iter = 0
        self._algo_init = False
        self.model:  Model = None
        self.phase_tensor: torch.Tensor = None

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()
        super().value_changed(param)

        if param.name() == 'exponent':
            self.model.exponent = param.value()

    def do_things_after_init(self):

        if not self._algo_init:
            self._algo_init = True

        self.define_input_phase()
        self.compute_forward_fft(update_plots=True)

        self.model = Model(object_field=self._object_field,
                           target_field=self._target_field,
                           exponent=self.settings['exponent'])

        if self.apply_mask(ApplyMaskTo.TARGET):
            self.model.slices = self.get_mask_slices(ApplyMaskTo.TARGET)

        self.optim = Minimizer(self.model.parameters(),
                               method=self.settings['method'],
                               max_iter=self.settings['max_iter'],
                               )

        self.iter = 0

    def closure(self):
        self.optim.zero_grad()
        loss = self.model.compute_loss(self.model())
        return loss

    def compute_phase(self, do_step=True, **kwargs):

        self.iter += 1
        self._fitness = self.optim.step(self.closure).item()

        self.set_phase_in_object_plane(list(self.model.parameters())[0].data.detach().numpy())
        img_array = self.model._image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))

        self.parent_app.fields_to_plot.emit(self.get_fields_to_plot())

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field """
        return self._fitness





