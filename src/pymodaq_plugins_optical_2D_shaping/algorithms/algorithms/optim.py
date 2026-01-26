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

from pymodaq_plugins_optical_2D_shaping.algorithms.loss import LossFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo, TargetPhase
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithms.minimizer import Minimize

import torch
from torch.nn import MSELoss, Module



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)


if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()
loss_factory = LossFactory()



@AlgorithmFactory.register_algorithm()
class TorchOptim(Minimize):
    """ Implementation of the ConjugateGradient iterative algorithm to create amplitude and phase modulated
    image with phase only spatial light modulators in the Fourier plane of a converging lens

    Based on Vol. 25, No. 10 | 15 May 2017 | OPTICS EXPRESS 11695 and implemented here with pytorch

    The corresponding experimental setup should define a working light wavelength and a focal length
    of the used lens
    """

    ALGO_NAME = 'TorchOptim'
    SETUP_TYPE = LensSetup.TwoF
    ITERATIVE = True
    MANUAL_LOOP = True


    def do_things_after_init(self):
        # Initialize phase distribution as trainable parameter
        self.phase_distribution = self.define_input_phase()
        self._amplitude_tensor = torch.tensor(self.object_field.amplitude)

        if not self._algo_init:
            self._algo_init = True
            self.do_things_after_set_target()

        self.compute_image_field(self._phase_tensor)
        self.update_data(self._phase_tensor.reshape(np.prod(self._phase_tensor.shape)))

        self.optimizer = torch.optim.LBFGS(
            [self._phase_tensor],
            lr=0.01, max_iter=self.settings['max_iter'], max_eval=None, tolerance_grad=self.settings['tolerance'],
            tolerance_change=1e-09,
            history_size=100, line_search_fn=None)
        self.iter = 0

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss(self._phase_tensor)
        loss.backward()
        return loss

    def compute_phase(self, do_step=True, ini_phase=None, **kwargs):
        if ini_phase is not None:
            self.phase_distribution = ini_phase

        self.optimizer.step(self.closure)

        self.set_phase_in_object_plane(self.phase_distribution)
        img_array = self.image_tensor.detach().numpy()
        self._image_field = self.scale_target_with_geometry(
            Field('image', np.abs(img_array), np.angle(img_array)))



