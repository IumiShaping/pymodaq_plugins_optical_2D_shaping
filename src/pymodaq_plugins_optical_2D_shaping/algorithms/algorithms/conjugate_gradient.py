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

from pytensor import tensor as tensor
from pytensor.tensor import fft
import pytensor as pt
from pytensor.compile.io import In

from scipy.fft import fft2

if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


class Model:
    """ Creates the model Parameter: the phase distribution from an initial phase

    Attributes:
    -----------
    phase_distribution:
        The tensor parameter to evolve
    phase_array: np.ndarray
        ReadOnly

    The Tensor and the array share the same memory
    """

    def __init__(self, object_field: Field, target_field: Field):
        super().__init__()
        self.target_field = target_field

        # object_field = object_field.pad(
        #     ((int(target_field.shape[0] / 2), int(target_field.shape[0] / 2)),
        #      (int(target_field.shape[1] / 2), int(target_field.shape[1] / 2)))
        # )
        self._object_amplitude_array = object_field.amplitude
        self._phase_distribution = pt.shared(object_field.phase, 'phase')

        self.loss = self.compute_loss_function()

    @property
    def phase_distribution(self):
        return self._phase_distribution.get_value()

    @phase_distribution.setter
    def phase_distribution(self, value: np.ndarray):
        self._phase_distribution.set_value(value)

    def compute_loss_function(self) -> type[pt.function]:
        fft = fft2(self._object_amplitude_array * tensor.cos(self._phase_distribution), norm='ortho')
        loss = tensor.abs((self.image_field[...,0] - self.target_field.real_field)**2 +
                          (self.image_field[...,1] - self.target_field.imag_field)**2)
        return pt.function([], loss)



if __name__ == '__main__':
    from pymodaq_gui.utils.dock import DockArea

    from pymodaq_plugins_optical_2D_shaping.field.field_loader_app import FieldLoaderApp
    from pymodaq_plugins_optical_2D_shaping.field.loaders.save_field_loader import SavedFieldLoader
    from pymodaq_plugins_optical_2D_shaping.field.loaders.from_functions_loader import GaussianIntensity
    from qtpy import QtWidgets
    from pymodaq_data import DataToExport, DataRaw

    app = mkQApp('CG')

    field_loader = FieldLoaderApp(DockArea())

    field_loader.loader = 'GaussianIntensityLoader'
    field_loader.load_field()
    object_field = field_loader.field
    object_field.phase = np.random.rand(*object_field.shape) * 2 * np.pi
    object_field.amplitude_as_dwa().plot('qt')

    field_loader.loader = 'SavedFieldLoader'
    field_loader.load_field()
    ferris_field = field_loader.field
    ferris_field.amplitude_as_dwa().plot('qt')

    model = Model(object_field, ferris_field)


    app.exec()



# @AlgorithmFactory.register_algorithm()
# class ConjugateGradient(AlgoBase):
#     """ Implementation of the ConjugateGradient iterative algorithm to create amplitude and phase modulated
#     image with phase only spatial light modulators in the Fourier plane of a converging lens
#
#     Based on Vol. 25, No. 10 | 15 May 2017 | OPTICS EXPRESS 11695 and implemented here with pytorch
#
#     The corresponding experimental setup should define a working light wavelength and a focal length
#     of the used lens
#     """
#
#     ALGO_NAME = 'ConjugateGradient'
#     SETUP_TYPE = LensSetup.TwoF
#     ITERATIVE = True
#
#     params = []
#
#     def __init__(self, parent: 'AlgoApp' = None):
#         super().__init__(parent)
#         self._slices: tuple[slice, slice] = None
#
#         self.phase_distribution: torch.nn.Parameter = None
#         self._calculated_fitness = 0.
#
#         self._loss: torch.Tensor = None
#         self.target_torch: torch.Tensor = None
#         self.image_torch: torch.Tensor = None
#
#
#     def value_changed(self, param: Parameter):
#         self.parent_app.algo_settings_changed()
#
#     def do_things_after_init(self):
#         QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
#
#         # Initialize phase distribution as trainable parameter
#         self.phase_distribution = torch.nn.Parameter(torch.asarray(self.object_field.phase))
#         self.target_torch = torch.asarray(self._target_field.field)
#
#         QtWidgets.QApplication.restoreOverrideCursor()
#
#     @property
#     def object_torch(self) -> torch.Tensor:
#         return torch.Tensor(torch.asarray(self.object_field.amplitude)) * torch.exp(1j * self.phase_distribution)
#
#     def propagate_field(self):
#         self.image_torch = torch.fft.fft2(self.object_torch)
#
#
#
#     def compute_loss(self):
#
#         if self.apply_mask(apply_to=ApplyMaskTo.TARGET):
#             if self.update_mask:
#
#                 self._slices = self.get_mask_slices(ApplyMaskTo.TARGET)
#                 self.update_mask = False
#         else:
#             self._slices = (Ellipsis, Ellipsis)
#
#         d=2
#         max_amplitude = torch.max(torch.abs(self.image_torch[*self._slices])**2 *
#                                   torch.abs(self.target_torch[*self._slices])**2)
#
#         loss = 10 ** d * (
#                 1 - torch.sum(
#             torch.sqrt(
#                 1/ max_amplitude *
#                 torch.abs(self.image_torch[*self._slices])**2 *
#                 torch.abs(self.target_torch[*self._slices])**2) *
#             torch.cos(torch.angle(self.image_torch[*self._slices]) -
#                       torch.angle(self.target_torch[*self._slices]))))**2
#
#         loss = (self.image_torch[*self._slices] - self.target_torch[*self._slices]).abs().pow(2).sum()
#         loss.backward(retain_graph=True)
#
#         self._calculated_fitness = loss.item()
#         return loss
#
#     def evolve_field(self):
#         pass
#
#
#     def compute_phase(self):
#         self.propagate_field()
#         self.compute_loss()
#         self.evolve_field()
#
#         self.set_phase_in_object_plane(self.phase_distribution.data.numpy(force=True))
#         img_array = self.image_torch.numpy(force=True)
#         self._image_field = self.scale_target_with_geometry(
#             Field('image', np.abs(img_array), np.angle(img_array)))
#
#
#     @property
#     def fitness(self) -> float:
#         """ Compute fitness with respect to the image_field and target_field """
#         return float(self._calculated_fitness)
#
#
#
#
#
