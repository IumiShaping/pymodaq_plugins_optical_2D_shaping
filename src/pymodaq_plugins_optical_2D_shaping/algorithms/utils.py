
from abc import ABCMeta, abstractproperty
from typing import Tuple, TYPE_CHECKING, Union

import numpy as np
from qtpy import QtWidgets
from pymodaq_utils import math_utils as mutils
from pymodaq_gui.managers.parameter_manager import ParameterManager, Parameter
from pymodaq_utils.enums import BaseEnum
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataRaw, DataToExport
from pymodaq_utils.enums import StrEnum

from pymodaq_plugins_optical_2D_shaping.field import Field, FieldLoader, Q_

if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))


class MaskError(Exception):
    pass


class TargetPhase(StrEnum):
    RANDOM = 'random'
    QUADRATIC = 'quadratic'  # see https://doi.org/10.1364/OE.25.014323


class AlgoParameterManager(ParameterManager):
    settings_name = 'algo_settings'

    def __init__(self):
        super().__init__()
        self.settings_tree.header().setVisible(True)
        self.settings_tree.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.settings_tree.header().setMinimumSectionSize(150)
        self.settings_tree.setMinimumHeight(150)


class AlgoBase(AlgoParameterManager, metaclass=ABCMeta):
    """
    Here goes the abstract methods and shared properties/attributes of all algorithms used to
    calculate amplitude/phase shaping
    """

    ALGO_NAME = abstractproperty()
    ITERATIVE = False
    params = []

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__()

        self.parent_app = parent
        self._target_field = Field()
        self._input_field = Field()
        self._object_field = Field(amplitude=self._input_field.amplitude.copy())
        self._object_field.calibrate_axes(self._input_field.pixels_sizes)
        self._image_field = Field()
        self.mask: tuple[slice, slice] = None

    def quit(self):
        """ to reimplement if neccessary"""
        pass

    def set_mask(self, slices: Union[tuple[slice, slice], None] = None):
        """ Get a mask object from which one can compute image_field constraints (or not)"""
        self.mask = slices

    def mask_from_slices(self) -> Field:
        """ Return a Field to be used to mask within the algorithm

        To be reimplemented if needed

        Examples
        --------
        mask = Field.init_from_field(self._target_field).amplitude * 0
        mask[*self.mask] = 1
        return Field(amplitude=mask)
        """
        raise MaskError

    @property
    def image_field(self) -> Field:
        return self._image_field

    @property
    def object_field(self) -> Field:
        return self._object_field

    def set_input_field(self, field: Field):
        self._input_field = field
        self.do_things_after_set_input()

    def define_input_phase(self, phase_type: 'TargetPhase'):
        shape = self._object_field.shape
        if phase_type == TargetPhase.RANDOM:
            phase = np.random.random_sample(shape) * 2 *np.pi
        elif phase_type == TargetPhase.QUADRATIC:
            ny, nx = shape
            x = np.pi / nx * np.linspace(-nx/2, nx/2 , nx , endpoint=False)**2
            y = np.pi / ny * np.linspace(-ny/2, ny/2 , ny , endpoint=False)**2
            xv, yv = np.meshgrid(x, y)
            phase = xv + yv
        else:
            raise ValueError('Unknown phase type')

        self.set_phase_in_object_plane(phase)

    def set_object_field(self, field: Field):
        self._object_field = field

    def set_target_field(self, field: Field):
        self._target_field = field
        self._image_field = Field.init_from_field(self._target_field)

    def set_target_intensity(self, intensity: np.ndarray):
        self._target_field.amplitude = np.sqrt(intensity)

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        if phase.shape == self._object_field.shape:
            self._object_field.phase = phase.copy()
            self._object_field.amplitude = (
                    self._input_field.amplitude.copy() *
                    (induced_amplitude if induced_amplitude is not None else 1))
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    @property
    def intensity_image(self):
        return self._image_field.intensity

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field

        To be subclassed"""
        raise NotImplementedError

    def fitness_as_dwa(self):
        return DataRaw('fitness', data=[np.array([self.fitness])])

    def compute_phase(self):
        """ Compute the phase to apply to SLM given the target object

        To be subclassed in real implementation
        """

        raise NotImplementedError

    def get_fields_to_plot(self) -> DataToExport:
        return DataToExport('AlgoData', data=[
            self.image_field.amplitude_as_dwa('image'),
            self.image_field.phase_as_dwa('image'),
            self.fitness_as_dwa(),
            self.object_field.amplitude_as_dwa('object'),
            self.object_field.phase_as_dwa('object'),
        ])

    def do_things_after_set_input(self):
        """ to reimplement if needed"""
        pass

    def scale_target_with_geometry(self, field: Field):
        """ Apply an axis scaling to have the target and its axes in correct units with respect to
        a given algorithm implementation and experimental setup
        """
        field.calibrate_axes(self.get_target_pixels_size())
        return field

    def get_target_pixels_size(self, slm_size: Tuple[Q_, Q_] = None) -> list[Q_]:
        """ Get the expected physical size of the pixels in the target plane given
        the chosen algorithm and physical parameters: focal length, wavelength...

        To be reimplemented by real Algorithm
        """
        raise NotImplementedError

    def get_npad_between_image_object(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ Get the padding necessary to match object shape and image shape

        If positive, the image shape is bigger than the object
        If negative, the object shape is bigger than the image
        """
        npad_before = ((np.array(self._image_field.shape) -
                        np.array(self._object_field.shape)) // 2).astype(int)
        npad_after = (np.array(self._image_field.shape) -
                        np.array(self._object_field.shape)) - npad_before
        return (npad_before[0], npad_after[0]), (npad_before[1], npad_after[1])

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()