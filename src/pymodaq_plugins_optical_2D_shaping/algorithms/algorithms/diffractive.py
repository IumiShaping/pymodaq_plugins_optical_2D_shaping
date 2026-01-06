# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Tuple, List, Callable, TYPE_CHECKING

import numpy as np


from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_data import Q_, DataToExport


from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, Field, LensSetup, ApplyMaskTo
from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig


if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()



class PhasePattern(ABC):
    
    NAME: str = None  # to be reimplemented
    METHOD: str = None  # letter from the method description in the DOI:10.1364/OE.24.006249 | OPTICS EXPRESS 6253 paper

    def __init__(self, input_field: Field, target_field: Field, gamma: int):
        self.input_field = input_field
        self.target_field = target_field
        self.gamma = gamma

    @abstractmethod
    def generate_phase(self) -> np.ndarray:
        """ Main method that generate the phase pattern to be applied to the SLM"""
        ...

    def generate_grating_phase(self) -> np.ndarray:
        x_array = np.linspace(0, self.input_field.shape[1], self.input_field.shape[1])
        return (2 * np.pi * x_array / self.gamma) % 2*np.pi


class PatternFactory:
    """The factory class for creating Diffractive Phases to be used with the diffractive algorithm below"""

    _registry: dict[str, type[PhasePattern]] = {}

    @classmethod
    def register(cls) -> Callable:
        """Class decorator method to register class to the internal registry. Must be used as
        decorator above the definition of a DiffractiveBase inherited class.

        The Diffractive class must implement specific class attributes and methods
        """

        def inner_wrapper(wrapped_class: type[PhasePattern]) -> type[PhasePattern]:
            name = wrapped_class.NAME
            if name is None:
                raise NotImplementedError('The reimplemented class must have a valid NAME string attribute')
            if wrapped_class.METHOD is None:
                raise NotImplementedError('The reimplemented class must have a valid METHOD letter string attribute')
            if name not in cls._registry:
                cls._registry[name] = wrapped_class
            # Return wrapped_class
            return wrapped_class
        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_phase(cls, name: str, **kwargs) -> np.ndarray:
        """Factory command to get registered Phase patterns
        """
        if name not in cls._registry:
            raise ValueError(f".{name} is not a supported Algorithm.")

        return cls._registry[name](**kwargs).generate_phase()

    @property
    def keys(self):
        return list(self._registry.keys())




@PatternFactory.register()
class RelativeAmplitude(PhasePattern):
    NAME = 'Relative Amplitude'
    METHOD = 'A'

    def generate_phase(self) -> np.ndarray:
        relative_amplitude = ((self.target_field.amplitude / self.target_field.amplitude.max()) /
                              (self.input_field.amplitude / self.input_field.amplitude.max()))
        relative_amplitude = relative_amplitude / relative_amplitude.max()
        return relative_amplitude * self.generate_grating_phase()


@PatternFactory.register()
class Scattering(PhasePattern):
    NAME = 'Scattering'
    METHOD = 'B'

    def generate_phase(self) -> np.ndarray:
        alpha = np.max(self.input_field.amplitude) / np.max(self.target_field.amplitude)

        interference = (self.input_field +
                        alpha * self.target_field.amplitude *
                        np.exp(1j * (self.target_field.phase +  self.generate_grating_phase())))
        phase = interference.phase
        return phase



diff_factory = PatternFactory()


@AlgorithmFactory.register_algorithm()
class OE2016(AlgoBase):
    """ Implementation of the algorithms discussed in OPTICS EXPRESS DOI:10.1364/OE.24.006249
    The algorithms create amplitude image with phase only spatial light modulators
    in a 4f line with two lens and a circular filter in the Fourier Plane to filter out undiffracted light

    The corresponding experimental setup should define a working light wavelength and focal lengths
    of the used lens
    """

    ALGO_NAME = 'OE2016 - Amplitude diffractive'
    SETUP_TYPE = LensSetup.FourF
    ITERATIVE = False

    params = [
        {'title': 'Gamma', 'name': 'gamma', 'type': 'int', 'value': 4,
         'tip': 'Grating period of the modulation in pixels'},
        {'title': 'Diffractive Models', 'name': 'diff_model', 'type': 'list', 'value': diff_factory.keys[0],
         'limits': diff_factory.keys},
    ]

    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__(parent)

    def compute_phase(self):

        diff_phase = diff_factory.get_phase(self.settings['diff_model'],
                                            input_field=self._input_field,
                                            target_field=self._target_field,
                                            gamma=self.settings['gamma'])

        self.set_phase_in_object_plane(diff_phase)

        circ_aperture = self.get_mask_field(apply_to=ApplyMaskTo.INTERMEDIATE, inner_value=1, outer_value=0)
        self.intermediate_field = self.object_field.fft2() * circ_aperture.amplitude
        self.intermediate_field.calibrate_axes(self.intermediate_pixel_sizes)
        self.intermediate_field.axes = self.intermediate_field.get_axes()

        self._image_field = self.intermediate_field.ifft2()

        self.image_field.calibrate_axes(self._target_field.pixels_sizes)
        self.image_field.axes = self.image_field.get_axes()

    def value_changed(self, param):
        self.parent_app.compute_phase()





