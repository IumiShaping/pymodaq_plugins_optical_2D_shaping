from abc import ABC, abstractproperty

import numpy as np

from pymodaq.utils.enums import BaseEnum, enum_checker
from pymodaq.utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))


class LoadTypeEnum(BaseEnum):

    AMPLITUDE = 'amplitude'
    PHASE = 'phase'


class Field:
    def __init__(self):
        self._amplitude: np.ndarray = None
        self._phase: np.ndarray = None

    def __repr__(self):
        return f'Field of shape {self.shape}'

    @property
    def shape(self):
        if self.amplitude is not None:
            return self.amplitude.shape
        elif self.phase is not None:
            return self.phase.shape
        else:
            raise AttributeError('No amplitude nor phase array has been defined')

    @property
    def field(self):
        """ Get set the field as a complex 2D array"""
        return self._amplitude * np.exp(1j * self._phase)

    @field.setter
    def field(self, field_array: np.ndarray):
        self._amplitude = np.abs(field_array)
        self._phase = np.angle(self.field)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amp_array: np.ndarray):
        if self._phase is not None:
            if self._phase.shape == self._amplitude.shape:
                self._amplitude = amp_array
            else:
                logger.warning('New amplitude is not coherent with existing phase shape'
                               'Setting the phase to flat zeros')
                self._phase = np.zeros_like(amp_array)
        else:
            self._phase = np.zeros_like(amp_array)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_array: np.ndarray):
        if self._amplitude is not None:
            if self._amplitude.shape == self._phase.shape:
                self._phase = phase_array
            else:
                logger.warning('New phase is not coherent with existing amplitude shape'
                               'Setting the amplitude to flat ones')
                self._amplitude = np.ones_like(phase_array)
        else:
            self._amplitude = np.ones_like(phase_array)


class TargetLoader(ABC):
    LOADER_NAME = abstractproperty()

    def __init__(self):
        self.field = Field()

    def load(self, *args, load_type=LoadTypeEnum.AMPLITUDE, **kwargs):
        """ Abstract method used to load a target to populate the field attribute"""

        raise NotImplementedError
