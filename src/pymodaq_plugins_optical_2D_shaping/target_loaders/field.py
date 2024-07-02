import numpy as np

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataRaw


logger = set_logger(get_module_name(__file__))


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
            if not self._phase.shape == amp_array.shape:
                logger.warning('New amplitude is not coherent with existing phase shape'
                               'Setting the phase to flat zeros')
                self._phase = np.zeros_like(amp_array)
        else:
            self._phase = np.zeros_like(amp_array)

        self._amplitude = amp_array

    def amplitude_as_dwa(self):
        return DataRaw('amplitude', data=[self.amplitude])

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_array: np.ndarray):
        if self._amplitude is not None:
            if not self._amplitude.shape == phase_array.shape:
                logger.warning('New phase is not coherent with existing amplitude shape'
                               'Setting the amplitude to flat ones')
                self._amplitude = np.ones_like(phase_array)
        else:
            self._amplitude = np.ones_like(phase_array)

        self._phase = phase_array

    def phase_as_dwa(self):
        return DataRaw('phase', data=[self.phase])
