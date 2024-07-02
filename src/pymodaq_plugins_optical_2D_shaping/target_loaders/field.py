import numpy as np
from copy import copy

from numbers import Number

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataRaw
from pymodaq.utils import math_utils as mutils


logger = set_logger(get_module_name(__file__))


class Field:
    def __init__(self, amplitude: np.ndarray = None, phase: np.ndarray = None):
        self._amplitude: np.ndarray = None
        self._phase: np.ndarray = None

        if amplitude is not None:
            self.amplitude = amplitude
        if phase is not None:
            self.phase = phase

    def __repr__(self):
        return f'Field of shape {self.shape}'

    def __mul__(self, other):
        if isinstance(other, Number):
            field = copy(self)
            field.amplitude = field.amplitude * other
            return field

    @property
    def shape(self):
        if self.amplitude is not None:
            return self.amplitude.shape
        elif self.phase is not None:
            return self.phase.shape
        else:
            raise AttributeError('No amplitude nor phase array has been defined')

    @property
    def field(self) -> np.ndarray:
        """ Get set the field as a complex 2D array"""
        return self._amplitude * np.exp(1j * self._phase)

    @field.setter
    def field(self, field_array: np.ndarray):
        self._amplitude = np.abs(field_array)
        self._phase = np.angle(self.field)

    @property
    def amplitude(self) -> np.ndarray:
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

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.amplitude) ** 2

    def amplitude_as_dwa(self):
        return DataRaw('amplitude', data=[self.amplitude])

    @property
    def phase(self) -> np.ndarray:
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

    def intensity_as_dwa(self):
        return DataRaw('intensity', data=[self.intensity])

    def normalise_to_intensity(self, field: 'Field'):
        """ Normalise a Field object to this input total intensity

        Parameters
        ----------
        field: Field
            the Field to be normalised with respect to the total intensity

        Returns
        -------
        Field
        """
        return field * np.sum(self.intensity) / np.sum(field.intensity)


class GaussianIntensityField(Field):

    def __init__(self, npixels=(768, 1024), size_pixel=0.036, size=(11, 11)):
        super().__init__()
        self.size_pixel = size_pixel  # pixel size of SLM in mm
        self.size_x = size[1]  # x-axis intensity beam size in mm (FWHM)
        self.size_y = size[0]  # y-axis intensity beam size in mm (FWHM)
        self.npixels = npixels

        x = np.arange(0, npixels[1], 1)
        y = np.arange(0, npixels[0], 1)

        self.amplitude = np.sqrt(mutils.gauss2D(x, npixels[1] / 2, self.size_x / size_pixel,
                                                y, npixels[0] / 2, self.size_y / size_pixel))

