import numpy as np
from copy import copy
from typing import Tuple, Iterable
from numbers import Number

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataRaw
from pymodaq.utils import math_utils as mutils


logger = set_logger(get_module_name(__file__))


try:
    import pyfftw
    pyfftw.interfaces.cache.enable()


    def wrap_fft(*args, **kwargs):
        fft2 = pyfftw.interfaces.numpy_fft.fft2(threads=8, *args, **kwargs)
        return fft2


    def wrap_ifft(*args, **kwargs):
        ifft2 = pyfftw.interfaces.numpy_fft.ifft2(threads=8, *args, **kwargs)
        return ifft2


    def wrap_fftshift(*args, **kwargs):
        fftshift = pyfftw.interfaces.numpy_fft.fftshift(*args, **kwargs)
        return fftshift


    def wrap_ifftshift(*args, **kwargs):
        ifftshift = pyfftw.interfaces.numpy_fft.ifftshift(*args, **kwargs)
        return ifftshift


    fft2 = wrap_fft
    ifft2 = wrap_ifft

    fftshift = wrap_fftshift
    ifftshift = wrap_ifftshift

except ImportError:
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2
    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift
    print("Warning: using numpy FFT implementation.  "
          "Consider using pyFFTW for faster Fourier transforms.")


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

    @staticmethod
    def init_from_field(field: 'Field') -> 'Field':
        """ Create a field object with flat amplitude and zero phase but with the same shape as
        the field parameter"""
        field_new = Field()
        field_new.amplitude = np.ones_like(field.amplitude)
        return field_new

    @property
    def shape(self):
        if self.amplitude is not None:
            return self.amplitude.shape
        elif self.phase is not None:
            return self.phase.shape
        else:
            raise AttributeError('No amplitude nor phase array has been defined')

    def pad(self, pad_width: Tuple[Tuple[int, int], Tuple[int, int]], **kwargs) -> 'Field':
        """ Get a Field object similar to self but padded

        see numpy.pad method for the signature and possible named arguments
        """
        return Field(amplitude=np.pad(self.amplitude, pad_width, **kwargs),
                     phase=np.pad(self.phase, pad_width, **kwargs))

    def unpad(self, pad_width: Tuple[Tuple[int, int], Tuple[int, int]],
              ini_shape: Tuple[int, int]) -> 'Field':
        """ Get a Field object from a padded Field object

        see numpy.pad method for the signature
        """
        return Field(self.amplitude[pad_width[0][0] + 1:pad_width[0][0] + 1 + ini_shape[0],
                     pad_width[1][0] + 1:pad_width[1][0] + 1 + ini_shape[1]],
                     self.phase[pad_width[0][0] + 1:pad_width[0][0] + 1 + ini_shape[0],
                     pad_width[1][0] + 1:pad_width[1][0] + 1 + ini_shape[1]])

    def fft2(self):
        field_array = fftshift(fft2(fftshift(self.field))) / \
                      np.sqrt(np.prod(self.shape))
        field = Field()
        field.field = field_array
        return field

    def ifft2(self):
        field_array = fftshift(ifft2(fftshift(self.field))) / \
                      np.sqrt(np.prod(self.shape))
        field = Field()
        field.field = field_array
        return field

    @property
    def field(self) -> np.ndarray:
        """ Get set the field as a complex 2D array"""
        return self._amplitude * np.exp(1j * self._phase)

    @field.setter
    def field(self, field_array: np.ndarray):
        self._amplitude = np.abs(field_array)
        self._phase = np.angle(field_array)

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

