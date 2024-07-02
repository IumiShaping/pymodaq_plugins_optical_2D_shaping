# -*- coding: utf-8 -*-
"""
Created the 20/07/2023

@author: Sebastien Weber
"""
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale, resize

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq.utils import math_utils as mutils

from pymodaq_plugins_optical_2D_shaping.algorithms.factory import AlgorithmFactory
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase, GaussianIntensityField, Field

logger = set_logger(get_module_name(__file__))

cheshire_cat_path = Path(__file__).parent.parent.joinpath('resources/cheshirecat_rect.png')

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


@AlgorithmFactory.register_algorithm()
class GbSax(AlgoBase):

    ALGO_NAME = 'Gerchberg–Saxton'

    params = [

    ]

    def __init__(self):
        super().__init__()

    def load_target_data(self, data: DataRaw):
        self.set_target(data[0])

    def set_target(self, target_intensity: np.ndarray):

        target_intensity = self.check_target_object_ratio(target_intensity)

        self.target_intensity = target_intensity

        self.image_shape = self.target_intensity.shape
        self.target_intensity = self.input_intensity.normalise_to_intensity(self.target_intensity)

        self.set_phase_in_object_plane((np.random.rand(*self.object_shape) - 0.5) * 2 * np.pi)
        self.propagate_field()

    def check_target_object_ratio(self, target: np.ndarray):

        ratio = np.max(np.array(self.object_shape) / np.array(target.shape))
        target = rescale(target, 1.1 * ratio)

        return target

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        self.field_object_phase = phase
        if phase.shape == self.object_shape:
            self.field_object = self.input_intensity.amplitude *\
                                (induced_amplitude if induced_amplitude is not None else 1) * np.exp(1j * phase)
            self.propagate_field()
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    def get_phase(self):
        return self.field_object_phase

    def get_npad(self):
        npad_before = (np.abs(np.array(self.object_shape) - np.array(self.image_shape)) // 2).astype(int)
        npad_after = (npad_before + np.abs(np.array(self.object_shape) - np.array(self.image_shape)) % 2).astype(int)
        return (npad_before[0], npad_after[0]), (npad_before[1], npad_after[1])

    def propagate_field(self):
        field_object_padded = np.pad(self.field_object, self.get_npad(), constant_values=(0, 0))
        self.field_image = fftshift(fft2(fftshift(field_object_padded))) / \
                           np.sqrt(np.prod(field_object_padded.shape))

    def evolve_field(self) -> np.ndarray:
        npad = self.get_npad()
        self._sse = 100 * np.sum(np.abs(np.sqrt(self.target_intensity) - self.field_image)) ** 2 \
                    / np.prod(self.image_shape) / np.sum(self.target_intensity)
        field_image_corrected = np.sqrt(self.target_intensity) * np.exp(1j * np.angle(self.field_image))
        field_object_corrected = ifftshift(ifft2(ifftshift(field_image_corrected)))
        field_object_corrected = field_object_corrected[npad[0][0] + 1:npad[0][0] + 1 + self.object_shape[0],
                                 npad[1][0] + 1:npad[1][0] + 1 + self.object_shape[1]]
        self.field_object_phase = np.angle(field_object_corrected)
        return self.field_object_phase

    @property
    def fitness(self) -> float:
        return self._sse

    def grab(self):
        self.propagate_field()
        return self.field_image

    def evolve(self):
        self.evolve_field()

    @property
    def sse(self):
        return self._sse

    @property
    def intensity_image(self):
        return np.abs(self.field_image) ** 2


if __name__ == '__main__':

    from pymodaq.utils.gui_utils.utils import mkQApp
    app = mkQApp('GbSax')

    target_intensity = imread(cheshire_cat_path)
    if len(target_intensity.shape) == 3:
        target_intensity = rgb2gray(target_intensity[..., 0:3])

    target = Field(amplitude=np.sqrt(target_intensity))

    algo = GbSax()

    algo.set_object_field(GaussianIntensityField())
    algo.set_target_field()




