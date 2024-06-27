from abc import ABC, abstractproperty

import numpy as np
from utils import math_utils as mutils


class InputIntensity:
    def __init__(self, npixels=(768, 1024), size_pixel=0.036, size=(11, 11)):
        self.size_pixel = size_pixel  # pixel size of SLM in mm
        self.size_x = size[1]  # x-axis intensity beam size in mm (FWHM)
        self.size_y = size[0]  # y-axis intensity beam size in mm (FWHM)

        self.npixels = npixels
        x = np.arange(0, npixels[1], 1)
        y = np.arange(0, npixels[0], 1)

        #   ===   Amplitude   ===============================================
        self._amp = np.sqrt(mutils.gauss2D(x, npixels[1] / 2, self.size_x / size_pixel,
                                           y, npixels[0] / 2, self.size_y / size_pixel))

    @property
    def amplitude(self):
        return self._amp

    @property
    def intensity(self):
        return np.power(self._amp, 2.)

    def normalise_to_intensity(self, data_int: np.ndarray):
        """ Normalise an intensity like 2D array to this input total intensity

        Parameters
        ----------
        data_int: ndarray
            the array to be normalised with respect to the total intensity

        Returns
        -------
        ndarray
        """
        return data_int * np.sum(self.intensity) / np.sum(data_int)


class AlgoBase(ABC):
    """
    Here goes the abstract methods and shared properties/attributes of all algorithms used to
    calculate amplitude/phase shaping
    """

    ALGO_NAME = abstractproperty()

    def __init__(self):
        pass
