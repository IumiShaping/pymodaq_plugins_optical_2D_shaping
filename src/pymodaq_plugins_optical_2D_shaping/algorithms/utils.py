
from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets
from pymodaq.utils import math_utils as mutils
from pymodaq.utils.managers.parameter_manager import ParameterManager, Parameter
from pymodaq.utils.enums import BaseEnum
from pymodaq.utils.logger import set_logger, get_module_name

from pymodaq_plugins_optical_2D_shaping.target_loaders.field import Field, GaussianIntensityField


logger = set_logger(get_module_name(__file__))


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

    def __init__(self):
        super().__init__()

        self._target_field = Field()
        self._object_field = GaussianIntensityField()
        self._image_field = Field()

        self._sse: float = 100

    def set_object_field(self, field: Field):
        self._object_field = field

    def set_target_field(self, field: Field):
        self._target_field = field

    def set_target_intensity(self, intensity: np.ndarray):
        self._target_field.amplitude = np.sqrt(intensity)

    @property
    def fitness(self) -> float:
        return self._sse

    @property
    def sse(self) -> float:
        return self._sse

    def evolve(self):
        """ Compute the image field given the input field and compute the fitness

        To be subclassed in real implementation
        """

        raise NotImplementedError
