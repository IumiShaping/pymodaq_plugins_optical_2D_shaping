from numbers import Number
from typing import Tuple, Union

import numpy as np

from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pyqtgraph.parametertree import Parameter
from pymodaq.utils import math_utils as mutils


class GaussianIntensityField(Field):

    def __init__(self,
                 name='',
                 npixels: Tuple[int, int] = (768, 1024),
                 size_pixel: Union[float, Tuple[float, float]] = 0.036,
                 size_beam: Union[float, Tuple[float, float]] = (11., 11.)):
        """

        Parameters
        ----------
        npixels: Tuple[int, int]
            Number of pixels defining the field object
        size_pixel: Size of the underlying pixels in mm
        size_beam: Size of the underlying laser beam in mm
        """

        size_hor = size_beam[1]  # x-axis intensity beam size in mm (FWHM)
        size_ver = size_beam[0]  # y-axis intensity beam size in mm (FWHM)

        if isinstance(size_pixel, Number):
            size_pixel = (size_pixel, size_pixel)

        self.calibrate_axes(np.array(size_pixel) * 1e-3)  # calibration of the axes in meter

        x = np.arange(0, npixels[1], 1)
        y = np.arange(0, npixels[0], 1)

        amplitude = np.sqrt(mutils.gauss2D(x, npixels[1] / 2, size_hor / size_pixel[1],
                                           y, npixels[0] / 2, size_ver / size_pixel[0]))
        super().__init__(name, amplitude=amplitude)


@LoaderFactory.register_loader()
class GaussianIntensity(FieldLoader):

    LOADER_NAME = 'GaussianIntensityLoader'

    params = FieldLoader.params + \
        [{'title': 'Ny:', 'name': 'ny_pixels', 'type': 'int', 'value': 768,},
         {'title': 'Nx:', 'name': 'nx_pixels', 'type': 'int', 'value': 1024, },
         {'title': 'Pixel size x (µm):', 'name': 'pixel_size_x', 'type': 'float', 'value': 36., },
         {'title': 'Pixel size y (µm):', 'name': 'pixel_size_y', 'type': 'float', 'value': 36., },
         {'title': 'Beam size x (mm):', 'name': 'beam_size_x', 'type': 'float', 'value': 11., },
         {'title': 'Beam size y (mm):', 'name': 'beam_size_y', 'type': 'float', 'value': 11., },

         ]

    def value_changed(self, param: Parameter):
        field = self.compute_field()
        self.notify_listeners(field)

    def compute_field(self):
        return GaussianIntensityField(
            'GaussianField',
            (self.settings['ny_pixels'], self.settings['nx_pixels']),
            (self.settings['pixel_size_y'], self.settings['pixel_size_x']),
            (self.settings['beam_size_y'], self.settings['beam_size_x']),
        )

    def load(self, *args, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        self.field = self.compute_field()
        return self.field
