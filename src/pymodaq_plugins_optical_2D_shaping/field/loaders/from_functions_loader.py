from numbers import Number
from typing import Tuple, Union

import numpy as np

from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pyqtgraph.parametertree import Parameter
from pymodaq.utils import math_utils as mutils
from pymodaq import Q_


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

        x = Q_(np.arange(0, self.settings['nx_pixels'], 1) * self.settings['pixel_size_x'],
               'micron')
        y = Q_(np.arange(0, self.settings['ny_pixels'], 1) * self.settings['pixel_size_y'],
               'micron')

        amplitude = np.sqrt(mutils.gauss2D(
            x.m_as('mm'), np.mean(x.m_as('mm')),
            Q_(self.settings['beam_size_x'], 'mm').m_as('mm'),
            y.m_as('mm'), np.mean(y.m_as('mm')),
            Q_(self.settings['beam_size_y'], 'mm').m_as('mm')))

        field = Field('GaussianIntensity', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.settings['pixel_size_y'],
                                               self.settings['pixel_size_x'])),
                                     'um'))
        return field

    def load(self, *args, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        self.field = self.compute_field()
        return self.field
