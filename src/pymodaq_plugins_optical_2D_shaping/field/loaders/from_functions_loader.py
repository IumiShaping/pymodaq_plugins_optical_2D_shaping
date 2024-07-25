from numbers import Number
from typing import Tuple, Union

import numpy as np

from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pyqtgraph.parametertree import Parameter
from pymodaq.utils import math_utils as mutils
from pymodaq import Q_

from pymodaq_plugins_optical_2D_shaping import config as plugin_config

SLM = plugin_config('SLM', 'default_slm')


@LoaderFactory.register_loader()
class GaussianIntensity(FieldLoader):

    LOADER_NAME = 'GaussianIntensityLoader'

    params = FieldLoader.params + \
        [{'title': 'Ny:', 'name': 'ny_pixels', 'type': 'int',
          'value': plugin_config('SLM', SLM, 'height'), },
         {'title': 'Nx:', 'name': 'nx_pixels', 'type': 'int',
          'value': plugin_config('SLM', SLM, 'width'), },
         {'title': 'Pixel size x (µm):', 'name': 'pixel_size_x', 'type': 'float',
          'value': plugin_config('SLM', SLM, 'pixel_size'), },
         {'title': 'Pixel size y (µm):', 'name': 'pixel_size_y', 'type': 'float',
          'value': plugin_config('SLM', SLM, 'pixel_size'), },
         {'title': 'Beam size x (mm):', 'name': 'beam_size_x', 'type': 'float', 'value': 5., },
         {'title': 'Beam size y (mm):', 'name': 'beam_size_y', 'type': 'float', 'value': 5., },
         ]

    def __init__(self, *args, **kwargs):
        super().__init__()
        if 'width' in kwargs:
            self.settings.child('nx_pixels').setValue(kwargs['width'])
        if 'height' in kwargs:
            self.settings.child('ny_pixels').setValue(kwargs['height'])
        if 'pixel_size' in kwargs:
            self.settings.child('pixel_size_x').setValue(kwargs['pixel_size'])
            self.settings.child('pixel_size_x').setValue(kwargs['pixel_size'])

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


@LoaderFactory.register_loader()
class LaguerreGauss(GaussianIntensity):

    LOADER_NAME = 'LaguerreGauss'

    params = FieldLoader.params + GaussianIntensity.params + \
             [{'title': 'Angular Momentum:', 'name': 'angular_momentum', 'type': 'int',
               'value': 3, },
              ]

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

        xx, yy = np.meshgrid(x, y)

        phase_array = np.arctan((yy - np.mean(y)).m_as('mm') /
                                (xx - np.mean(x)).m_as('mm')) * \
                      self.settings['angular_momentum']

        field = Field('GaussianIntensity', amplitude=amplitude, phase=phase_array,
                      pixel_sizes=Q_(np.array((self.settings['pixel_size_y'],
                                               self.settings['pixel_size_x'])),
                                     'um'))
        return field

