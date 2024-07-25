from numbers import Number
from typing import Tuple, Union

import numpy as np

from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pyqtgraph.parametertree import Parameter
from pymodaq.utils import math_utils as mutils
from pymodaq import Q_, Unit

from pymodaq_plugins_optical_2D_shaping import config as plugin_config

SLM = plugin_config('SLM', 'default_slm')


class BaseFieldLoader(FieldLoader):
    params = FieldLoader.params + \
             [{'title': 'Ny:', 'name': 'ny_pixels', 'type': 'int',
               'value': plugin_config('SLM', SLM, 'height'), },
              {'title': 'Nx:', 'name': 'nx_pixels', 'type': 'int',
               'value': plugin_config('SLM', SLM, 'width'), },
              {'title': 'Pixel size x (µm):', 'name': 'pixel_size_x', 'type': 'float',
               'value': plugin_config('SLM', SLM, 'pixel_size'), },
              {'title': 'Pixel size y (µm):', 'name': 'pixel_size_y', 'type': 'float',
               'value': plugin_config('SLM', SLM, 'pixel_size'), },]

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
        raise NotImplementedError

    def load(self, *args, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        self.field = self.compute_field()
        return self.field


@LoaderFactory.register_loader()
class GaussianIntensity(BaseFieldLoader):

    LOADER_NAME = 'GaussianIntensityLoader'

    params = BaseFieldLoader.params + \
        [
            {'title': 'Beam size x (mm):', 'name': 'beam_size_x', 'type': 'float', 'value': 5., },
            {'title': 'Beam size y (mm):', 'name': 'beam_size_y', 'type': 'float', 'value': 5., },
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

        field = Field('GaussianIntensity', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.settings['pixel_size_y'],
                                               self.settings['pixel_size_x'])),
                                     'um'))
        return field


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


@LoaderFactory.register_loader()
class SinusRectangle(BaseFieldLoader):

    LOADER_NAME = 'SinusRectangle'

    params = BaseFieldLoader.params + \
             [
                 {'title': 'Rect width (um):', 'name': 'rect_width', 'type': 'float',
                  'value': 500., },
                 {'title': 'Rect height (um):', 'name': 'rect_height', 'type': 'float',
                  'value': 300., },
                 {'title': 'Sinus period (um):', 'name': 'sinus_period', 'type': 'float',
                  'value': 250., },
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

    def compute_field(self):

        x = Q_(np.arange(0, self.settings['nx_pixels'], 1) * self.settings['pixel_size_x'],
               'micron')
        y = Q_(np.arange(0, self.settings['ny_pixels'], 1) * self.settings['pixel_size_y'],
               'micron')

        xx, yy = np.meshgrid(x, y)
        amplitude = np.sin(2 * np.pi * xx / Q_(self.settings['sinus_period'], 'um'))
        amplitude[np.mean(x) - Q_(self.settings['rect_width'], 'um') / 2 > xx] = 0
        amplitude[np.mean(x) + Q_(self.settings['rect_width'], 'um') / 2 <= xx] = 0
        amplitude[np.mean(y) - Q_(self.settings['rect_height'], 'um') / 2 > yy] = 0
        amplitude[np.mean(y) + Q_(self.settings['rect_height'], 'um') / 2 <= yy] = 0

        field = Field('SinusRectangle', amplitude=amplitude.magnitude,
                      pixel_sizes=Q_(np.array((self.settings['pixel_size_y'],
                                               self.settings['pixel_size_x'])),
                                     'um'))
        return field

