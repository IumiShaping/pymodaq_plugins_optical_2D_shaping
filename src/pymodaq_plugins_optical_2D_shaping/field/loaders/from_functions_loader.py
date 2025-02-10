from numbers import Number
from typing import Tuple, Union

import numpy as np

from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pymodaq_gui.parameter import Parameter
from pymodaq_utils import math_utils as mutils
from pymodaq_data import Q_, Unit

from pymodaq_plugins_optical_2D_shaping import config as plugin_config

SLM = plugin_config('SLM', 'default_slm')


class BaseFieldLoader(FieldLoader):
    with_physical_pixels_size = True

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

        x = Q_(np.arange(0, self.n_pixel_width, 1) * self.pixel_width,
               'micron')
        y = Q_(np.arange(0, self.n_pixel_height, 1) * self.pixel_height,
               'micron')

        amplitude = np.sqrt(mutils.gauss2D(
            x.m_as('mm'), np.mean(x.m_as('mm')),
            Q_(self.settings['beam_size_x'], 'mm').m_as('mm'),
            y.m_as('mm'), np.mean(y.m_as('mm')),
            Q_(self.settings['beam_size_y'], 'mm').m_as('mm')))

        field = Field('GaussianIntensity', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
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

        x = Q_(np.arange(0, self.n_pixel_width, 1) * self.pixel_width,
               'micron')
        y = Q_(np.arange(0, self.n_pixel_height, 1) * self.pixel_height,
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
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
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

    def compute_field(self):

        x = Q_(np.arange(0, self.n_pixel_width, 1) * self.pixel_width,
               'micron')
        y = Q_(np.arange(0, self.n_pixel_height, 1) * self.pixel_height,
               'micron')

        xx, yy = np.meshgrid(x, y)
        amplitude = np.sin(2 * np.pi * xx / Q_(self.settings['sinus_period'], 'um'))
        amplitude[np.mean(x) - Q_(self.settings['rect_width'], 'um') / 2 > xx] = 0
        amplitude[np.mean(x) + Q_(self.settings['rect_width'], 'um') / 2 <= xx] = 0
        amplitude[np.mean(y) - Q_(self.settings['rect_height'], 'um') / 2 > yy] = 0
        amplitude[np.mean(y) + Q_(self.settings['rect_height'], 'um') / 2 <= yy] = 0

        field = Field('SinusRectangle', amplitude=amplitude.magnitude,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field



@LoaderFactory.register_loader()
class TwoCirclesOnLine(BaseFieldLoader):

    LOADER_NAME = 'Two Circles on a Line'

    params = BaseFieldLoader.params + \
             [
                 {'title': 'Radius (um):', 'name': 'radius_circle', 'type': 'float',
                  'value': 300., },
                 {'title': 'Position x from center (um):', 'name': 'x_from_center', 'type': 'float',
                  'value': 500., },
                  {'title': 'Add asymetry (um) :', 'name': 'asymetry', 'type': 'float',
                  'value': 0.,},
                 {'title': 'Rotation around center (degree):', 'name': 'rotation_around_center', 'type': 'float',
                  'value': 0., },
                {'title': 'linewidth (um):', 'name': 'linewidth', 'type': 'float',
                  'value': 100., },
        ]


    def circle(self, x_tab: np.ndarray, y_tab: np.ndarray, x_center: float, y_center: float):

        xx, yy = np.meshgrid(x_tab, y_tab)
        beam_center_rotated = [Q_(x_center, 'um'), Q_(y_center, 'um')]
        X_shifted = xx - beam_center_rotated[0]
        Y_shifted = yy - beam_center_rotated[1]

        # Calculate the radial distance from the beam center
        radial_distance = Q_(np.sqrt(X_shifted**2 + Y_shifted**2), 'um')
        amplitude = np.where(radial_distance < Q_(self.settings['radius_circle'], 'um'), 255, 0) 
        
        return amplitude

    def compute_field(self):
        x = Q_(np.arange(-self.n_pixel_width/2, self.n_pixel_width/2, 1) * self.pixel_width,
               'micron')
        y = Q_(np.arange(-self.n_pixel_height/2, self.n_pixel_height/2, 1) * self.pixel_height,
               'micron')

        theta = np.radians(Q_(self.settings['rotation_around_center'], 'degree'))
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        #Circle 1
        beam_center_unrotated = [Q_(self.settings['x_from_center'], 'um'), Q_(0, 'um')]
        beam_center_unrotated_magn = [beam_center_unrotated[0].magnitude, beam_center_unrotated[1].magnitude]
        
        x1_rot, y1_rot = R@beam_center_unrotated_magn
        amplitude = self.circle(x, y, x1_rot, y1_rot)

        #Circle 2
        beam_center_unrotated_magn = [beam_center_unrotated[0].magnitude + Q_(self.settings['asymetry'], 'um').magnitude, beam_center_unrotated[1].magnitude]   
        x2_rot, y2_rot = -R@beam_center_unrotated_magn
        amplitude += self.circle(x, y, x2_rot, y2_rot)

        #add line of thickness l between circles
        x1_idx, y1_idx = int(x[len(x)-1].magnitude + x1_rot), int(y[len(y)-1].magnitude + y1_rot)
        x2_idx, y2_idx = int(x[len(x)-1].magnitude + x2_rot), int(y[len(y)-1].magnitude + y2_rot)
        num_points = int(max(abs(x2_idx - x1_idx)/self.pixel_width, abs(y2_idx - y1_idx)/self.pixel_height))
        

        if Q_(self.settings['linewidth'], 'um') < Q_(self.pixel_height, 'um'):
            #Check if the linewidth is smaller than the pixel size
            # If so, the linewidth is set to be 0
            self.settings['linewidth'] = 0

        
        #Convert the linewidth (which is in um) in number of pixel
        l = int(self.settings['linewidth']/self.pixel_height)

        t_values = np.linspace(0, 1, num_points)
        x_line = ((x1_idx * (1 - t_values) + x2_idx * t_values) / self.pixel_width).astype(int)
        y_line = ((y1_idx * (1 - t_values) + y2_idx * t_values) / self.pixel_height).astype(int)
        dx_range = np.arange(-l // 2, l // 2 + 1)
        dy_range = np.arange(-l // 2, l // 2 )

        for x, y in zip(x_line, y_line):
            x_offsets, y_offsets = np.meshgrid(dx_range, dy_range, indexing='ij')
            x_offsets = (x + x_offsets).ravel()  # Flatten for valid indexing
            y_offsets = (y + y_offsets).ravel()
            
            mask = (0 <= y_offsets) & (y_offsets < self.n_pixel_height) & (0 <= x_offsets) & (x_offsets < self.n_pixel_width)
            amplitude[y_offsets[mask], x_offsets[mask]] = 255

        field = Field('2circles_on_line', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field


