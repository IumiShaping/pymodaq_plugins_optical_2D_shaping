from numbers import Number
from typing import Tuple, Union, TYPE_CHECKING, Callable, Iterable

import numpy as np


from pymodaq_plugins_optical_2D_shaping.field import Field, LoaderFactory, FieldLoader
from pymodaq_gui.parameter import Parameter
from pymodaq_utils import math_utils as mutils
from pymodaq_data import Q_, Unit

from scipy.special import genlaguerre

from pymodaq_plugins_optical_2D_shaping import config as plugin_config

SLM = plugin_config('SLM', 'default_slm')



class BaseFieldLoader(FieldLoader):
    with_physical_pixels_size = True

    def compute_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """ Compute the centered positions on the grid in microns given the value of the pixel width"""
        x = np.arange(0, self.n_pixel_width, 1) * self.pixel_width   #um
        x = x-np.mean(x)
        y = np.arange(0, self.n_pixel_height, 1) * self.pixel_height  #um
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    @staticmethod
    def compute_polygon(xx, yy, n_coordinates):
        """ Compute the location of the pixels within a closed polygon of n points

        The method use the fact that all points situated at the left of any polygone edge are within the polygon. The
        coordinates should be taken counter-clockwise. See https://stackoverflow.com/questions/2752725/
        """
        d = np.full(xx.shape, True)
        for ind in range(-1, len(n_coordinates) - 1):
            d = d & (((n_coordinates[ind+1][0] - n_coordinates[ind][0]) * (yy - n_coordinates[ind][1]) -
                     (xx - n_coordinates[ind][0]) * (n_coordinates[ind+1][1] - n_coordinates[ind][1])) >= 0)
        return np.where(d, 1, 0)

    def compute_circle(self, xx: np.ndarray, yy: np.ndarray,
                       x_center: float, y_center: float,
                       radius: float = 100.):
        """ Compute the location of the pixels within a circle given its center and its radius """
        xx_shifted = xx - x_center
        yy_shifted = yy - y_center

        # Calculate the radial distance from the beam center
        radial_distance = np.sqrt(xx_shifted ** 2 + yy_shifted ** 2)
        amplitude = np.where(radial_distance < radius, 1, 0)
        return amplitude

    @staticmethod
    def gaussian_fwhm(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
        """ Get a Gaussian amplitude distribution centered in x0 with a full width at half maximum in intensity of fwhm

        Parameters
        ----------
        x: np.ndarray
            distribution on which the gaussian is evaluated (in microns)
        x0: float
            center of the gaussian distribution (in micron)
        fwhm: float
            full width at half maximum in intensity expressed (in micron)

        Returns
        -------
        np.ndarray
        """
        return np.exp(- 2 * np.log(2) * ((x - x0) / fwhm) ** 2)

    def settings_changed(self, param: Parameter):
        field = self.compute_field()
        self.notify_listeners(field)

    def compute_field(self) -> Field:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        self.field = self.compute_field()
        return self.field


@LoaderFactory.register_loader()
class GaussianIntensity(BaseFieldLoader):

    LOADER_NAME = 'GaussianIntensity'

    params = BaseFieldLoader.params + \
        [
            {'title': 'Beam size x (um):', 'name': 'beam_size_x', 'type': 'float',
             'value': plugin_config('input', 'gaussian', 'fwhm_x'), 'tip' : 'FWHM in intensity'},
            {'title': 'Beam size y (um):', 'name': 'beam_size_y', 'type': 'float',
             'value': plugin_config('input', 'gaussian', 'fwhm_y'), 'tip' : 'FWHM in intensity'},
         ]

    def compute_field(self) -> Field:
        xx, yy = self.compute_grid()
        self.progressbar = 30
        amplitude = (self.gaussian_fwhm(xx, 0, self.settings['beam_size_x']) *
                     self.gaussian_fwhm(yy, 0, self.settings['beam_size_y']))
        self.progressbar = 60
        field = Field('GaussianIntensity',
                      amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.pixel_height, self.pixel_width)),'um'))
        return field


@LoaderFactory.register_loader()
class DoubleGaussian(BaseFieldLoader):
    LOADER_NAME = 'DoubleGaussian'

    params = BaseFieldLoader.params + \
             [
                 {'title': 'sigma_x (um):', 'name': 'sigma_x', 'type': 'float',
                  'value': 300., },
                 {'title': 'sigma_y (um):', 'name': 'sigma_y', 'type': 'float',
                  'value': 300., },
                 {'title': 'Position x from center (um):', 'name': 'x_from_center', 'type': 'float',
                  'value': 500., },
                 {'title': 'Add asymetry (um) :', 'name': 'asymetry', 'type': 'float',
                  'value': 0., },
                 {'title': 'Rotation around center (degree):', 'name': 'rotation_around_center', 'type': 'float',
                  'value': 0., },
             ]

    def compute_field(self) -> Field:

        xx, yy = self.compute_grid()

        self.progressbar = 15

        theta = np.radians(Q_(self.settings['rotation_around_center'], 'degree'))
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        self.progressbar = 30

        # Gaussian 1
        beam_center_unrotated = [self.settings['x_from_center'], 0.]

        self.progressbar = 45

        x1_rot, y1_rot = rotation_matrix @ beam_center_unrotated
        amplitude = (self.gaussian_fwhm(xx, x1_rot, self.settings['sigma_x']) *
                     self.gaussian_fwhm(yy, y1_rot, self.settings['sigma_y']))
        self.progressbar = 60

        # Gaussian 2
        beam_center_unrotated_magn = [
            beam_center_unrotated[0] + self.settings['asymetry'],
            beam_center_unrotated[1]]

        self.progressbar = 80
        x2_rot, y2_rot = -rotation_matrix @ beam_center_unrotated_magn
        amplitude += (self.gaussian_fwhm(xx, x2_rot, self.settings['sigma_x']) *
                     self.gaussian_fwhm(yy, y2_rot, self.settings['sigma_y']))

        field = Field('Double_gaussian', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field


@LoaderFactory.register_loader()
class LaguerreGaussian(BaseFieldLoader):
    """ Calculation based on mathematical expression from 10.61835/gd8 using scipy special Laguerre function"""

    LOADER_NAME = 'LaguerreGaussian'

    params = BaseFieldLoader.params + \
        [
            {'title': 'Beam waist (um):', 'name': 'waist', 'type': 'float',
             'value': plugin_config('input', 'laguerre', 'waist'), },
            {'title': 'Radial order :', 'name': 'radial_index', 'type': 'int',
             'value': plugin_config('input', 'laguerre', 'radial_index'), },
            {'title': 'Azimutal order:', 'name': 'azimutal_index', 'type': 'int',
             'value': plugin_config('input', 'laguerre', 'azimutal_index'), },
         ]

    def compute_polynomial(self, m: int , l: int) -> Callable[[np.ndarray], np.ndarray]:
        return genlaguerre(m, abs(l))

    def compute_laguerre(self, m: int, l: int, xx: np.ndarray, yy: np.ndarray, waist_fwhm: float):
        poly = self.compute_polynomial(m, l)
        radius = np.sqrt(xx ** 2 + yy ** 2)
        phi = np.arctan2(yy, xx)
        waist = waist_fwhm / np.sqrt(2 * np.log(2))
        return (poly(2 * radius ** 2 / waist ** 2) *
                self.gaussian_fwhm(radius, 0, waist_fwhm) *
                np.exp(1j * l * phi) *
                (np.sqrt(2) * radius / waist) ** abs(l)
                )

    def compute_field(self) -> Field:
        xx, yy = self.compute_grid()

        self.progressbar = 20

        lg_complex = self.compute_laguerre(self.settings['radial_index'],
                                           self.settings['azimutal_index'],
                                           xx, yy,
                                           self.settings['waist'])
        self.progressbar = 80

        field = Field('LaguerreGaussian',
                      amplitude=np.abs(lg_complex),
                      phase=np.angle(lg_complex),
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field


@LoaderFactory.register_loader()
class FerrisWheel(LaguerreGaussian):

    LOADER_NAME = 'FerrisWheel'

    params = BaseFieldLoader.params + \
        [
            {'title': 'Beam waist (um):', 'name': 'waist', 'type': 'float',
             'value': plugin_config('input', 'ferris', 'waist'), },
            {'title': 'Azimutal order 1:', 'name': 'azimutal_index_1', 'type': 'int',
             'value': plugin_config('input', 'ferris', 'azimutal_index_1'), },
            {'title': 'Azimutal order 2:', 'name': 'azimutal_index_2', 'type': 'int',
             'value': plugin_config('input', 'ferris', 'azimutal_index_2'), },
            {'title': 'Alpha:', 'name': 'alpha', 'type': 'float',
             'value': plugin_config('input', 'ferris', 'alpha'), },
         ]

    def compute_field(self) -> Field:
        xx, yy = self.compute_grid()
        self.progressbar = 10

        field1 = super().compute_laguerre(0, self.settings['azimutal_index_1'], xx, yy,
                                          self.settings['waist'])
        self.progressbar = 50

        field2 = super().compute_laguerre(0, self.settings['azimutal_index_2'], xx, yy,
                                          self.settings['waist'])
        self.progressbar = 80

        wheel = field1 + self.settings['alpha'] * field2

        amplitude = np.abs(wheel)
        phase = np.angle(wheel)

        self.progressbar = 100

        field = Field('FerrisWheel',
                      amplitude=amplitude,
                      phase = phase,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field


@LoaderFactory.register_loader()
class TwoCirclesOnLine(BaseFieldLoader):

    LOADER_NAME = 'TwoCirclesOnALine'

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

    def compute_field(self) -> Field:
        xx, yy = self.compute_grid()

        theta = np.radians(Q_(self.settings['rotation_around_center'], 'degree'))
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        #Circle 1
        beam_center_unrotated = np.array([self.settings['x_from_center'], 0.])

        x1_rot, y1_rot = rotation_matrix@beam_center_unrotated
        amplitude = self.compute_circle(xx, yy, x1_rot, y1_rot, self.settings['radius_circle'])

        #Circle 2
        beam_center_unrotated2 = np.array([beam_center_unrotated[0] + self.settings['asymetry'],
                                          beam_center_unrotated[1]])
        x2_rot, y2_rot = -rotation_matrix@beam_center_unrotated2
        amplitude += self.compute_circle(xx, yy, x2_rot, y2_rot, self.settings['radius_circle'])

        rectangle_coordinates = [beam_center_unrotated + np.array([0, -self.settings['linewidth'] / 2]),
                                 beam_center_unrotated + np.array([0, +self.settings['linewidth'] / 2]),
                                 beam_center_unrotated + np.array([self.settings['asymetry'], 0]) + np.array([0, -self.settings['linewidth'] / 2]),
                                 beam_center_unrotated + np.array([self.settings['asymetry'], 0]) + np.array([0, self.settings['linewidth'] / 2]),
                                 ]
        rotated_coordinates = []
        for ind, coord in enumerate(rectangle_coordinates):
            if ind < 2:
                rot = rotation_matrix
            else:
                rot = -rotation_matrix
            rotated_coordinates.append(rot@coord)

        amplitude += self.compute_polygon(xx, yy, rotated_coordinates)

        amplitude[amplitude > 0] = 1

        field = Field('2circles_on_line', amplitude=amplitude,
                      pixel_sizes=Q_(np.array((self.pixel_height,
                                               self.pixel_width)),
                                     'um'))
        return field


@LoaderFactory.register_loader()
class RectangleIntensity(BaseFieldLoader):
    LOADER_NAME = 'RectangleIntensity'

    params = BaseFieldLoader.params + [
        {'title': 'Width (um):', 'name': 'length', 'type': 'float', 'value': 1000., 'suffix': 'um'},
        {'title': 'Height (um):', 'name': 'height', 'type': 'float', 'value': 1000., 'suffix': 'um'},
        {'title': 'Edge Thickness (um):', 'name': 'thickness', 'type': 'float', 'value': 200., 'suffix': 'um'},
    ]

    def compute_field(self) -> Field:

        xx, yy = self.compute_grid()
        self.progressbar = 30

        # Rectangle parameters

        inner_coordinates = [[self.settings['length'] /2 - self.settings['thickness'] /2 , - self.settings['height'] /2  + self.settings['thickness'] /2],
                             [self.settings['length'] /2 - self.settings['thickness'] /2 , + self.settings['height'] /2  - self.settings['thickness'] /2],
                             [-self.settings['length'] /2 + self.settings['thickness'] /2 , + self.settings['height'] /2 - self.settings['thickness'] /2],
                             [-self.settings['length'] /2 + self.settings['thickness'] /2 , - self.settings['height'] /2  + self.settings['thickness'] /2],]
        outer_coordinates = [[self.settings['length'] /2 , - self.settings['height'] /2 ],
                             [self.settings['length'] /2 , + self.settings['height'] /2 ],
                             [-self.settings['length'] /2 , + self.settings['height'] /2 ],
                             [-self.settings['length'] /2 , - self.settings['height'] /2 ],]

        amplitude = self.compute_polygon(xx, yy, outer_coordinates)
        amplitude -= self.compute_polygon(xx, yy, inner_coordinates)
        self.progressbar = 90

        field = Field(
            'RectangleIntensity',
            amplitude=amplitude,
            pixel_sizes=Q_(np.array((self.pixel_height, self.pixel_width)), 'um')
        )
        return field
