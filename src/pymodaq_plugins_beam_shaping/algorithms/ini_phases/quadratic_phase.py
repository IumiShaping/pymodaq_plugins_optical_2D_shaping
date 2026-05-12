from dataclasses import dataclass

import numpy as np

from pymodaq_data import Q_

from pymodaq_plugins_beam_shaping.algorithms.ini_phase import PhaseFactory, PhaseBase
from pymodaq_plugins_beam_shaping.utils import Config
from pymodaq_plugins_beam_shaping.utilities import sizing
from pymodaq_plugins_beam_shaping.algorithms.utils import LensSetup, ApplyMaskTo

plugin_config = Config()


@dataclass
class FieldSize:
    x_size: float = None
    y_size: float = None

    def __init__(self, field: Field):
        self.x_size = my_moment(field.get_axis_from_index(1)[0].get_data(),
                                np.sum(field.amplitude, axis=0))[1]
        self.y_size = my_moment(field.get_axis_from_index(0)[0].get_data(),
                                np.sum(field.amplitude, axis=1))[1]



@PhaseFactory.register_phase()
class QuadraticPhase(PhaseBase):
    """ The phase is built from this expression: R (p**2 + q**2) + D (p cos θ + q sin θ) where p and q are
    the normalized pixel indexes

    see https://doi.org/10.1364/OE.25.011692
    """
    params = [
        {'title': 'Quadratic Amplitude', 'name': 'quadratic', 'type': 'group', 'children': [
            {'title': 'Auto: ', 'name': 'auto', 'type': 'bool', 'value': True,
             'tip': 'If True, the X and Y quadratic phase amplotude are computed/updated from the target mask size '},
            {'title': 'X Quad. ', 'name': 'x_quad', 'type': 'float', 'value': 1.},
            {'title': 'Y Quad. ', 'name': 'y_quad', 'type': 'float', 'value': 1.},
        ]},
        {'title': 'Linear Shift', 'name': 'shift', 'type': 'group', 'children': [
            {'title': 'Auto: ', 'name': 'auto', 'type': 'bool', 'value': True,
             'tip': 'If True, the X and Y shift are computed/updated from the target mask position '},
            {'title': 'X Shift (1/m) ', 'name': 'x_shift', 'type': 'float', 'value': 0.},
            {'title': 'Y shift (1/m)', 'name': 'y_shift', 'type': 'float', 'value': 0.},
        ]},
    ]

    def get_input_output_size(self) -> tuple[FieldSize, FieldSize]:

        return FieldSize(self.algo.modulator_field), FieldSize(self.algo.target_field_pixels_sizes[0])

    def compute_phase(self, **kwargs) -> np.ndarray:
        ny, nx = self.algo.shape
        xlin = np.linspace(-nx // 2, nx // 2, nx, endpoint=True)
        ylin = np.linspace(-ny // 2, ny // 2, ny, endpoint=True)

        if self.settings['quadratic', 'auto']:
            self._compute_quadratic_factor()

        xx_quad, yy_quad = np.meshgrid(self.settings['quadratic', 'x_quad'] * xlin ** 2,
                                       self.settings['quadratic', 'y_quad'] * ylin ** 2)
        phase = xx_quad + yy_quad

        if self.settings['shift', 'auto']:
            self._compute_linear_factor()  # approximated from lens computation

        xxlin, yylin = np.meshgrid(self.settings['shift', 'x_shift'] * xlin,
                                   self.settings['shift', 'y_shift'] * ylin)
        phase += xxlin + yylin
        return phase

    def focal_quad(self) -> np.ndarray:
        """ Compute focal to add in order to have all light on the size of the target """
        focal = Q_(plugin_config('setup', self.algo.SETUP_TYPE.value, 'focals')[0], 'mm')
        object_size = Q_(np.array(sizing.get_effective_slm_size()) * sizing.get_effective_slm_pixel_size(),
                         'um')
        if self.algo.apply_mask(ApplyMaskTo.TARGET):
            _slices = self.algo.get_mask_slices(ApplyMaskTo.TARGET)
            size = [(_slice.stop - _slice.start)
                     // (_slice.step if _slice.step is not None else 1) + 1 for _slice in _slices]
        else:
            size = self.algo.shape

        target_size = Q_(np.array([self.algo.target_field_pixels_sizes[ind].magnitude *
                                   size[ind] for ind in range(2)]),
                         self.algo.target_field_pixels_sizes[0].units)

        focal_quad = focal * (object_size / target_size + 1)
        return focal_quad

    def _compute_quadratic_factor(self):
        binning = plugin_config('sizing', 'binning')
        pixel_sizes = Q_(np.array([self.algo.input_field_pixels_sizes[ind].magnitude / binning for ind in range(2)]),
                         self.algo.input_field_pixels_sizes[0].units)

        wavelength = Q_(plugin_config('setup', 'wavelength_nm', ), 'nm')
        coeff = (pixel_sizes ** 2 / (wavelength * self.focal_quad()) * np.pi).to_reduced_units().magnitude
        self.settings.child('quadratic', 'x_quad').setValue(coeff[1])
        self.settings.child('quadratic', 'y_quad').setValue(coeff[0])

    def _compute_linear_factor(self):
        if self.algo.apply_mask(ApplyMaskTo.TARGET):
            _slices = self.algo.get_mask_slices(ApplyMaskTo.TARGET)
            shift_y, shift_x = tuple([(_slice.stop + _slice.start) / 2 - self.algo.shape[ind] / 2
                                      for ind, _slice in enumerate(_slices)])
        else:
            shift_y, shift_x = (0., 0.)

        pixel_SLM = Q_(sizing.get_effective_slm_pixel_size(), 'um')
        setup_type = plugin_config('setup', 'setup_type')[0]
        focal_postSLM = Q_(plugin_config('setup', setup_type, 'focals')[0], 'mm')
        wavelength = Q_(plugin_config('setup', 'wavelength_nm'), 'nm')

        coeff = (2*np.pi / (wavelength * focal_postSLM))

        coeff = ((shift_y * coeff * pixel_SLM).to_reduced_units().magnitude,
                 (shift_x * coeff * pixel_SLM).to_reduced_units().magnitude)

        self.settings.child('shift', 'x_shift').setValue(coeff[1])
        self.settings.child('shift', 'y_shift').setValue(coeff[0])