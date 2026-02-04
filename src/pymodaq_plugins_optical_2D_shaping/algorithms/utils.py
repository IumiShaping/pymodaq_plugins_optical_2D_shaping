from abc import ABCMeta
from typing import Tuple, TYPE_CHECKING, Union, Optional

import numpy as np
from qtpy import QtWidgets
from scipy.ndimage import gaussian_filter

from pymodaq_plugins_optical_2D_shaping.utilities.masking import MaskType
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataRaw, DataToExport
from pymodaq_utils.enums import StrEnum

from pymodaq_gui.managers.parameter_manager import ParameterManager, Parameter

from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.field import Field, Q_

if TYPE_CHECKING:
    from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

logger = set_logger(get_module_name(__file__))
plugin_config = PluginConfig()


class MaskError(Exception):
    pass


class TargetPhase(StrEnum):
    RANDOM = 'random'
    QUADRATIC = 'quadratic'  # see https://doi.org/10.1364/OE.25.014323
    QUADRATIC_SHIFT = 'quadratic_shift'  # see https://doi.org/10.1364/OE.25.011692


class LensSetup(StrEnum):
    NoLens = 'no_lens'
    TwoF = '2f'
    FourF = '4f'


class ApplyMaskTo(StrEnum):
    TARGET = 'target_mask'
    INTERMEDIATE = 'intermediate_mask'


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

    ALGO_NAME: str = None  # to be reimplemented
    SETUP_TYPE: LensSetup = None # to be reimplemented
    ITERATIVE = False
    MANUAL_LOOP = True
    params = []


    def __init__(self, parent: 'AlgoApp' = None):
        super().__init__()

        self._running = False

        self.parent_app = parent
        self._target_field = Field()
        self._input_field = Field()
        self.intermediate_field: Optional[Field] = None

        self._object_field = Field(amplitude=self._input_field.amplitude.copy())
        self._object_field.calibrate_axes(self._input_field.pixels_sizes)
        self._image_field = Field()

        self.update_mask = True


    def quit(self):
        """ to reimplement if necessary"""
        pass

    def apply_mask(self, apply_to: Union[ApplyMaskTo, str]) -> bool:
        return self.parent_app.apply_mask(apply_to)

    def get_mask_slices(self, apply_to: Union[ApplyMaskTo, str]) -> tuple[slice, slice]:
        """ Get the slices defined in the settings depending on the field it should apply to"""
        return self.parent_app.get_mask_as_slices(apply_to)

    def get_mask_field(self, apply_to: Union[ApplyMaskTo, str], inner_value=1, outer_value=0) -> Field:
        """ Return a Field to be used to mask fields within the algorithm

        To be reimplemented if needed
        """
        if self.apply_mask(apply_to):
            mask = Field.init_from_field(self._target_field).amplitude * outer_value
            slices = self.get_mask_slices(apply_to)
            if self.get_mask_type(apply_to) == MaskType.SQUARE:
                mask[*slices] = inner_value
            else:

                y0, x0 = tuple([(_slice.stop + _slice.start) / 2 for _slice in slices])
                ry, rx = tuple([(_slice.stop - _slice.start) / 2 for _slice in slices])

                x = np.arange(0, self._target_field.shape[1], 1)
                y = np.arange(0, self._target_field.shape[0], 1)

                xx, yy = np.meshgrid(x, y)
                mask[
                    (xx - x0) ** 2 / rx **2 + (yy - y0) ** 2 / ry **2 <= 1] = inner_value
        else:
            mask = Field.init_from_field(self._target_field).amplitude
        return Field(amplitude=mask)

    def get_mask_type(self, apply_to: Union[ApplyMaskTo, str]) -> MaskType:
        return self.parent_app.get_mask_type(apply_to)

    @property
    def image_field(self) -> Field:
        return self._image_field

    @property
    def object_field(self) -> Field:
        return self._object_field

    def set_input_field(self, field: Field):
        self._input_field = field
        self.do_things_after_set_input()

    def get_phase_type(self) -> TargetPhase:
        return TargetPhase(self.parent_app.settings['target_phase_group', 'target_phase'])


    def focal_quad(self) -> np.ndarray:
        """ Compute focal to add in order to have all light on the size of the target """
        focal = Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0], 'mm')
        object_size = Q_(np.array([self.object_field.pixels_sizes[ind].magnitude *
                                   self.object_field.shape[ind] for ind in range(2)]),
                         self.object_field.pixels_sizes[0].units)
        if self.apply_mask(ApplyMaskTo.TARGET):
            _slices = self.get_mask_slices(ApplyMaskTo.TARGET)
            size = [(_slice.stop - _slice.start)
                     // (_slice.step if _slice.step is not None else 1) + 1 for _slice in _slices]
        else:
            size = self._target_field.shape

        target_size = Q_(np.array([self._target_field.pixels_sizes[ind].magnitude *
                                   size[ind] for ind in range(2)]),
                         self._target_field.pixels_sizes[0].units)

        focal_quad = focal * (object_size / target_size + 1)
        return focal_quad

    def _compute_quadratic_factor(self):
        pixel_sizes = Q_(np.array([self.object_field.pixels_sizes[ind].magnitude for ind in range(2)]),
                         self.object_field.pixels_sizes[0].units)

        wavelength = Q_(plugin_config('setup', 'wavelength_nm', ), 'nm')
        return pixel_sizes ** 2 / (wavelength * self.focal_quad()) * np.pi

    def _compute_linear_factor(self):
        if self.apply_mask(ApplyMaskTo.TARGET):
            _slices = self.get_mask_slices(ApplyMaskTo.TARGET)
            shift_y, shift_x = tuple([(_slice.stop + _slice.start) / 2 - self.object_field.shape[ind] / 2
                                      for ind, _slice in enumerate(_slices)])
        else:
            shift_y, shift_x = (0., 0.)

        pixel_SLM = Q_(plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'pixel_size'), 'um')
        setup_type = plugin_config('setup', 'setup_type')[0]
        focal_postSLM = Q_(plugin_config('setup', setup_type, 'focals')[0], 'mm')
        wavelength = Q_(plugin_config('setup', 'wavelength_nm'), 'nm')

        coeff = (2*np.pi / (wavelength * focal_postSLM))

        return  ((shift_y * coeff * pixel_SLM).to_reduced_units().magnitude,
                 (shift_x * coeff * pixel_SLM).to_reduced_units().magnitude)

    def define_input_phase(self, phase_type: TargetPhase = None, force_reset = False) -> np.ndarray:
        if not force_reset and self.parent_app.current_phase is not None:
            phase = self.parent_app.current_phase
        else:

            if phase_type is None:
                phase_type = self.get_phase_type()
            shape = self._object_field.shape
            if phase_type == TargetPhase.RANDOM:
                phase = np.random.random_sample(shape) * 2 *np.pi

            elif phase_type == TargetPhase.QUADRATIC or phase_type == TargetPhase.QUADRATIC_SHIFT:
                ny, nx = shape
                xlin = np.linspace(-nx//2, nx//2 , nx , endpoint=True)
                ylin = np.linspace(-ny//2, ny//2 , ny , endpoint=True )

                r = ((self._compute_quadratic_factor().to_reduced_units().magnitude *  # approximated from two lens computation
                     self.parent_app.settings['target_phase_group', 'params', 'quad_amp'])  # manual coefficient to move the shift
                     * 1.75)  # adhoc coefficient to match target size
                xx_quad, yy_quad = np.meshgrid( r[1] * xlin ** 2,
                                                r[0] * ylin ** 2)
                phase = xx_quad + yy_quad

                if phase_type == TargetPhase.QUADRATIC_SHIFT:
                    coeff = self._compute_linear_factor()  # approximated from lens computation
                    d = (self.parent_app.settings['target_phase_group', 'params', 'shift_amp']  # manual coefficient to move the shift
                         * 3) # adhoc coefficient to correctly match target roi position
                    xxlin, yylin = np.meshgrid(d * coeff[1] * xlin,
                                               d * coeff[0] * ylin)
                    phase += xxlin + yylin

            else:
                raise ValueError('Unknown phase type')

        phase = (phase + np.pi) % (2 * np.pi) - np.pi
        self.set_phase_in_object_plane(phase)
        return phase

    def compute_forward_fft(self, update_plots = True):
        self._image_field = self.normalize_wrt(self._object_field.fft2(norm='forward'),
                                               self.object_field)
        self._image_field = self.scale_target_with_geometry(self._image_field)

        if update_plots and self.parent_app is not None:
            self.parent_app.fields_to_plot.emit(self.get_fields_to_plot())

    @staticmethod
    def normalize_wrt(field: Field, ref_field: Field) -> Field:
        """ Make sure the intensities are normalized """
        field.amplitude  = field.amplitude * np.sqrt(np.sum(np.abs(ref_field.amplitude) ** 2) /
                                                     np.sum(np.abs(field.amplitude) ** 2))
        return field

    def stop(self):
        self._running = False

    def start(self):
        self._running = True

    def set_object_field(self, field: Field):
        self._object_field = field
        self.do_things_after_set_object()

    def set_target_field(self, field: Field):
        self._target_field = self.normalize_wrt(field, self._input_field)
        self._image_field = Field.init_from_field(self._target_field)
        self.do_things_after_set_target()

    def set_target_intensity(self, intensity: np.ndarray):
        self._target_field.amplitude = np.sqrt(intensity)

    def set_phase_in_object_plane(self, phase: np.ndarray, induced_amplitude: np.ndarray = None):
        if phase.shape == self._object_field.shape:
            phase = phase.copy()
            if self.parent_app.settings['smoothing', 'apply_smoothing']:
                phase = gaussian_filter(phase, sigma=(
                    self.parent_app.settings['smoothing', 'sigma_y'],
                    self.parent_app.settings['smoothing', 'sigma_x']
                ))

            self._object_field.phase = phase
            self._object_field.amplitude = (
                    self._input_field.amplitude.copy() *
                    (induced_amplitude if induced_amplitude is not None else 1))
        else:
            raise ValueError('The phase shape is incoherent with the parameters')

    @property
    def intensity_image(self):
        return self._image_field.intensity

    @property
    def fitness(self) -> float:
        """ Compute fitness with respect to the image_field and target_field

        To be subclassed if the given implementation below is not correct for your algorithm"""

        return 100 * np.sum(
            np.abs(np.sqrt(self._target_field.intensity) - self._image_field.intensity)) ** 2 \
            / np.prod(self._image_field.shape) / np.sum(self._target_field.intensity)

    @property
    def efficiency(self) -> float:
        """ Compute efficiency as the ratio between image_field intensity within a given region
        and total intensity

        Meaningfully only for algorithm using a Target defined mask

        To be subclassed if the given implementation below is not correct for your algorithm"""

        return (np.sum(np.abs(self._image_field.intensity *
                             self.get_mask_field(ApplyMaskTo.TARGET).amplitude)) /
                np.sum(np.abs(self._image_field.intensity)))

    def fitness_as_dwa(self):
        return DataRaw('fitness', data=[np.array([self.fitness])])

    def metrics_as_dwa(self):
        return DataRaw('metrics', data=[np.array([self.fitness]),
                                        np.array([self.efficiency])],
                       labels=['fitness', 'efficiency'])

    def compute_phase(self, do_step=True, ini_phase: np.ndarray = None, **kwargs):
        """ Compute the phase to apply to SLM given the target object

        To be subclassed in real implementation
        """

        raise NotImplementedError

    def get_fields_to_plot(self) -> DataToExport:
        dte =  DataToExport('AlgoData', data=[
            self.image_field.amplitude_as_dwa('image'),
            self.image_field.phase_as_dwa('image'),
            self.metrics_as_dwa(),
            self.object_field.amplitude_as_dwa('object'),
            self.object_field.phase_as_dwa('object'),
        ])
        if self.intermediate_field is not None:
            dte.append(self.intermediate_field.intensity_as_dwa('intermediate'))
        return dte

    def do_things_after_init(self):
        """ to reimplement if needed"""
        pass

    def do_things_after_set_input(self):
        """ to reimplement if needed"""
        pass

    def do_things_after_set_target(self):
        """ to reimplement if needed"""
        pass

    def do_things_after_set_object(self):
        """ to reimplement if needed"""
        pass


    def scale_target_with_geometry(self, field: Field):
        """ Apply an axis scaling to have the target and its axes in correct units with respect to
        a given algorithm implementation and experimental setup
        """
        field.calibrate_axes(self.get_target_pixels_size())
        return field

    def get_target_pixels_size(self, input_size: Tuple[Q_, Q_] = None) -> list[Q_]:
        """ Get the expected physical size of the pixels in the target plane given
        the chosen algorithm and physical parameters: focal length, wavelength...

        Here we use either a 4f setup or a 2f setup, so the image field has the same size as the SLM with a ratio given by the focal
        length ratio

        """
        if input_size is None:
            input_size = [self._input_field.shape[ind] * self._input_field.pixels_sizes[ind]
                          for ind in range(2)]

        if self.SETUP_TYPE == LensSetup.TwoF:
            return [Q_(plugin_config('setup', 'wavelength_nm', ), 'nm') *
                    Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0], 'mm') /
                    size for size in input_size]

        elif self.SETUP_TYPE == LensSetup.FourF:
            pixels_size = self._input_field.pixels_sizes
            focal_ratio = (plugin_config('setup', self.SETUP_TYPE.value, 'focals')[1] /
                           plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0])
            return [size * focal_ratio for size in pixels_size]

        else:
            raise NotImplementedError('The setup type is not recognized and cannot compute the pixels size')

    @property
    def intermediate_pixel_sizes(self):
        if self.SETUP_TYPE == LensSetup.FourF:
            slm_pixel_sizes = self._input_field.pixels_sizes
            intermediate_pixel_sizes = [((Q_(plugin_config('setup', 'wavelength_nm',), 'nm') *
                                         Q_(plugin_config('setup', self.SETUP_TYPE.value, 'focals')[0], 'mm')) /
                                        (slm_pixel_sizes[ind] * self._target_field.shape[ind])
                                        ).to('um') for ind in range(2)]

            return intermediate_pixel_sizes
        else:
            raise ValueError('Intermediate pixel size can only be computed for 4f setups')

    def get_npad_between_image_object(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ Get the padding necessary to match object shape and image shape

        If positive, the image shape is bigger than the object
        If negative, the object shape is bigger than the image
        """
        npad_before = ((np.array(self._image_field.shape) -
                        np.array(self._object_field.shape)) // 2).astype(int)
        npad_after = (np.array(self._image_field.shape) -
                        np.array(self._object_field.shape)) - npad_before
        return (npad_before[0], npad_after[0]), (npad_before[1], npad_after[1])

    def value_changed(self, param: Parameter):
        self.parent_app.algo_settings_changed()