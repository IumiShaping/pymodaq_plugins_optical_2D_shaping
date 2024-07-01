from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets

from pymodaq.utils.managers.parameter_manager import ParameterManager, Parameter
from pymodaq.utils.parameter.utils import iter_children
from pymodaq.utils.enums import BaseEnum, enum_checker
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq.utils.data import DataRaw


logger = set_logger(get_module_name(__file__))


class LoadTypeEnum(BaseEnum):

    AMPLITUDE = 'amplitude'
    PHASE = 'phase'


class Field:
    def __init__(self):
        self._amplitude: np.ndarray = None
        self._phase: np.ndarray = None

    def __repr__(self):
        return f'Field of shape {self.shape}'

    @property
    def shape(self):
        if self.amplitude is not None:
            return self.amplitude.shape
        elif self.phase is not None:
            return self.phase.shape
        else:
            raise AttributeError('No amplitude nor phase array has been defined')

    @property
    def field(self):
        """ Get set the field as a complex 2D array"""
        return self._amplitude * np.exp(1j * self._phase)

    @field.setter
    def field(self, field_array: np.ndarray):
        self._amplitude = np.abs(field_array)
        self._phase = np.angle(self.field)

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amp_array: np.ndarray):
        if self._phase is not None:
            if not self._phase.shape == self._amplitude.shape:
                logger.warning('New amplitude is not coherent with existing phase shape'
                               'Setting the phase to flat zeros')
                self._phase = np.zeros_like(amp_array)
        else:
            self._phase = np.zeros_like(amp_array)

        self._amplitude = amp_array

    def amplitude_as_dwa(self):
        return DataRaw('amplitude', data=[self.amplitude])

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase_array: np.ndarray):
        if self._amplitude is not None:
            if not self._amplitude.shape == self._phase.shape:
                logger.warning('New phase is not coherent with existing amplitude shape'
                               'Setting the amplitude to flat ones')
                self._amplitude = np.ones_like(phase_array)
        else:
            self._amplitude = np.ones_like(phase_array)

        self._phase = phase_array

    def phase_as_dwa(self):
        return DataRaw('phase', data=[self.phase])


class TargetParameterManager(ParameterManager):
    settings_name = 'target_settings'

    def __init__(self):
        super().__init__()
        self.settings_tree.header().setVisible(True)
        self.settings_tree.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.settings_tree.header().setMinimumSectionSize(150)
        self.settings_tree.setMinimumHeight(150)


class TargetLoader(TargetParameterManager, metaclass=ABCMeta):
    LOADER_NAME = abstractproperty()

    params = [
        {'title': 'Target utils', 'name': 'utils', 'type': 'group', 'children': [
            {'title': 'Show target', 'name': 'show_target', 'type': 'bool_push', 'value': False},
            {'title': 'Flip ud', 'name': 'flipud', 'type': 'bool', 'value': False},
            {'title': 'Flip lr', 'name': 'fliplr', 'type': 'bool', 'value': False},
        ],
        },
    ]

    def __init__(self):
        super().__init__()

        self.amp_viewer: Viewer2D = None
        self.phase_viewer: Viewer2D = None
        self.field = Field()

        self.setup_ui()

    def setup_ui(self):
        self.target_widget = QtWidgets.QWidget()
        self.target_widget.setLayout(QtWidgets.QHBoxLayout())

        amp_widget = QtWidgets.QWidget()
        self.amp_viewer = Viewer2D(amp_widget)

        phase_widget = QtWidgets.QWidget()
        self.phase_viewer = Viewer2D(phase_widget)

        self.target_widget.layout().addWidget(amp_widget)
        self.target_widget.layout().addWidget(phase_widget)

    def value_changed(self, param: Parameter):
        if param.name() in ('flipud', 'fliplr'):
            self.transform_image(param.name())
        elif param.name() == 'show_target':
            self.target_widget.setVisible(param.value())

    def transform_image(self, param_name: str):
        if self.field is not None:
            if param_name == 'flipud':
                self.field.amplitude = np.flipud(self.field.amplitude)
                self.field.phase = np.flipud(self.field.phase)
            elif param_name == 'fliplr':
                self.field.amplitude = np.fliplr(self.field.amplitude)
                self.field.phase = np.fliplr(self.field.phase)

            self.update_viewers()

    def update_viewers(self):
        self.amp_viewer.show_data(self.field.amplitude_as_dwa())
        self.phase_viewer.show_data(self.field.phase_as_dwa())

    def load(self, *args, load_type=LoadTypeEnum.AMPLITUDE, **kwargs):
        """ Abstract method used to load a target to populate the field attribute"""

        raise NotImplementedError
