from abc import ABCMeta, abstractproperty
from typing import Union
import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_data import DataCalculated
from pymodaq_gui.plotting.utils.plot_utils import RoiInfo
from pymodaq_utils.utils import ThreadCommand


from pymodaq_data.data import DataRaw, DataToExport
from pymodaq_data.h5modules.data_saving import DataToExportSaver
from pymodaq_data.h5modules.saving import SaveType

from pymodaq_gui.managers.parameter_manager import Parameter
from pymodaq_gui.utils.custom_app import CustomApp
from pymodaq_utils.enums import StrEnum
from pymodaq_gui.utils.dock import DockArea, Dock
from pymodaq_gui.utils import QLED
from pymodaq_gui.utils.file_io import select_file
from pymodaq_gui.parameter import ioxml
from pymodaq_gui.parameter import utils as putils

from pymodaq_plugins_optical_2D_shaping.algorithms import AlgorithmFactory, AlgoBase
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import TargetPhase, ApplyMaskTo, MaskType, LensSetup
from pymodaq_plugins_optical_2D_shaping.field import Field
from pymodaq_plugins_optical_2D_shaping import config as plugin_config


algo_factory = AlgorithmFactory()


class Actions(StrEnum):

    STEP = 'step'
    CONTINUOUS = 'continuous'
    STOP = 'stop'


class AlgoApp(CustomApp):
    params = [{'title': 'Target Phase', 'name': 'target_phase_group', 'type': 'group',
         'children': [
             {'title': 'Target Phase', 'name': 'target_phase', 'type': 'list',
              'limits': TargetPhase.values(), 'value': str(TargetPhase.QUADRATIC_SHIFT),
              },
             {'title': 'Target Phase', 'name': 'params', 'type': 'group',
              'tip': 'The phase is built from this expression: R (p**2 + q**2) + D (p cos θ + q sin θ) where p and q are'
                     'the normalized pixel indexes',
              'children': [
                  {'title': 'Quadratic Amplitude ', 'name': 'quad_amp', 'type': 'float', 'value': 1.},
                  {'title': 'Shift Amplitude', 'name': 'shift_amp', 'type': 'float', 'value': 1.},
              ]},
         ]},
        {'title': 'Target Masking', 'name': str(ApplyMaskTo.TARGET), 'type': 'group', 'children': [
            {'title': 'Apply Mask', 'name': 'apply_mask', 'type': 'bool', 'value': False},
            {'title': 'Mask Type', 'name': 'mask_type', 'type': 'list', 'value': str(MaskType.SQUARE),
             'limits': MaskType.names()},
            {'title': 'Slices', 'name': 'slices', 'type': 'str',
             'value': '(slice(162, 882, None), slice(545, 1825, None))'},
        ]},
        {'title': 'Intermediate Masking', 'name': str(ApplyMaskTo.INTERMEDIATE), 'type': 'group',
         'visible': False, 'children': [
            {'title': 'Apply Mask', 'name': 'apply_mask', 'type': 'bool', 'value': False},
            {'title': 'Mask Type', 'name': 'mask_type', 'type': 'list', 'value': str(MaskType.ELLIPTICAL),
             'limits': MaskType.names()},
            {'title': 'Slices', 'name': 'slices', 'type': 'str',
             'value': '(slice(162, 882, None), slice(545, 1825, None))'},
        ]}
    ]

    command_runner = QtCore.Signal(ThreadCommand)
    object_field_signal = QtCore.Signal(Field)
    algo_changed = QtCore.Signal(AlgoBase)
    fields_to_plot = QtCore.Signal(DataToExport)

    def __init__(self, dockarea, toolbar: QtWidgets.QToolBar=None):
        super().__init__(dockarea)
        self.runner_thread: QtCore.QThread = None
        if toolbar is not None:
            self.set_toolbar(toolbar)

        self._algorithm: AlgoBase = None
        self._target_field: Field = None
        self._input_field: Field = None

        self._current_data: DataToExport = None
        self._current_phase: np.ndarray = None  # cached phase to be used for subsequent optimizations

        self.setup_ui()

        self.enable_things(False)

    @property
    def current_phase(self) -> np.ndarray:
        """ cached phase to be used for subsequent optimizations """
        return self._current_phase

    @property
    def algorithm_combo(self) -> QtWidgets.QComboBox:
        return self.get_action('algorithms')

    @property
    def algorithm(self):
        if self._algorithm is None:
            self.set_algorithm()
        return self._algorithm

    def set_target_field(self, field: Field):
        if self._algorithm is not None:
            field = self._algorithm.scale_target_with_geometry(field)
            self._algorithm.set_target_field(field)

        self._target_field = field

    def set_input_field(self, field: Field):

        object_field = field.deepcopy()

        if self._algorithm is not None:
            self._algorithm.set_object_field(object_field)
            self._algorithm.set_input_field(field)
            self._algorithm.define_input_phase(self.settings['target_phase_group', 'target_phase'])

        self._input_field = field

    def set_algorithm(self, algo_name: str = None):
        if algo_name is None:
            algo_name = self.algorithm_name
        try:
            if self._algorithm is not None:
                self._algorithm.quit()
                QtWidgets.QApplication.processEvents()

            self._algorithm: AlgoBase = \
                algo_factory.get_algorithm(algo_name)(self)

            #change the chosen setup type (defined by the algo) in the config, to be used elsewhere
            setup_types: list[str] = plugin_config['setup', 'setup_type']
            setup_types.remove(self._algorithm.SETUP_TYPE.value)
            plugin_config['setup', 'setup_type'] = [self._algorithm.SETUP_TYPE.value] + setup_types
            plugin_config.save()

            self.settings.child(str(ApplyMaskTo.INTERMEDIATE)).setOpts(
                visible=self._algorithm.SETUP_TYPE == LensSetup.FourF)

            while True:
                child = self._algo_settings_widget.layout().takeAt(0)
                if not child:
                    break
                child.widget().deleteLater()
                QtWidgets.QApplication.processEvents()

            self._algo_settings_widget.layout().addWidget(self._algorithm.settings_tree)

            if self._input_field is not None:
                self.set_input_field(self._input_field)  # defines it first as the target axes depends
                # on the input beam size and resolution
                self.set_target_field(self._target_field)
            self.set_action_visible(Actions.CONTINUOUS, self._algorithm.ITERATIVE)

            self.algo_changed.emit(self._algorithm)

        except ValueError as e:
            self.enable_things(False)

    def ini_algo(self):
        #self.set_action_enabled(Actions.CONTINUOUS, False)

        if self.is_action_checked('ini_algo'):
            self.get_action('algorithms').widget.setEnabled(False)
            self.get_action('algo_led').set_as_true()
            #self.set_action_enabled('ini_algo', False)
            #self.set_algorithm()

            self.runner_thread = QtCore.QThread()
            runner = AlgoRunner(self._algorithm)

            self.runner_thread.runner = runner
            runner.algo_output_signal.connect(self.process_output)
            runner.algo_stopped_signal.connect(self._algo_stopped)
            self.command_runner.connect(runner.queue_command)

            runner.moveToThread(self.runner_thread)

            self.runner_thread.start()

            self.compute_fft(update_plots=True)

            self.enable_things()

        else:
            self.get_action('algorithms').widget.setEnabled(True)
            if self.runner_thread is not None:
                self.get_action('algo_led').set_as_false()
                self.command_runner.disconnect()
                if self.runner_thread.isRunning():
                    self.runner_thread.terminate()
                    while not self.runner_thread.isFinished():
                        QtCore.QThread.msleep(100)
                    self.runner_thread = None
            self.enable_things(enable=False)

    def _algo_stopped(self):
        if self.is_action_checked(Actions.CONTINUOUS):
            self.get_action(Actions.CONTINUOUS).trigger()

    def setup_docks(self):

        self.algo_area = self.dockarea

        self.docks['algo_settings'] = Dock('Algorithm Settings')


        self.settings_widget = QtWidgets.QWidget()
        self.settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self.settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.settings_widget.layout().addWidget(splitter)
        splitter.addWidget(self.settings_tree)

        self._algo_settings_widget = QtWidgets.QWidget()
        self._algo_settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self._algo_settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._algo_settings_widget)

        self.settings_tree.setMinimumWidth(300)
        self.settings_tree.setMinimumHeight(150)

        self.docks['algo_settings'].addWidget(self.settings_widget)

    def setup_actions(self):
        self.add_widget('algorithms', QtWidgets.QComboBox,
                        tip='select the algorithm to compute the phase')
        self.get_action('algorithms').addItems(plugin_config('algo', 'default_algo'))
        self.get_action('algorithms').setCurrentText(plugin_config('algo', 'default_algo')[0])
        self.add_action('ini_algo', 'Init Algo', 'ini', checkable=True)
        self.add_widget('algo_led', QLED)

        self.add_action('reset_phase', 'Reset Phase', 'Refresh2', tip="Reset the SLM phase")
        self.add_action('compute_fft', 'Compute FFT', 'FFT', tip="Run a fft of the input phase")

        self.add_action(Actions.STEP, 'Step', 'snap', tip="Step a loop of the algorithm")
        self.add_action(Actions.CONTINUOUS, 'Continuous', 'run2', tip="Run continuously the algorithm",
                        checkable=True, icon_checked='stop')

        self.add_action('export', 'Export', 'SaveAs', 'Export data')

    def connect_things(self):
        self.connect_action(Actions.STEP, self.compute_phase)
        self.connect_action('compute_fft', lambda: self.compute_fft(update_plots=True))
        self.connect_action(Actions.CONTINUOUS, self.compute_phase_loop)
        self.connect_action('ini_algo', self.ini_algo)
        self.connect_action('export', self.export_data)
        self.connect_action('reset_phase', lambda: self.define_phase(force_reset=True))
        self.connect_action('algorithms', slot=self.set_algorithm,
                            signal_name='currentTextChanged')

    def enable_things(self, enable=True, exclude: tuple[str]= ()):
        """ Given the initialization state of the chosen algorithm enable or not some actions and settings"""
        for action in (Actions.STEP, Actions.CONTINUOUS, 'reset_phase', 'export'):
            if action not in exclude:
                self.set_action_enabled(action, enable)
        self.set_action_enabled('algorithms', not enable)
        self.settings_widget.setEnabled(enable)

    @property
    def algorithms(self) -> list[str]:
        return algo_factory.algorithms

    @property
    def algorithm_name(self) -> str:
        """ get the current algorithm name """
        return self.get_action('algorithms').currentText()

    def define_phase(self, force_reset=False):
        if self._algorithm is not None:
            phase = self._algorithm.define_input_phase(self.settings['target_phase_group', 'target_phase'],
                                                       force_reset=force_reset)
            if force_reset:
                self._current_phase = phase

    def compute_fft(self, update_plots=True):
        if self._algorithm is not None:
            self._algorithm.compute_forward_fft(update_plots=update_plots)

    def compute_phase_loop(self):
        if self.is_action_checked(Actions.CONTINUOUS):
            self.command_runner.emit(ThreadCommand(Actions.CONTINUOUS, attribute=self._current_phase))
        else:
            self.command_runner.emit(ThreadCommand(Actions.STOP))

    def compute_phase(self):
        self.command_runner.emit(ThreadCommand(Actions.STEP, attribute=self._current_phase))
        self.set_action_enabled(Actions.CONTINUOUS, True)

    def apply_mask(self, apply_to: ApplyMaskTo) -> bool:
        return self.settings[str(apply_to), 'apply_mask']

    def get_mask_as_slices(self, apply_to: ApplyMaskTo) -> Union[None, tuple[slice, slice]]:
        if self.apply_mask(apply_to):
            slices = eval(self.settings[str(apply_to), 'slices'])
            if hasattr(slices, '__iter__'):
                for _slice in slices:
                    if not isinstance(_slice, slice):
                        return None
            return slices
        else:
            return None

    def get_mask_type(self, apply_to: ApplyMaskTo) -> MaskType:
        return MaskType[self.settings[str(apply_to), 'mask_type']]

    def update_intermediate_slices(self, roi_info: RoiInfo):
        self.settings.child(str(ApplyMaskTo.INTERMEDIATE), 'slices').setValue(str(roi_info.to_slices()))

    def update_target_slices(self, roi_info: RoiInfo):
        self.settings.child(str(ApplyMaskTo.TARGET), 'slices').setValue(str(roi_info.to_slices()))

    def value_changed(self, param: Parameter):
        for applied in ApplyMaskTo.values():
            if applied in putils.get_param_path(param):
               self._algorithm.update_mask = True

    def algo_settings_changed(self):
        self.algo_changed.emit(self.algorithm)

    def export_data(self):
        if self._current_data is not None:
            file_path = select_file(save=True, ext='h5', force_save_extension=True)
            with DataToExportSaver(file_path, save_type=SaveType.custom) as saver:
                saver.add_data('/RawData', self._current_data, ioxml.parameter_to_xml_string(self._algorithm.settings))


    def process_output(self, dte: DataToExport):
        self._current_data = dte.deepcopy()
        self._current_phase: np.ndarray = dte.get_data_from_full_name('object/phase')[0].copy()


        self.object_field_signal.emit(
            Field('object',
                  amplitude=dte.get_data_from_full_name('object/amplitude')[0],
                  phase=dte.get_data_from_full_name('object/phase')[0],
                  pixel_sizes=self._input_field.pixels_sizes))

        self.fields_to_plot.emit(dte)


class AlgoRunner(QtCore.QObject):
    algo_output_signal = QtCore.Signal(DataToExport)
    algo_stopped_signal = QtCore.Signal()

    def __init__(self, algo: AlgoBase):
        super().__init__()

        self.algo: AlgoBase = algo
        self.algo.do_things_after_init()
        self.running = False

    def queue_command(self, command: ThreadCommand):
        """
        """
        if command.command == Actions.CONTINUOUS:
            self.continuous_algo(command.attribute)

        elif command.command == Actions.STEP:
            self.step_algo(command.attribute)

        elif command.command == Actions.STOP:
            self.running = False
            self.algo.stop()

    def step_algo(self, ini_phase: np.ndarray = None):
        self.algo.start()
        self.algo.compute_phase(do_step=True, ini_phase=ini_phase)
        self.algo_output_signal.emit(self.algo.get_fields_to_plot())
        self.algo.stop()
        self.algo_stopped_signal.emit()

    def continuous_algo(self, ini_phase: np.ndarray = None):
        self.running = True
        if self.algo.MANUAL_LOOP:
            while self.running:
                self.algo.start()
                self.algo.compute_phase(do_step=True, ini_phase=ini_phase)
                self.algo_output_signal.emit(self.algo.get_fields_to_plot())
                QtWidgets.QApplication.processEvents()
        else:
            self.algo.start()
            self.algo.compute_phase(do_step=False, ini_phase=ini_phase)  # the continuous run is handled by the algo itself. If possible
            #it should update the plots during the course of the initialization... See minimizer.py as an example
        self.algo.stop()
        self.algo_stopped_signal.emit()

def main():
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path
    from pymodaq_utils.math_utils import normalize_to

    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import rescale, resize

    from pymodaq_gui.utils.utils import mkQApp

    cheshire_cat_path = Path(__file__).parent.parent.joinpath(
        'resources/cheshirecat_rect.png')
    cemes_path = Path(__file__).parent.parent.joinpath(
        'resources/Cemes - Logo - Sigle - Blanc.png')

    app = mkQApp('Optical Shaping')

    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')
    win.show()

    # get amplitude
    target_intensity = imread(cheshire_cat_path)
    if len(target_intensity.shape) == 3:
        target_intensity = rgb2gray(target_intensity[..., 0:3])

    ratio = np.max(np.array((1080, 1920)) / np.array(target_intensity.shape))
    target_intensity = rescale(target_intensity, 1 * ratio)

    # get phase
    target_phase = imread(cemes_path)
    if len(target_phase.shape) == 3:
        target_phase = rgb2gray(target_phase[..., 0:3])

    ratio = np.max(np.array((1080, 1920)) / np.array(target_phase.shape))
    target_phase = rescale(target_phase, 1 * ratio)
    target_phase =  normalize_to(target_phase, 2* np.pi)

    algo_app = AlgoApp(area)
    target = Field(amplitude=np.sqrt(np.flipud(normalize_to(target_phase, 1))),
                   phase=np.flipud(target_phase))
    input = Field(amplitude=np.ones(target.shape))

    algo_app.set_target_field(target)
    algo_app.set_input_field(input)


    app.exec()


if __name__ == '__main__':
    main()
