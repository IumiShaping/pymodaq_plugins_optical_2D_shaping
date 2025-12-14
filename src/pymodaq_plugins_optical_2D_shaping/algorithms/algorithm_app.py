from abc import ABCMeta, abstractproperty
from typing import Union
import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_utils.utils import ThreadCommand


from pymodaq_data.data import DataRaw, DataToExport
from pymodaq_data.h5modules.data_saving import DataToExportSaver
from pymodaq_data.h5modules.saving import SaveType

from pymodaq_gui.managers.parameter_manager import Parameter
from pymodaq_gui.plotting.data_viewers import ViewerDispatcher, Viewer0D
from pymodaq_gui.utils.custom_app import CustomApp
from pymodaq_gui.utils.dock import DockArea, Dock
from pymodaq_gui.utils import QLED
from pymodaq_gui.utils.file_io import select_file
from pymodaq_gui.parameter import ioxml
from pymodaq_gui.utils.widget_sync import WidgetSync, SyncMode
from pymodaq_gui.managers.roi_manager import ROI2D_TYPES, ROI
from pymodaq_gui.plotting.utils.plot_utils import RoiInfo

from pymodaq_plugins_optical_2D_shaping.algorithms import algo_factory, AlgoBase
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import TargetPhase
from pymodaq_plugins_optical_2D_shaping.field import Field
from pymodaq_plugins_optical_2D_shaping import config as plugin_config



class AlgoApp(CustomApp):
    params = [
        # {'title': 'Algorithm', 'name': 'algorithm', 'type': 'list',
        #  'limits': algo_factory.algorithms, 'value': plugin_config('algo', 'default_algo')},
        {'title': 'Target Phase', 'name': 'target_phase', 'type': 'list',
         'limits': TargetPhase.values(), 'value': TargetPhase.QUADRATIC.value, },
        {'title': 'Masking', 'name': 'masking', 'type': 'group', 'children': [
            {'title': 'Apply Mask', 'name': 'apply_mask', 'type': 'bool', 'value': False},
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

        self.setup_ui()

        #self.get_action('ini_algo').trigger()

    @property
    def algorithm_combo(self) -> QtWidgets.QComboBox:
        return self.get_action('algorithms')

    # @property
    # def algorithm_param(self) -> Parameter:
    #     return self.settings.child('algorithm')

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
            self._algorithm.define_input_phase(self.settings['target_phase'])

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
            self.set_action_visible('grab', self._algorithm.ITERATIVE)

            self.algo_changed.emit(self._algorithm)

        except ValueError as e:
            pass

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
        self.get_action('algorithms').addItems(self.algorithms)
        self.get_action('algorithms').setCurrentText(plugin_config('algo', 'default_algo'))
        self.add_action('ini_algo', 'Init Algo', 'ini', checkable=True)
        self.add_widget('algo_led', QLED)
        self.add_action('snap', 'Snap', 'snap', "Run a loop of the algorithm")
        self.add_action('grab', 'Grab', 'run2', "Run continuously the algorithm", checkable=True)
        self.add_action('stop', 'Stop', 'stop', "Stop the algorithm")
        self.add_action('reset_phase', 'Reset Phase', 'Refresh2', "Reset the SLM phase")
        self.add_action('export', 'Export', 'SaveAs', 'Export data')

    def connect_things(self):
        self.connect_action('snap', self.compute_phase)
        self.connect_action('grab', self.compute_phase_loop)
        self.connect_action('ini_algo', self.ini_algo)
        self.connect_action('stop', self.stop)
        self.connect_action('export', self.export_data)
        self.connect_action('reset_phase', self.define_phase)
        self.connect_action('algorithms', slot=self.set_algorithm,
                            signal_name='currentTextChanged')

    @property
    def algorithms(self) -> list[str]:
        return algo_factory.algorithms

    @property
    def algorithm_name(self) -> str:
        """ get the current algorithm name """
        return self.get_action('algorithms').currentText()

    def define_phase(self):
        if self._algorithm is not None:
            self._algorithm.define_input_phase(self.settings['target_phase'])

    def stop(self):
        self.command_runner.emit(ThreadCommand('stop'))
        self.set_action_checked('grab', False)

    def compute_phase_loop(self):
        if self.is_action_checked('grab'):
            self.command_runner.emit(ThreadCommand('grab'))
        else:
            self.command_runner.emit(ThreadCommand('stop'))

    def compute_phase(self):
        self.command_runner.emit(ThreadCommand('snap'))

    def value_changed(self, param: Parameter):

        if param.name() in ('apply_mask', 'slices'):
            if self.settings['masking', 'apply_mask']:
                slices = eval(self.settings['masking', 'slices'])
                if hasattr(slices, '__iter__'):
                    for _slice in slices:
                        if not isinstance(_slice, slice):
                            return
                    self.algorithm.set_mask(slices)
            else:
                self.algorithm.set_mask(None)

    def algo_settings_changed(self):
        self.algo_changed.emit(self.algorithm)

    def export_data(self):
        if self._current_data is not None:
            file_path = select_file(save=True, ext='h5', force_save_extension=True)
            with DataToExportSaver(file_path, save_type=SaveType.custom) as saver:
                saver.add_data('/RawData', self._current_data, ioxml.parameter_to_xml_string(self._algorithm.settings))


    def process_output(self, dte: DataToExport):
        self._current_data = dte.deepcopy()

        # fitness = dte.remove(dte.get_data_from_name('fitness'))
        # dte_image = dte.get_data_from_full_names(['image/amplitude', 'image/phase'])
        # dte_object = dte.get_data_from_full_names(['object/amplitude', 'object/phase'])
        # self.object_viewers.show_data(dte_object)
        # self.image_viewers.show_data(dte_image)
        # self.fitness_viewer.show_data(fitness)

        self.object_field_signal.emit(
            Field('object',
                  amplitude=dte.get_data_from_full_name('object/amplitude')[0],
                  phase=dte.get_data_from_full_name('object/phase')[0],
                  pixel_sizes=self._input_field.pixels_sizes))

        self.fields_to_plot.emit(dte)

    def ini_algo(self):
        if self.is_action_checked('ini_algo'):
            self.get_action('algo_led').set_as_true()
            #self.set_action_enabled('ini_algo', False)
            self.set_algorithm()

            self.runner_thread = QtCore.QThread()
            runner = AlgoRunner(self._algorithm)

            self.runner_thread.runner = runner
            runner.algo_output_signal.connect(self.process_output)
            self.command_runner.connect(runner.queue_command)

            runner.moveToThread(self.runner_thread)

            self.runner_thread.start()

        else:
            if self.runner_thread is not None:
                self.get_action('algo_led').set_as_false()
                self.command_runner.disconnect()
                if self.runner_thread.isRunning():
                    self.runner_thread.terminate()
                    while not self.runner_thread.isFinished():
                        QtCore.QThread.msleep(100)
                    self.runner_thread = None


class AlgoRunner(QtCore.QObject):
    algo_output_signal = QtCore.Signal(DataToExport)

    def __init__(self, algo: AlgoBase):
        super().__init__()

        self.algo: AlgoBase = algo
        self.running = False

    def queue_command(self, command: ThreadCommand):
        """
        """
        if command.command == "grab":
            self.run_algo()

        elif command.command == "snap":
            self.snap_algo()

        elif command.command == "stop":
            self.running = False

    def snap_algo(self):
        self.algo.compute_phase()
        self.algo_output_signal.emit(self.algo.get_fields_to_plot())


    def run_algo(self):
        self.running = True
        while self.running:
            self.snap_algo()
            QtWidgets.QApplication.processEvents()

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
