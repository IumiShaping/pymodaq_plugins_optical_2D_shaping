from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_gui.managers.parameter_manager import ParameterManager, Parameter
from pymodaq_gui.parameter import utils as putils
from pymodaq_gui.parameter.utils import iter_children
from pymodaq_utils.enums import BaseEnum, enum_checker
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.plotting.data_viewers import ViewerDispatcher, Viewer0D, Viewer2D
from pymodaq_data.data import DataRaw, DataToExport
from pymodaq_gui.utils.custom_app import CustomApp
from pymodaq_gui.utils.dock import DockArea, Dock
from pymodaq_gui.utils import QLED
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_optical_2D_shaping.algorithms import algo_factory, AlgoBase
from pymodaq_plugins_optical_2D_shaping.field import Field
from pymodaq_plugins_optical_2D_shaping import config as plugin_config


class AlgoApp(CustomApp):
    params = [
        {'title': 'Algorithm', 'name': 'algorithm', 'type': 'list',
         'limits': algo_factory.algorithms, 'value': plugin_config('algo', 'default_algo')},
    ]

    command_runner = QtCore.Signal(ThreadCommand)
    object_field_signal = QtCore.Signal(Field)
    algo_changed = QtCore.Signal(AlgoBase)

    def __init__(self, dockarea):
        super().__init__(dockarea)

        self.runner_thread: QtCore.QThread = None

        self._algorithm: AlgoBase = None
        self._target_field: Field = None
        self._input_field: Field = None

        self.setup_ui()

        self.set_algorithm(self.settings['algorithm'])

    @property
    def algorithm(self):
        return self._algorithm

    def set_target_field(self, field: Field):
        if self._algorithm is not None:
            field = self._algorithm.scale_target_with_geometry(field)
            self._algorithm.set_target_field(field)
            self.target_viewers.show_data(DataToExport('Target', data=[
                field.intensity_as_dwa(),
                field.amplitude_as_dwa(),
                field.phase_as_dwa(),
            ]))
        self._target_field = field

    def set_input_field(self, field: Field):

        object_field = field.deepcopy()
        object_field.phase = np.random.random(field.shape) * 2 * np.pi

        if self._algorithm is not None:
            self._algorithm.set_object_field(object_field)
            self._algorithm.set_input_field(field)

        self._input_field = field

    def set_algorithm(self, algo_name: str = None):
        if algo_name is None:
            algo_name = self.settings['algorithm']
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

        except ValueError as e:
            pass

    def setup_docks(self):

        self.algo_area = self.dockarea

        self.docks['algo_settings'] = Dock('Algorithm Settings')
        self.docks['image_field'] = Dock('Image Plane')
        self.docks['object_field'] = Dock('Object Plane')
        self.docks['fitness'] = Dock('Fitness')

        self.dockarea.addDock(self.docks['algo_settings'])
        self.dockarea.addDock(self.docks['fitness'], 'right', self.docks['algo_settings'])
        self.dockarea.addDock(self.docks['object_field'], 'bottom', self.docks['fitness'])
        self.dockarea.addDock(self.docks['image_field'], 'bottom', self.docks['object_field'])

        fitness_widget = QtWidgets.QWidget()
        self.fitness_viewer = Viewer0D(fitness_widget)
        self.docks['fitness'].addWidget(fitness_widget)

        self.target_widget = QtWidgets.QWidget()
        self.target_widget.setLayout(QtWidgets.QHBoxLayout())
        target_area = DockArea()
        self.target_viewers = ViewerDispatcher(target_area)
        self.target_widget.layout().addWidget(target_area)
        self.target_widget.setVisible(False)

        object_area = DockArea()
        self.object_viewers = ViewerDispatcher(object_area)
        self.docks['object_field'].addWidget(object_area)

        image_area = DockArea()
        self.image_viewers = ViewerDispatcher(image_area)
        self.docks['image_field'].addWidget(image_area)

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
        self.add_action('ini_algo', 'Init Algo', 'ini', checkable=True)
        self.add_widget('algo_led', QLED, toolbar=self.toolbar)
        self.add_action('snap', 'Snap', 'snap', "Take a snapshot from the detector")
        self.add_action('grab', 'Grab', 'run2', "Grab data from the detector", checkable=True)
        self.add_action('stop', 'Stop', 'stop', "Stop grabing")
        self.add_action('show_target', 'Show Target', 'target',
                        "Show Target in real units", checkable=True)

    def connect_things(self):
        self.connect_action('snap', self.compute_phase)
        self.connect_action('grab', self.compute_phase_loop)
        self.connect_action('ini_algo', self.ini_algo)
        self.connect_action('stop', self.stop)
        self.connect_action('show_target', lambda show: self.target_widget.setVisible(show))

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
        if param.name() == 'algorithm':
            self.algo_changed(self._algorithm)

    def process_output(self, dte: DataToExport):
        fitness = dte.remove(dte.get_data_from_name('fitness'))
        dte_image = dte.get_data_from_full_names(['image/amplitude', 'image/phase'])
        dte_object = dte.get_data_from_full_names(['object/amplitude', 'object/phase'])
        self.object_viewers.show_data(dte_object)
        self.image_viewers.show_data(dte_image)
        self.fitness_viewer.show_data(fitness)

        self.object_field_signal.emit(
            Field('object',
                  amplitude=dte.get_data_from_full_name('object/amplitude')[0],
                  phase=dte.get_data_from_full_name('object/phase')[0],
                  pixel_sizes=self._input_field.pixels_sizes))

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
        self.algo_output_signal.emit(DataToExport('AlgoData', data=[
            self.algo.image_field.amplitude_as_dwa('image'),
            self.algo.image_field.phase_as_dwa('image'),
            self.algo.fitness_as_dwa(),
            self.algo.object_field.amplitude_as_dwa('object'),
            self.algo.object_field.phase_as_dwa('object'),

        ]))

    def run_algo(self):
        self.running = True
        while self.running:
            self.snap_algo()
            QtWidgets.QApplication.processEvents()

def main():
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path

    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import rescale, resize

    from pymodaq_gui.utils.utils import mkQApp

    cheshire_cat_path = Path(__file__).parent.parent.joinpath(
        'resources/cheshirecat_rect.png')

    app = mkQApp('Optical Shaping')

    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')
    win.show()

    target_intensity = imread(cheshire_cat_path)
    if len(target_intensity.shape) == 3:
        target_intensity = rgb2gray(target_intensity[..., 0:3])

    ratio = np.max(np.array((1080, 1920)) / np.array(target_intensity.shape))
    target_intensity = rescale(target_intensity, 1 * ratio)

    algo_app = AlgoApp(area)
    target = Field(amplitude=np.sqrt(np.flipud(target_intensity)))
    input = Field(amplitude=np.ones(target.shape))

    algo_app.set_target_field(target)
    algo_app.set_input_field(input)


    app.exec()


if __name__ == '__main__':
    main()
