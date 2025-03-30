from typing import List, Union
import time


import numpy as np
from qtpy import QtWidgets, QtCore

from zernpy import ZernPol

from pymodaq_gui import utils as gutils
from pymodaq_utils import utils as utils
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.utils import  ThreadCommand

from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.data import DataToExport, DataActuator, DataCalculated
from pymodaq_gui.plotting.data_viewers.viewer0D import Viewer0D
from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_gui.parameter.pymodaq_ptypes import SliderSpinBox

from pymodaq_utils.config import Config
from pymodaq_gui.utils.widgets.tree_toml import TreeFromToml

from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp, AlgoBase

from pymodaq_plugins_optical_2D_shaping.field.field_loader_app import FieldLoaderApp, Field, Q_
from pymodaq_plugins_optical_2D_shaping.utilities.corrections import Correction
from pymodaq.extensions.utils import CustomExt

logger = set_logger(get_module_name(__file__))

config = Config()


EXTENSION_NAME = 'Optical Shaping'
CLASS_NAME = 'OpticalShaping'


class OpticalShaping(CustomExt):
    command_runner = QtCore.Signal(utils.ThreadCommand)

    params = [
    ]

    def __init__(self, dockarea, dashboard):
        super().__init__(dockarea, dashboard)

        self._plugin_config = PluginConfig()

        self.viewer_fitness: Viewer0D = None
        self.viewer_observable: ViewerDispatcher = None

        self._target_loader: FieldLoaderApp = None
        self._target_field: Field = None

        self._input_field_loader: FieldLoaderApp = None
        self._input_field: Field = None

        self._object_field: Field = None
        self._correction_phase: DataCalculated = None

        self._algorithm: AlgoApp = None

        self._corrections: Correction = None

        if self.modules_manager is not None and 'Shaper' in self.modules_manager.actuators_name:
            self._shaper = self.modules_manager.get_mod_from_name('Shaper', 'act')
        else:
            self._shaper = None

        self.setup_ui()

    def update_target(self, field: Field):
        self._target_field = field
        self._algorithm.set_target_field(self._target_field)

    def update_input(self, field: Field):
        self._input_field = field
        self._algorithm.set_input_field(self._input_field)

    def update_object(self, field: Field):
        """ field contains here the object field"""
        if field is None:
            field = self._input_field

        self._object_field = field


        if self._shaper is not None:
            phase_to_send = 0.
            send = self.is_action_checked('send_algo_to_shaper') or self.is_action_checked('send_correc_to_shaper')
            if self.is_action_checked('send_algo_to_shaper'):
                phase_to_send += field.phase_as_dwa()
            elif self.is_action_checked('send_correc_to_shaper'):
                phase_to_send += self._correction_phase

            self._shaper.move_abs(field.phase_as_dwa() + self._correction_phase)

    def update_correction_phase(self, dwa: DataCalculated):
        self._correction_phase = dwa
        self.update_object(self._object_field)

    def setup_docks(self):
        """
        to be subclassed to setup the docks layout
        for instance:

        self.docks['ADock'] = gutils.Dock('ADock name)
        self.dockarea.addDock(self.docks['ADock"])
        self.docks['AnotherDock'] = gutils.Dock('AnotherDock name)
        self.dockarea.addDock(self.docks['AnotherDock"], 'bottom', self.docks['ADock"])

        See Also
        ########
        pyqtgraph.dockarea.Dock
        """

        self._target_dockarea = gutils.DockArea()
        self._target_loader = FieldLoaderApp(self._target_dockarea,
                                             modules_manager=self.modules_manager)
        self._target_loader.set_loader_in_settings(
            self._plugin_config('target', 'default_loader'))
        self._target_field = Field()

        self._input_field_dockarea = gutils.DockArea()
        self._input_field_loader = FieldLoaderApp(self._input_field_dockarea)
        self._input_field_loader.set_loader_in_settings(
            self._plugin_config('input', 'default_loader'))
        self._input_field: Field = Field()

        self.docks['algo'] = gutils.Dock('Algo')
        self.dockarea.addDock(self.docks['algo'])
        algo_main_window = QtWidgets.QMainWindow()
        self._algo_dockarea = gutils.DockArea()
        algo_main_window.setCentralWidget(self._algo_dockarea)
        self.docks['algo'].addWidget(algo_main_window)

        self._algorithm = AlgoApp(self._algo_dockarea)

        self._corrections_dockarea = gutils.DockArea()
        self._corrections = Correction(self._corrections_dockarea)

    def setup_menu(self):
        """
        to be subclassed
        create menu for actions contained into the self.actions_manager, for instance:

        For instance:

        file_menu = self.menubar.addMenu('File')
        self.actions_manager.affect_to('load', file_menu)
        self.actions_manager.affect_to('save', file_menu)

        file_menu.addSeparator()
        self.actions_manager.affect_to('quit', file_menu)
        """
        pass

    def value_changed(self, param):
        """ to be subclassed for actions to perform when one of the param's value in self.settings is changed

        For instance:
        if param.name() == 'do_something':
            if param.value():
                print('Do something')
                self.settings.child('main_settings', 'something_done').setValue(False)

        Parameters
        ----------
        param: (Parameter) the parameter whose value just changed
        """
        ...

    def setup_actions(self):
        logger.debug('setting actions')
        self.add_action('quit', 'Quit', 'close2', "Quit program")

        self.add_action('settings', 'Plugin Settings', 'Settings',
                        'Open the plugin configuration file',
                        checkable=True)

        self.add_action('target', 'Target Selection', 'target',
                        'Open the Target FieldLoader window', checkable=True)
        self.add_action('input', 'Input Beam Selection', 'input',
                        'Open the InputBeam FieldLoader window', checkable=True)
        self.add_action('algo', 'Algo. Selection', 'algo', 'Open the Algorithm window', checkable=True)
        self.set_action_checked('algo', True)

        self.add_action('run', 'Run Optimisation', 'run2', checkable=True)
        self.add_action('pause', 'Pause Optimisation', 'pause', checkable=True)

        self.add_action('send_algo_to_shaper', 'Algo to shaper', 'random',
                        'Send calculated phase to the control module called *Shaper*',
                        checkable=True)
        self.add_action('corrections', 'Corrections', 'utility2',
                        tip='Open the Utility window with focal and Zernike correction',
                        checkable=True)
        self.add_action('send_correc_to_shaper', 'Correction to shaper', 'random',
                        'Send correction phase to the control module called *Shaper*',
                        checkable=True)

        logger.debug('actions set')

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('quit', self.quit, )

        self.connect_action('settings', self.show_config)
        self.connect_action('target', self.show_target)
        self.connect_action('input', self.show_input)
        self.connect_action('algo', self.show_algo)

        self.connect_action('run', self._algorithm.compute_phase_loop)
        self.connect_action('pause', self._algorithm.stop)

        self._algorithm.object_field_signal.connect(self.update_object)

        self._input_field_loader.field_signal.connect(self.update_input)
        self._target_loader.field_signal.connect(self.update_target)
        self._algorithm.algo_changed.connect(self.update_target_loader_from_algo)

        self._input_field_loader.load_field()
        self.update_target_loader_from_algo(self._algorithm.algorithm)
        self._target_loader.load_field()

        self.connect_action('corrections', self.show_corrections)
        self._corrections.phase_changed.connect(self.update_correction_phase)

    def show_corrections(self, show=True):
        self._corrections_dockarea.setVisible(show)
        self._corrections_dockarea.closeEvent = lambda event: self.set_action_checked('corrections', False)


    def _get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """ Get the pixel indexes from the selected SLM centered on the center of the SLM

        Return:
        -------
        x: np.ndarray
        y: np.ndarray
        """
        shape = self.shape
        return  (np.linspace(-shape[1] / 2, shape[1] / 2, shape[1], endpoint=True),
                 np.linspace(-shape[0] / 2, shape[0] / 2, shape[0], endpoint=True),
                 )

    @property
    def shape(self) -> tuple[int, int]:
        """ Get the shape of the configured SLM"""
        return (self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'height'),
                self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'width'),
                )

    def update_target_loader_from_algo(self, algo: AlgoBase):
        pixel_size = self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'pixel_size')
        height = Q_(self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'height'), 'um')
        width = Q_(self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'width'), 'um')
        slm_size = (pixel_size * height, pixel_size * width)
        self._target_loader.update_pixels(algo.get_target_pixels_size(slm_size))

    def show_config(self, show=True):
        if show:
            config_tree = TreeFromToml(self._plugin_config, capitalize=False)
            res = config_tree.show_dialog()
            if res:
                self._plugin_config = PluginConfig()
            self.set_action_checked('settings', False)
            self._target_loader.update_slm(
                self._plugin_config('SLM', 'default_slm'))
            self._input_field_loader.update_slm(
                self._plugin_config('SLM', 'default_slm'))

    def show_target(self, show=True):
        self._target_dockarea.setVisible(show)
        self._target_dockarea.closeEvent = lambda event: self.set_action_checked('target', False)

    def show_input(self, show=True):
        self._input_field_dockarea.setVisible(show)
        self._input_field_dockarea.closeEvent = lambda event: self.set_action_checked('input', False)

    def show_algo(self, show=True):
        self.docks['algo'].setVisible(show)

    def quit(self):
        self._input_field_dockarea.close()
        self._target_dockarea.close()
        self.dockarea.parent().close()


def main():
    from pathlib import Path
    from pymodaq.utils.config import get_set_preset_path
    from pymodaq_gui.utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset
    from pymodaq_gui.utils.dock import DockArea

    app = mkQApp('Optical Shaping')

    preset_file_name = 'holography_mock'

    file = Path(get_set_preset_path()).joinpath(f"{preset_file_name}.xml")
    if file.exists():
        dashboard, extension, win = load_dashboard_with_preset(preset_file_name, 'Optical Shaping')
    else:
        win = QtWidgets.QMainWindow()
        dockarea = DockArea()
        win.setCentralWidget(dockarea)
        extension = OpticalShaping(dockarea, None)
        win.show()

    app.exec()


if __name__ == '__main__':
    main()



