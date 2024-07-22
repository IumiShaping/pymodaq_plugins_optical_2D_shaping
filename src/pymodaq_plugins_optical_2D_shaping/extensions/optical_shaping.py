from typing import List, Union
import time


import numpy as np
from qtpy import QtWidgets, QtCore


from pymodaq.utils import gui_utils as gutils
from pymodaq.utils import daq_utils as utils
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.data import DataToExport, DataActuator, DataCalculated
from pymodaq.utils.plotting.data_viewers.viewer0D import Viewer0D
from pymodaq.utils.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_plugins_optical_2D_shaping.utils import (get_optimisation_models,
                                                      OptimisationModelGeneric,
                                                      DataToActuatorOpti)
from pymodaq.utils.gui_utils import QLED
from pymodaq.utils.managers.modules_manager import ModulesManager
from pymodaq.utils.config import Config

from pymodaq_plugins_optical_2D_shaping import config as plugin_config
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp

from pymodaq_plugins_optical_2D_shaping.field.field_loader_app import FieldLoaderApp, Field

logger = set_logger(get_module_name(__file__))

config = Config()


EXTENSION_NAME = 'Optical Shaping'
CLASS_NAME = 'OpticalShaping'


class OpticalShaping(gutils.CustomApp):
    command_runner = QtCore.Signal(utils.ThreadCommand)
    models = get_optimisation_models()

    params = [
    ]

    def __init__(self, dockarea, dashboard):
        super().__init__(dockarea, dashboard)

        self.viewer_fitness: Viewer0D = None
        self.viewer_observable: ViewerDispatcher = None
        self.model_class: OptimisationModelGeneric = None

        self._target_loader: FieldLoaderApp = None
        self._target_field: Field = None

        self._input_field_loader: FieldLoaderApp = None
        self._input_field: Field = None

        self._algorithm: AlgoApp = None

        self.setup_ui()

    def update_target(self, field: Field):
        self._target_field = field
        self._algorithm.set_target_field(self._target_field)

    def update_input(self, field: Field):
        self._input_field = field
        self._algorithm.set_input_field(self._input_field)

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
        self._target_loader = FieldLoaderApp(self._target_dockarea)
        self._target_loader.set_loader_in_settings(plugin_config('target', 'default_loader'))
        self._target_field = Field()

        self._input_field_dockarea = gutils.DockArea()
        self._input_field_loader = FieldLoaderApp(self._input_field_dockarea)
        self._input_field_loader.set_loader_in_settings(plugin_config('input', 'default_loader'))
        self._input_field: Field = Field()

        self.docks['algo'] = gutils.Dock('Algo')
        self.dockarea.addDock(self.docks['algo'])
        algo_main_window = QtWidgets.QMainWindow()
        self._algo_dockarea = gutils.DockArea()
        algo_main_window.setCentralWidget(self._algo_dockarea)
        self.docks['algo'].addWidget(algo_main_window)

        self._algorithm = AlgoApp(self._algo_dockarea)

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

        self.add_action('target', 'Target', 'target', 'Open the Target FieldLoader window',
                        checkable=True)
        self.add_action('input', 'Input', 'input', 'Open the InputBeam FieldLoader window',
                        checkable=True)
        self.add_action('algo', 'Algo.', 'algo', 'Open the Algorithm window', checkable=True)
        self.set_action_checked('algo', True)

        self.add_action('run', 'Run Optimisation', 'run2', checkable=True)
        self.add_action('pause', 'Pause Optimisation', 'pause', checkable=True)
        logger.debug('actions set')

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('quit', self.quit, )

        self.connect_action('target', self.show_target)
        self.connect_action('input', self.show_input)
        self.connect_action('algo', self.show_algo)

        self._target_loader.field_signal.connect(self.update_target)
        self._input_field_loader.field_signal.connect(self.update_input)

        self._target_loader.load_field()
        self._input_field_loader.load_field()

    def show_target(self, show=True):
        self._target_dockarea.setVisible(show)
        self._target_dockarea.closeEvent = lambda event: self.set_action_checked('target', False)

    def show_input(self, show=True):
        self._input_field_dockarea.setVisible(show)
        self._input_field_dockarea.closeEvent = lambda event: self.set_action_checked('input', False)

    def show_algo(self, show=True):
        self.docks['algo'].setVisible(show)

    def quit(self):
        self._input_field_dockarea.parent().close()
        self._target_dockarea.parent().close()
        self.dockarea.parent().close()


def main_only_app():
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path
    from pymodaq.utils.gui_utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset

    app = mkQApp('Optical Shaping')

    win = QtWidgets.QMainWindow()
    area = gutils.DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')
    win.show()

    optical_app = OpticalShaping(area, None)

    app.exec()


def main():
    import sys
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path
    from pymodaq.utils.gui_utils.utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import load_dashboard_with_preset

    app = mkQApp('Optical Shaping')

    preset_file_name = str(Path(get_set_preset_path()).joinpath(f"{'holography'}.xml"))
    dashboard, extension, win = load_dashboard_with_preset(preset_file_name, 'Optical Shaping')
    app.exec()

    return dashboard, extension, win


if __name__ == '__main__':
    main_only_app()



