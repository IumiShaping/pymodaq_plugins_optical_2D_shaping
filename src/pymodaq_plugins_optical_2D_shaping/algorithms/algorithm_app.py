from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.utils.managers.parameter_manager import ParameterManager, Parameter
from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.parameter.utils import iter_children
from pymodaq.utils.enums import BaseEnum, enum_checker
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq.utils.plotting.data_viewers.viewer0D import Viewer0D
from pymodaq.utils.data import DataRaw
from pymodaq.utils.gui_utils.custom_app import CustomApp
from pymodaq.utils.gui_utils.dock import DockArea, Dock


from pymodaq_plugins_optical_2D_shaping.algorithms import algo_factory, AlgoBase
from pymodaq_plugins_optical_2D_shaping.target_loaders.field import Field


class AlgoApp(CustomApp):
    params = [
        {'title': 'Algorithm', 'name': 'algorithm', 'type': 'list',
         'limits': algo_factory.algorithms, 'value': algo_factory.algorithms[0]},

    ]

    def __init__(self, dockarea):
        super().__init__(dockarea)

        self._algorithm: AlgoApp = None
        self._target_field: Field = None

        self.setup_ui()

        self.set_algorithm(algo_factory.algorithms[0])

    def set_field(self, field: Field):
        self._target_field = field

    def set_algorithm(self, algo_name: str):
        try:
            self._algorithm: AlgoApp = \
                algo_factory.get_algorithm(algo_name)()

            while True:
                child = self._algo_settings_widget.layout().takeAt(0)
                if not child:
                    break
                child.widget().deleteLater()
                QtWidgets.QApplication.processEvents()

            self._algo_settings_widget.layout().addWidget(self._algorithm.settings_tree)

        except ValueError as e:
            pass

    def setup_docks(self):
        self.docks['algo'] = Dock('Algorithm')
        self.dockarea.addDock(self.docks['algo'])

        self.algo_area = DockArea()
        self.docks['algo'].addWidget(self.algo_area)

        self.docks['algo_settings'] = Dock('Algorithm Settings')
        self.docks['object_field'] = Dock('Object')
        self.docks['fitness'] = Dock('Fitness')

        self.algo_area.addDock(self.docks['algo_settings'])
        self.algo_area.addDock(self.docks['fitness'], 'right', self.docks['algo_settings'])
        self.algo_area.addDock(self.docks['object_field'], 'bottom', self.docks['fitness'])


        fitness_widget = QtWidgets.QWidget()
        self.fitness_viewer = Viewer0D(fitness_widget)
        self.docks['fitness'].addWidget(fitness_widget)

        amp_widget = QtWidgets.QWidget()
        self.amp_viewer = Viewer2D(amp_widget)

        phase_widget = QtWidgets.QWidget()
        self.phase_viewer = Viewer2D(phase_widget)

        self.docks['object_field'].addWidget(amp_widget)
        self.docks['object_field'].addWidget(phase_widget, row=0, col=1)

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
        ...

    def connect_things(self):
        ...


def main():
    from pathlib import Path
    from pymodaq.utils.daq_utils import get_set_preset_path
    from pymodaq.utils.gui_utils.utils import mkQApp

    app = mkQApp('Optical Shaping')

    win = QtWidgets.QMainWindow()
    area = DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')
    win.show()

    optical_app = AlgoApp(area)

    app.exec()


if __name__ == '__main__':
    main()
