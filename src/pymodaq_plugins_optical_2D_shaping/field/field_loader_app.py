
import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.utils.managers.parameter_manager import Parameter
from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq.utils.gui_utils.custom_app import CustomApp
from pymodaq.utils.gui_utils.dock import DockArea, Dock

from pymodaq_plugins_optical_2D_shaping.field import Field, field_loader_factory
from pymodaq_plugins_optical_2D_shaping.field.factory import LoaderFactory, FieldLoader


class FieldLoaderApp(CustomApp):

    params = [
        {'title': 'Target Loader', 'name': 'loader', 'type': 'list',
         'limits': field_loader_factory.field_loaders,
         'value': field_loader_factory.field_loaders[0]},

        {'title': 'Target utils', 'name': 'utils', 'type': 'group', 'children': [
            {'title': 'Reload:', 'name': 'reload', 'type': 'bool_push', 'label': 'Reload!',
             'value': False},
            {'title': 'Show target', 'name': 'show_target', 'type': 'bool_push', 'value': True},
            {'title': 'Flip ud', 'name': 'flipud', 'type': 'bool', 'value': False},
            {'title': 'Flip lr', 'name': 'fliplr', 'type': 'bool', 'value': False},
        ],
         },
    ]

    field_signal = QtCore.Signal(Field)

    def __init__(self, dockarea):
        super().__init__(dockarea)
        self._main_widget: QtWidgets.QWidget = None
        self.target_widget: QtWidgets.QWidget = None
        self.settings_widget: QtWidgets.QWidget = None
        self._target_settings_widget: QtWidgets.QWidget = None

        self.amp_viewer: Viewer2D = None
        self.phase_viewer: Viewer2D = None

        self._target_loader: FieldLoader = None

        self.field = Field()

        self.setup_ui()

        self.set_loader(field_loader_factory.field_loaders[0])

    def value_changed(self, param: Parameter):
        if param.name() == 'loader':
            self.set_loader(param.value())

        elif param.name() in ('flipud', 'fliplr'):
            self.transform_image(param.name())

        elif param.name() == 'show_target':
            self.target_widget.setVisible(param.value())

        elif param.name() == 'reload':
            self._target_loader.load_target()

    def update_field(self, field: Field):
        """ Method used for notification when its parent object is registered within a FieldLoader
        """
        self.field = field
        self.update_viewers()
        self.field_signal.emit(self.field)

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
        if self.field is not None:
            self.amp_viewer.show_data(self.field.amplitude_as_dwa())
            self.phase_viewer.show_data(self.field.phase_as_dwa())

    def set_loader(self, loader_name: str):
        try:
            self._target_loader: FieldLoader =\
                field_loader_factory.get_loader(loader_name)()

            while True:
                child = self._target_settings_widget.layout().takeAt(0)
                if not child:
                    break
                child.widget().deleteLater()
                QtWidgets.QApplication.processEvents()

            self._target_settings_widget.layout().addWidget(self._target_loader.settings_tree)
            self._target_loader.register_listener(self)

        except ValueError as e:
            pass

    def setup_docks(self):
        self.docks['target'] = Dock('Target')
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(QtWidgets.QHBoxLayout())

        self.target_widget = QtWidgets.QWidget()
        self.target_widget.setLayout(QtWidgets.QHBoxLayout())

        amp_widget = QtWidgets.QWidget()
        self.amp_viewer = Viewer2D(amp_widget)

        phase_widget = QtWidgets.QWidget()
        self.phase_viewer = Viewer2D(phase_widget)

        self.target_widget.layout().addWidget(amp_widget)
        self.target_widget.layout().addWidget(phase_widget)

        self.dockarea.addDock(self.docks['target'])

        self.settings_widget = QtWidgets.QWidget()
        self.settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self.settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.settings_widget.layout().addWidget(splitter)
        splitter.addWidget(self.settings_tree)

        self._target_settings_widget = QtWidgets.QWidget()
        self._target_settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self._target_settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._target_settings_widget)

        self._main_widget.layout().addWidget(self.settings_widget)
        self.settings_tree.setMinimumWidth(300)
        self.settings_tree.setMinimumHeight(150)
        self._main_widget.layout().addWidget(self.target_widget)
        self.docks['target'].addWidget(self._main_widget)

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

    optical_app = FieldLoaderApp(area)

    app.exec()


if __name__ == '__main__':
    main()
