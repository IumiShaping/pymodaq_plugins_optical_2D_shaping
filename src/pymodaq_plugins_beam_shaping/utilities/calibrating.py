import sys
from typing import Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtWidgets

from pymodaq_gui.messenger import messagebox, dialog
from pymodaq_gui.plotting.data_viewers import Viewer1D
from pymodaq_utils.config import get_set_path, get_set_local_dir

from pymodaq_data import DataWithAxes
from pymodaq_data.h5modules.data_saving import DataLoader, DataSaverLoader
from pymodaq_gui.utils import DockArea
from pymodaq.extensions.scan.daq_scan import DAQScan

if TYPE_CHECKING:
    from pymodaq.dashboard import DashBoard


def copy_scanner_settings():
    import shutil
    from pymodaq.extensions.scan.scan_manager import ScanManager
    scanner_settings = Path(__file__).parent.parent.joinpath('resources/holography_calibration.xml')
    shutil.copyfile(str(scanner_settings),
                    str(ScanManager.get_scanner_folder().joinpath('holography_calibration.xml')))


copy_scanner_settings()


class BeamShapingCalibration(DAQScan):
    def __init__(self, parent: Union[DockArea, QtWidgets.QWidget, QtWidgets.QMainWindow],
                 dashboard: 'DashBoard'):

        self.phase_viewer: Viewer1D = None
        super().__init__(parent, dashboard)

    @classmethod
    def get_calibration_folder(cls, user=False) -> Path:
        return cls.get_local_folder(user=user)

    @classmethod
    def get_calibration_filepath(cls) -> Path:
        return cls.get_calibration_folder().joinpath('calibration.h5')

    def do_things_after_ui_setup(self):
        super().do_things_after_ui_setup()
        self.show_widgets(False)

    def show_widgets(self, show=True):
        for widget in (self.ui.dock_command, self.ui.get_toolbar('scan_manager')):
            widget.setVisible(show)

    def do_things_after_experiment_set(self, experiment_name: str):
        if not ('Shaper' in self.modules_manager.actuators_name and
                'Camera' in self.modules_manager.detectors_name):
            messagebox(title='Control Modules',
                       text='To perform Calibration, you should have an actuator'
                            'named Shaper and a camera named Camera in the DashBoard')
            return
        super().do_things_after_experiment_set(experiment_name)
        self.scan_manager.entry = 'holography_calibration'
        self.scan_manager.execute_entry()

        QtWidgets.QApplication.processEvents()

        self.connect_start_stop_step()

    def connect_start_stop_step(self):
        self.connect_action('start_val', self.scanner.scanner.settings.child('start').setValue,
                            signal_name='valueChanged')
        self.connect_action('start_val', lambda: self.scanner.settings.child('calculate_positions').setValue(True),
                            signal_name='valueChanged')
        self.connect_action('stop_val', self.scanner.scanner.settings.child('stop').setValue,
                            signal_name='valueChanged')
        self.connect_action('stop_val', lambda: self.scanner.settings.child('calculate_positions').setValue(True),
                            signal_name='valueChanged')
        self.connect_action('step_val', self.scanner.scanner.settings.child('step').setValue,
                            signal_name='valueChanged')
        self.connect_action('step_val', lambda: self.scanner.settings.child('calculate_positions').setValue(True),
                            signal_name='valueChanged')

    def setup_docks_and_widgets(self):
        super().setup_docks_and_widgets()

        self.phase_viewer = Viewer1D(title='Phase')
        self.phase_viewer.parent.setVisible(False)

    def setup_actions(self):
        super().setup_actions()
        self.toolbar.addSeparator()
        self.add_widget('start_val', QtWidgets.QSpinBox(minimum=0, maximum=255, value=0),
                        tip='Start value of the calibration')
        self.add_widget('stop_val', QtWidgets.QSpinBox(minimum=0, maximum=255, value=255),
                        tip='Stop value of the calibration')
        self.add_widget('step_val', QtWidgets.QSpinBox(minimum=0, maximum=255, value=1),
                        tip='Step value of the calibration')
        self.toolbar.addSeparator()
        self.add_action('show_options', 'Show Scanner Options', 'build_circle', tip='Show options', checkable=True)
        self.toolbar.addSeparator()
        self.add_action('show_calibration', 'Show Calibration', 'visibility',
                            'Show/Hide the Calibration Viewer', checkable=True,
                            icon_color=self.get_theme().green,
                            icon_checked='visibility_off',
                            icon_checked_color=self.get_theme().red)

    def connect_things(self):
        super().connect_things()
        self.connect_action('show_options', self.show_widgets)
        self.connect_action('show_calibration', lambda: self.show_calibration())
        self.scan_done_signal.connect(self.save_calibration)

    def save_calibration(self):
        with DataLoader(self.h5saver) as loader:
            try:
                dwa_calibration = loader.load_data_from_name_origin(
                    name='Phase', origin='Scanner')

                #rename origin
                dwa_calibration.origin = 'Calibration'
                # unwrap between 0 and 2pi
                dwa_calibration[0] = np.abs(np.unwrap(dwa_calibration[0] -dwa_calibration[0][0]))
                # remove nav indexes
                dwa_calibration.nav_indexes = ()  # empty tuple

                do_save = True
                if self.get_calibration_filepath().is_file():
                    do_save = dialog(title='Calibration File Overwrite',
                                 message='Calibration file exists, do you want '
                                         'to overwrite it?')
                    if do_save:
                        self.get_calibration_filepath().unlink()
                if do_save:
                    with DataSaverLoader(self.get_calibration_filepath()) as saver:
                        saver.add_data(where=saver.raw_group, data=dwa_calibration)
            except NameError:
                pass
        self.show_calibration(dwa_calibration)

    @classmethod
    def get_calibration_dwa(cls) -> DataWithAxes:
        if cls.get_calibration_filepath().is_file():
            with DataLoader(cls.get_calibration_filepath()) as saver:
                dwa = saver.load_data_from_name_origin(where=saver.raw_group,
                                                       name='Phase',
                                                       origin='Calibration')
            return dwa
        else:
            raise NameError('Calibration file not found.')

    def show_calibration(self, calibration: DataWithAxes = None):
        if calibration is None:
            try:
                calibration = self.get_calibration_dwa()
            except NameError:
                messagebox(title='Calibration Data',
                           text='Could not load calibration data, '
                                'you should do a new calibration')
                self.set_action_checked('show_calibration', False)
                return

        self.phase_viewer.setVisible(True)
        self.phase_viewer.show_data(calibration)


def main():
    from pymodaq_gui.qt_utils import mkQApp
    from pymodaq.dashboard import create_load_dashboard
    from pymodaq.utils.gui_utils.loader_utils import create_extension

    qapp = mkQApp('SLMCalibration')

    win, dashboard = create_load_dashboard()
    win.mainwindow.setVisible(False)

    win_ext, scan = create_extension(dashboard, BeamShapingCalibration)
    win_ext.show()

    sys.exit(qapp.exec())


if __name__ == "__main__":
    main()