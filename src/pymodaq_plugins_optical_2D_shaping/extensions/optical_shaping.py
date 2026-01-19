import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.utils.managers import PresetManager
from pymodaq_utils import utils as utils
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.config import Config
from pymodaq.utils.data import DataToExport, DataCalculated

from pymodaq_data.h5modules.data_saving import DataToExportSaver

from pymodaq_gui.plotting.data_viewers.viewer0D import Viewer0D
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_gui.utils.file_io import select_file
from pymodaq_gui import utils as gutils
from pymodaq_gui.utils.widgets.tree_toml import TreeFromToml

from pymodaq.extensions.utils import CustomExt

from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp, AlgoBase
from pymodaq_plugins_optical_2D_shaping.field.field_loader_app import FieldLoaderApp, Field, Q_
from pymodaq_plugins_optical_2D_shaping.utilities.corrections import Correction
from pymodaq_plugins_optical_2D_shaping.algorithms import AlgorithmFactory, AlgoBase

logger = set_logger(get_module_name(__file__))

config = Config()
plugin_config = PluginConfig()
algo_factory = AlgorithmFactory

EXTENSION_NAME = 'Optical Shaping'
CLASS_NAME = 'OpticalShaping'


class OpticalShaping(CustomExt):
    command_runner = QtCore.Signal(utils.ThreadCommand)

    params = [
    ]

    def __init__(self, dockarea, dashboard):
        super().__init__(dockarea, dashboard)

        self.viewer_fitness: Viewer0D = None
        self.viewer_observable: ViewerDispatcher = None

        self._target_loader: FieldLoaderApp = None
        self._input_field_loader: FieldLoaderApp = None

        self._object_field: Field = None
        self._correction_phase: DataCalculated = None

        self.object_viewers: ViewerDispatcher = None
        self.image_viewers: ViewerDispatcher = None
        self.intermediate_viewer: Viewer2D = None
        self.other_viewers: ViewerDispatcher = None


        self._algorithm: AlgoApp = None

        self._corrections: Correction = None

        self.preset_manager: PresetManager = None

        if self.modules_manager is not None and 'Shaper' in self.modules_manager.actuators_name:
            self._shaper = self.modules_manager.get_mod_from_name('Shaper', 'act')
        else:
            self._shaper = None

        self.setup_ui()

        self.do_things_after_init()

    @property
    def input_field(self) -> Field:
        return self._input_field_loader.field

    @property
    def target_field(self) -> Field:
        return self._target_loader.field

    def do_things_after_init(self):
        self._input_field_loader.load_field()
        self._target_loader.load_field()

    def update_object(self, field: Field):
        """ field contains here the object field"""

        if field is None:
            field = self.input_field

        self._object_field = field

        if self._shaper is not None:
            phase_to_send = 0.
            if self.is_action_checked('send_algo_to_shaper') or self.is_action_checked('send_correc_to_shaper'):
                if self.is_action_checked('send_algo_to_shaper'):
                    phase_to_send = phase_to_send + field.phase_as_dwa()
                if self.is_action_checked('send_correc_to_shaper'):
                    if self._correction_phase is not None:
                        phase_to_send = phase_to_send + self._correction_phase

                self._shaper.move_abs(phase_to_send)

    def save_phase(self):

        fname = select_file(save=True, ext='h5', force_save_extension=True)
        if fname:

            correction_values = self._corrections.get_corrections()

            quad_phase_array = self._corrections.compute_focal_phase(correction_values.focal_length)
            linear_phase_array = self._corrections.compute_linear_phase(correction_values.tilt_x, correction_values.tilt_y)
            zernike_phase = self._corrections.compute_zernike_phase(correction_values.zernike)
            dte = DataToExport('Phases')
            if self._object_field is not None:
                dte.append(self._object_field.phase_as_dwa(name='Algo Phase'))

            dte.append(DataCalculated('Quadratic Phase', data=[quad_phase_array]),)
            dte.append(DataCalculated('Linear Phase', data=[linear_phase_array]),)
            dte.append(DataCalculated('Zernike Phase', data=[zernike_phase]))


            with DataToExportSaver(fname) as h5saver:
                h5saver.add_data('/', dte)


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
        self.add_toolbar('dashboard', 'Dashboard Toolbar')
        self.add_toolbar('algorithm', 'Algorithm Toolbar')

        self.mainwindow.addToolBar(self.get_toolbar('dashboard'))
        self.mainwindow.addToolBar(self.get_toolbar('algorithm'))

        self._algorithm = AlgoApp(self.dockarea, toolbar=self.get_toolbar('algorithm'))

        self._target_dockarea = gutils.DockArea()
        self._target_loader = FieldLoaderApp(self._target_dockarea,
                                             modules_manager=self.modules_manager,
                                             title='Target Field Loader')
        self._target_loader.set_loader_in_settings(
            plugin_config('target', 'default_loader')[0])

        self._input_field_dockarea = gutils.DockArea()
        self._input_field_loader = FieldLoaderApp(self._input_field_dockarea,
                                                  title='Input Field Loader')
        self._input_field_loader.set_loader_in_settings(
            plugin_config('input', 'default_loader')[0])

        self._corrections_dockarea = gutils.DockArea()
        self._corrections = Correction(self._corrections_dockarea,
                                       title='Phase Corrections')

        self.docks['image_field'] = gutils.Dock('Image Plane')
        self.docks['object_field'] = gutils.Dock('Object Plane')
        self.docks['fitness'] = gutils.Dock('Fitness')

        self.dockarea.addDock(self.docks['fitness'])
        self.dockarea.addDock(self._algorithm.docks['algo_settings'], 'bottom',
                              self.docks['fitness'])
        self.dockarea.addDock(self.docks['object_field'], 'right')
        self.dockarea.addDock(self.docks['image_field'], 'bottom', self.docks['object_field'])

        fitness_widget = QtWidgets.QWidget()
        self.fitness_viewer = Viewer0D(fitness_widget, title='Fitness')
        self.docks['fitness'].addWidget(fitness_widget)

        self.target_widget = QtWidgets.QWidget()
        self.target_widget.setLayout(QtWidgets.QHBoxLayout())
        target_area = gutils.DockArea()
        self.target_viewers = ViewerDispatcher(target_area)
        self.target_widget.layout().addWidget(target_area)
        self.target_widget.setVisible(False)

        object_area = gutils.DockArea()
        self.object_viewers = ViewerDispatcher(object_area)
        self.docks['object_field'].addWidget(object_area)

        image_area = gutils.DockArea()
        self.image_viewers = ViewerDispatcher(image_area)
        self.docks['image_field'].addWidget(image_area)

        self.intermediate_widget = QtWidgets.QWidget()
        self.intermediate_viewer = Viewer2D(self.intermediate_widget, title='Intermediate Field Intensity')

        self.other_plots_widget = QtWidgets.QWidget()
        self.other_plots_widget.setLayout(QtWidgets.QVBoxLayout())
        other_area = gutils.DockArea()
        self.other_plots_widget.layout().addWidget(other_area)
        self.other_viewers = ViewerDispatcher(other_area, title='Other Plots')

    def plot_target(self, field: Field):
        self.target_viewers.show_data(DataToExport('Target', data=[
            field.intensity_as_dwa(),
            field.amplitude_as_dwa(),
            field.phase_as_dwa(),
        ]))

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

    def show_dashboard(self):
        self.dashboard.mainwindow.setVisible(self.is_action_checked('show_dashboard'))

    def setup_actions(self):
        logger.debug('Main actions')
        self.add_action('quit', 'Quit', 'close2', "Quit program")
        self.add_action('settings', 'Plugin Settings', 'Settings',
                        'Open the plugin configuration file',
                        checkable=True)

        self.add_action('target', 'Target Selection', 'target',
                        'Open the Target FieldLoader window', checkable=True)
        self.add_action('input', 'Input Beam Selection', 'input',
                        'Open the InputBeam FieldLoader window', checkable=True)
        self.add_action('save_phase', 'Save', 'SaveAs_32',
                        'Save Phases to a file',)

        self.add_action('show_other_plots', 'Show Other Plots', 'visibility', checkable=True,
                        icon_checked='visibility_off')
        self.add_action('show_intermediate', 'Show Intermediate', 'visibility', checkable=True,
                        icon_checked='visibility_off', tip='Show Field intensity in intermediate plane')

        logger.debug('DashBoard related actions')
        self.add_widget('dashboard_label', QtWidgets.QLabel('Dashboard:'),
                        toolbar='dashboard')
        self.add_action('show_dashboard', 'Show Dashboard', 'show',
                        'Show/Hide the Dashboard window', checkable=True,
                        icon_checked='unshow', toolbar='dashboard')
        self.preset_manager = PresetManager(self.dashboard, toolbar=self.get_toolbar('dashboard'))

        self.add_action('send_algo_to_shaper', 'Algo to shaper', 'random',
                        'Send calculated phase to the control module called *Shaper*',
                        checkable=True, toolbar='dashboard')

        self.add_action('corrections', 'Corrections', 'utility2',
                        tip='Open the Utility window with focal and Zernike correction',
                        checkable=True, toolbar='dashboard')
        self.add_action('send_correc_to_shaper', 'Correction to shaper', 'random',
                        'Send correction phase to the control module called *Shaper*',
                        checkable=True, toolbar='dashboard')
        if self.dashboard is not None:
            self.add_action('add_corrections', 'Add Corrections', 'Add_Step',
                            'Add Focal and Zernike polynomials as individual actuators in Dashboard',
                        toolbar='dashboard'
                            )
    logger.debug('actions set')

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('quit', self.quit, )

        self.connect_action('settings', self.show_config)
        self.connect_action('show_other_plots', self.show_other_plots)
        self.connect_action('show_intermediate', self.show_intermediate_field)

        self.connect_action('show_dashboard', self.show_dashboard)

        self.connect_action('target', self.show_target)
        self.connect_action('input', self.show_input)

        self._algorithm.object_field_signal.connect(self.update_object)

        self._input_field_loader.field_signal.connect(self._algorithm.set_input_field)
        self._target_loader.field_signal.connect(self._algorithm.set_target_field)
        self._target_loader.field_signal.connect(self.plot_target)
        self._algorithm.algo_changed.connect(self.update_target_loader_from_algo)
        self._algorithm.fields_to_plot.connect(self.plot_fields)
        self.update_target_loader_from_algo(self._algorithm.algorithm)

        self.connect_action('corrections', self.show_corrections)
        self._corrections.phase_changed.connect(self.update_correction_phase)
        if self.dashboard is not None:
            self.connect_action('add_corrections', self.add_corrections_actuators)

        self.connect_action('save_phase', self.save_phase)

        self.intermediate_viewer.roi_select_signal.connect(self._algorithm.update_intermediate_slices)
        for viewer in (self._target_loader.amp_viewer, self._target_loader.phase_viewer):
            viewer.roi_select_signal.connect(self._algorithm.update_target_slices)

    def plot_fields(self, dte: DataToExport):
        fitness = dte.remove(dte.get_data_from_name('fitness'))
        dte_image = DataToExport('image', data=[
            dte.remove(dte.get_data_from_full_name(full_name)) for full_name in ['image/amplitude', 'image/phase']])
        dte_object = DataToExport('object', data=[
            dte.remove(dte.get_data_from_full_name(full_name)) for full_name in ['object/amplitude', 'object/phase']])
        try:
            dwa_intermediate = dte.remove(dte.get_data_from_full_name('intermediate/intensity'))
            self.intermediate_viewer.show_data(dwa_intermediate)
        except ValueError:  # means no intermediate data to plot
            pass
        self.object_viewers.show_data(dte_object)
        self.image_viewers.show_data(dte_image)
        self.fitness_viewer.show_data(fitness)
        self.other_viewers.show_data(dte)

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
        return (plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
                plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width'),
                )

    def add_corrections_actuators(self):
        try:
            if plugin_config('corrections', 'actuators', 'focal_length'):
                self.dashboard.add_move_from_extension(f'FocalLength', "FocalLength",
                                                       self._corrections,
                                                       ui_identifier='Simple')
            for n in range(plugin_config('corrections', 'zernike', 'order_max')):
                if plugin_config('corrections', 'zernike', 'actuators', f'n{n}'):
                    for m in range(-n, n+2, 2):
                        self.dashboard.add_move_from_extension(f'Zernike {n}/{m}',
                                                               "Zernike",
                                                               self._corrections,
                                                               ui_identifier = 'Simple')
                        self.dashboard.actuators_modules[-1].axis_name = f'{n}{m}'
            self.set_action_enabled("add_corrections", False)

        except Exception as e:
            logger.exception('Could not create Corrections Actuators', exc_info=e)

    def update_target_loader_from_algo(self, algo: AlgoBase):
        pixel_size = Q_(plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'pixel_size'), 'um')
        height = plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height')
        width = plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width')
        slm_size = (pixel_size * height, pixel_size * width)
        self._target_loader.update_pixels(algo.get_target_pixels_size(slm_size))

    def show_config(self, show=True):
        if show:
            config_tree = TreeFromToml(PluginConfig(), capitalize=False)
            res = config_tree.show_dialog()
            if res:
                plugin_config = PluginConfig()
            self.set_action_checked('settings', False)
            self._target_loader.update_slm(
                plugin_config('SLM', 'default_slm')[0])
            self._input_field_loader.update_slm(
                plugin_config('SLM', 'default_slm')[0])

    def show_target(self, show=True):
        self._target_dockarea.setVisible(show)
        self._target_dockarea.closeEvent = lambda event: self.get_action('target').trigger()

    def show_input(self, show=True):
        self._input_field_dockarea.setVisible(show)
        self._input_field_dockarea.closeEvent = lambda event: self.get_action('input').trigger()

    def show_other_plots(self, show=True):
        self.other_plots_widget.setVisible(show)
        self.other_plots_widget.closeEvent = lambda event: self.get_action('show_other_plots').trigger()

    def show_intermediate_field(self, show=True):
        self.intermediate_widget.setVisible(show)
        self.intermediate_widget.closeEvent = lambda event: self.get_action('show_intermediate').trigger()

    def quit(self):
        self._input_field_dockarea.close()
        self._target_dockarea.close()
        self.intermediate_widget.close()
        self.other_plots_widget.close()
        self.dockarea.parent().close()
        self.dashboard.quit_fun()


def main():
    from pymodaq_gui.qt_utils import mkQApp
    from pymodaq.utils.gui_utils.loader_utils import create_load_dashboard
    from pymodaq_gui.utils.dock import DockArea

    app = mkQApp('Optical Shaping')

    win, dashboard = create_load_dashboard()
    win.mainwindow.setVisible(False)

    win_optical = QtWidgets.QMainWindow()
    dockarea = DockArea()
    win_optical.setCentralWidget(dockarea)
    extension = OpticalShaping(dockarea, dashboard)
    win_optical.show()

    app.exec()


if __name__ == '__main__':
    main()



