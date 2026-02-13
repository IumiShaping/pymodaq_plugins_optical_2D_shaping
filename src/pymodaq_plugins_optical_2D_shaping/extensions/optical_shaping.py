import numpy as np
from qtpy import QtWidgets, QtCore
from pathlib import Path

from pymodaq_plugins_optical_2D_shaping.algorithms.utils import ApplyMaskTo
from pymodaq_utils import utils as utils
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.config import Config
from pymodaq.utils.data import DataToExport, DataCalculated, DataActuator, DataDim
from pymodaq_utils.math_utils import greater2n
from pymodaq_data.h5modules.data_saving import DataToExportSaver

from pymodaq_gui.plotting.data_viewers.viewer0D import Viewer0D
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_gui.plotting.data_viewers.viewer import ViewerDispatcher
from pymodaq_gui.utils.file_io import select_file
from pymodaq_gui import utils as gutils
from pymodaq_gui.utils.widgets.tree_toml import TreeFromToml
from pymodaq_gui.utils.layout import save_layout_state, load_layout_state
from pymodaq_gui.parameter.ioxml import parameter_to_xml_string

from pymodaq.extensions.custom_ext import CustomExt
from pymodaq.utils.config import get_set_layout_path

from pymodaq_plugins_optical_2D_shaping.utils import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.algorithms.algorithm_app import AlgoApp, AlgoBase
from pymodaq_plugins_optical_2D_shaping.field.field_loader_app import FieldLoaderApp, Field, Q_
from pymodaq_plugins_optical_2D_shaping.utilities.corrections import Correction
from pymodaq_plugins_optical_2D_shaping.algorithms import AlgorithmFactory, AlgoBase
from pymodaq_plugins_optical_2D_shaping.utilities import sizing


logger = set_logger(get_module_name(__file__))
layout_path = get_set_layout_path()

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

        self._target_field_loader: FieldLoaderApp = None
        self._input_field_loader: FieldLoaderApp = None

        self._object_field: Field = None
        self._correction_phase: DataCalculated = None

        self.object_viewers: ViewerDispatcher = None
        self.image_viewers: ViewerDispatcher = None
        self.intermediate_viewer: Viewer2D = None
        self.other_viewers: ViewerDispatcher = None


        self._algorithm: AlgoApp = None

        self._corrections: Correction = None

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
        return self._target_field_loader.field

    def do_things_after_init(self):
        self._input_field_loader.load_field()
        self._target_field_loader.load_field()

    def get_slm_slices(self) -> tuple[slice, slice]:
        """ get slices to apply to fields to get only pixels corresponding to the SLM"""
        shape = self.input_field.shape
        center = tuple(np.array(shape) // 2)
        size = self.slm_shape
        slices = (slice(max(0, center[0] - size[0] // 2), min(shape[0], center[0] + size[0] // 2)),
                  slice(max(0, center[1] - size[1] // 2), min(shape[1], center[1] + size[1] // 2)))
        return slices

    def do_things_after_preset_set(self, preset_name: str):
        super().do_things_after_preset_set(preset_name)
        if self.modules_manager is not None and 'Shaper' in self.modules_manager.actuators_name:
            self._shaper = self.modules_manager.get_mod_from_name('Shaper', 'act')
        else:
            self._shaper = None

    def update_object(self, field: Field):
        """ field contains here the object field"""

        if field is None:
            field = self.input_field

        self._object_field = field

        if self._shaper is not None:
            phase_to_send = 0.
            if self.is_action_checked('send_algo_to_shaper') or self.is_action_checked('send_correc_to_shaper'):
                if self.is_action_checked('send_algo_to_shaper'):
                    phase_to_send = phase_to_send + field.phase_as_dwa()[0][*self.get_slm_slices()]
                if self.is_action_checked('send_correc_to_shaper'):
                    if self._correction_phase is not None:
                        phase_to_send = phase_to_send + self._correction_phase[0][*self.get_slm_slices()]

                self._shaper.move_abs(DataActuator('phase', data=sizing.unbin_to_real_slm(phase_to_send)))

    def save(self, fname: Path = None):
        """ Save fields: input, object, image, target into a hdf5 file together with settings/metadata

        Also save the phase sent to the SLM (cropped/unbined to the SLM shape)

        """
        if fname is None:
            fname = select_file(save=True, ext='h5', force_save_extension=True)
        if fname:

            correction_values = self._corrections.get_corrections()

            quad_phase_array = self._corrections.compute_focal_phase(correction_values.focal_length)
            linear_phase_array = self._corrections.compute_linear_phase(correction_values.tilt_x, correction_values.tilt_y)
            zernike_phase = self._corrections.compute_zernike_phase(correction_values.zernike)
            dte = DataToExport('SLM Phases')
            if self._object_field is not None:
                dte.append(self._object_field.phase_as_dwa(name='Algo Phase').isig[*self.get_slm_slices()])

            dte.append(DataCalculated('Quadratic Phase', data=[quad_phase_array[*self.get_slm_slices()]]),)
            dte.append(DataCalculated('Linear Phase', data=[linear_phase_array[*self.get_slm_slices()]]),)
            dte.append(DataCalculated('Zernike Phase', data=[zernike_phase[*self.get_slm_slices()]]))

            with DataToExportSaver(fname, metadata={'settings': plugin_config.to_xml_string()}) as h5saver:
                group = h5saver.add_data_group('/RawData', DataDim.Data2D, title='SLM Phases', group_name='SLM')
                h5saver.add_data(group, dte)

            self._input_field_loader.save_field(fname, where='/RawData', group_name='Input', title='Input Field')
            self._target_field_loader.save_field(fname, where='/RawData', group_name='Target', title='Target Field')

            self.algorithm.save(fname, where='/RawData', group_name='Algorithm', title='Algorithm')


    def update_correction_phase(self, dwa: DataCalculated):
        self._correction_phase = dwa
        self.update_object(self._object_field)

    @property
    def algorithm(self):
        return self._algorithm

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
        self.create_dashboard_toolbar(add_break=False)

        self._target_dockarea = gutils.DockArea()
        self._target_field_loader = FieldLoaderApp(self._target_dockarea,
                                             modules_manager=self.modules_manager,
                                             title='Target Field Loader')
        self._target_field_loader.set_loader_in_settings(
            plugin_config('target', 'default_loader')[0])


        self._input_field_dockarea = gutils.DockArea()
        self._input_field_loader = FieldLoaderApp(self._input_field_dockarea,
                                                  title='Input Field Loader')
        self._input_field_loader.set_loader_in_settings(
            plugin_config('input', 'default_loader')[0])
        self._input_field_loader.updated_slm(
            plugin_config('SLM', 'default_slm')[0])
        self._input_field_loader.update_apply_mask(size=self.slm_shape, apply=True)

        self._corrections_dockarea = gutils.DockArea()
        self._corrections = Correction(self._corrections_dockarea,
                                       title='Phase Corrections')

        self.docks['image_field'] = gutils.Dock('Image Plane')
        self.docks['object_field'] = gutils.Dock('Object Plane')
        self.docks['metrics'] = gutils.Dock('Fitness')
        self.dockarea.addDock(self.docks['metrics'], 'left')
        self.dockarea.addDock(self.docks['object_field'], 'right')
        self.dockarea.addDock(self.docks['image_field'], 'bottom', self.docks['object_field'])


        metrics_area = gutils.DockArea()
        self.metrics_viewer = ViewerDispatcher(metrics_area, title='Metrics', direction='bottom')
        self.docks['metrics'].addWidget(metrics_area)

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

    def setup_actions(self):
        logger.debug('Main actions')
        self.add_action('save', 'Save', 'save_as', 'Save Everything to a file')
        self.add_action('target', 'Target Selection', 'target',
                        'Open the Target FieldLoader window', checkable=True,
                        icon_checked_color=self.get_theme().green)
        self.add_action('input', 'Input Beam Selection', 'input',
                        'Open the InputBeam FieldLoader window', checkable=True,
                        icon_checked_color=self.get_theme().green)

        self.add_action('show_other_plots', 'Show Other Plots', 'visibility', checkable=True,
                        icon_checked='visibility_off')
        self.add_action('show_intermediate', 'Show Intermediate', 'visibility', checkable=True,
                        icon_checked='visibility_off', tip='Show Field intensity in intermediate plane')

        self.get_toolbar('dashboard').addSeparator()
        self.add_action('send_algo_to_shaper', 'Algo to shaper', 'blur_off',
                        icon_color=self.get_theme().red,
                        tip='Send calculated phase to the control module called *Shaper*',
                        checkable=True, toolbar='dashboard',
                        icon_checked='blur_on', icon_checked_color=self.get_theme().green)

        self.add_action('corrections', 'Corrections', 'build_circle',
                        tip='Open the Utility window with focal and Zernike correction',
                        checkable=True, toolbar='dashboard')
        self.add_action('send_correc_to_shaper', 'Correction to shaper', 'deblur',
                        'Send correction phase to the control module called *Shaper*',
                        checkable=True, toolbar='dashboard',
                        icon_color=self.get_theme().red,
                        icon_checked_color=self.get_theme().green)

        if self.dashboard is not None:
            self.add_action('add_corrections', 'Add Corrections', 'add_circle',
                            'Add Focal and Zernike polynomials as individual actuators in Dashboard',
                        toolbar='dashboard'
                            )

        logger.debug('actions set')

    def do_things_after_ui_setup(self):
        self.mainwindow.removeToolBarBreak(self.get_toolbar('dashboard'))

        self._algorithm = AlgoApp(self.dockarea, toolbar=self.toolbar)
        self.mainwindow.insertToolBarBreak(self.toolbar)
        self.dockarea.addDock(self._algorithm.docks['algo_settings'], 'left')
        self._algorithm.algo_changed.connect(self.update_target_loader_from_algo)
        self._algorithm.fields_to_plot.connect(self.plot_fields)
        self.update_target_loader_from_algo(self._algorithm.algorithm)
        self._algorithm.object_field_signal.connect(self.update_object)
        self._input_field_loader.field_signal.connect(self._algorithm.set_input_field)
        self._target_field_loader.field_signal.connect(self._algorithm.set_target_field)

        self.intermediate_viewer.roi_select_signal.connect(
            lambda roi: self._algorithm.update_intermediate_slices(roi.to_slices()))
        self.show_set_target_roi_select()

        for viewer in (self._target_field_loader.amp_viewer, self._target_field_loader.phase_viewer):
            viewer.roi_select_signal.connect(lambda roi: self._algorithm.update_target_slices(roi.to_slices()))

        if layout_path.joinpath('shaping.dock').is_file():
            load_layout_state(self.dockarea, layout_path.joinpath('shaping.dock'))

    def show_set_target_roi_select(self):
        slices = self._algorithm.constrains_slices(eval(self._algorithm.settings[ApplyMaskTo.TARGET, 'slices']))
        self._target_field_loader.amp_viewer.view.set_action_checked('ROIselect', True)
        pos = [slices[0].start, slices[1].start]
        size = [slices[0].stop - slices[0].start, slices[1].stop - slices[0].start]
        self._target_field_loader.amp_viewer.view.show_ROI_select(
            size=size[::-1],
            pos=pos[::-1])

    def connect_things(self):
        logger.debug('connecting things')
        self.connect_action('show_other_plots', self.show_other_plots)
        self.connect_action('show_intermediate', self.show_intermediate_field)

        self.connect_action('target', self.show_target)
        self.connect_action('input', self.show_input)

        self._target_field_loader.field_signal.connect(self.plot_target)

        self.connect_action('corrections', self.show_corrections)
        self._corrections.phase_changed.connect(self.update_correction_phase)
        if self.dashboard is not None:
            self.connect_action('add_corrections', self.add_corrections_actuators)

        self.connect_action('save', lambda: self.save())
        self.config_changed.connect(self.do_things_after_config_changed)

    def plot_fields(self, dte: DataToExport):
        metrics = dte.remove(dte.get_data_from_name('metrics'))
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
        self.metrics_viewer.show_data(metrics.split_as_dte('Metrics'))
        self.other_viewers.show_data(dte)

    def show_corrections(self, show=True):
        self._corrections_dockarea.setVisible(show)
        self._corrections_dockarea.closeEvent = lambda event: self.set_action_checked('corrections', False)

    @property
    def slm_shape(self) -> tuple[int, int]:
        """ Get the shape of the configured SLM"""
        return sizing.get_effective_slm_size()

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
        pixel_size = Q_(sizing.get_effective_slm_pixel_size(), 'um')
        needed_pixel_size = sizing.get_effective_needed_field_size()

        field_size = (pixel_size * needed_pixel_size[0], pixel_size * needed_pixel_size[1])
        self._target_field_loader.update_pixels(algo.get_target_pixels_size(field_size))

    def do_things_after_config_changed(self):
        self._input_field_loader.updated_slm(
            plugin_config('SLM', 'default_slm')[0])
        self._input_field_loader.update_apply_mask(size=self.slm_shape, apply=True)
        self._input_field_loader.loader.load_field(notify=True)

        self._target_field_loader.updated_slm(
            plugin_config('SLM', 'default_slm')[0])
        self.update_target_loader_from_algo(self.algorithm.algorithm) #will reload the target

    def show_target(self, show=True):
        self._target_dockarea.setVisible(show)
        if show:
            self._target_dockarea.showMaximized()
        self._target_dockarea.closeEvent = lambda event: self.get_action('target').trigger()

    def show_input(self, show=True):
        self._input_field_dockarea.setVisible(show)
        if show:
            self._input_field_dockarea.showMaximized()
        self._input_field_dockarea.closeEvent = lambda event: self.get_action('input').trigger()

    def show_other_plots(self, show=True):
        self.other_plots_widget.setVisible(show)
        self.other_plots_widget.closeEvent = lambda event: self.get_action('show_other_plots').trigger()

    def show_intermediate_field(self, show=True):
        self.intermediate_widget.setVisible(show)
        if show:
            self.intermediate_widget.showMaximized()
        self.intermediate_widget.closeEvent = lambda event: self.get_action('show_intermediate').trigger()

    def quit_fun(self):

        save_layout_state(self.dockarea, file=layout_path.joinpath('shaping.dock'))

        self._input_field_dockarea.close()
        self._target_dockarea.close()
        self.intermediate_widget.close()
        self.other_plots_widget.close()
        QtWidgets.QApplication.processEvents()  # allows the dashboard modules to close properly

        super().quit_fun()


def main():
    import sys
    from pymodaq_gui.qt_utils import mkQApp
    from pymodaq.dashboard import create_load_dashboard
    from pymodaq.utils.gui_utils.loader_utils import create_extension
    app = mkQApp('OpticalShaping')

    win, dashboard = create_load_dashboard()
    win.mainwindow.setVisible(False)

    win_ext, scan = create_extension(dashboard, OpticalShaping)
    win_ext.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()



