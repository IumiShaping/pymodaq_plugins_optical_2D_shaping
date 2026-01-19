from typing import Tuple, Union
import numpy as np
from qtpy import QtWidgets, QtCore
from skimage.transform import rescale, resize
from scipy.ndimage import gaussian_filter

from pymodaq_utils.config import Config
from pymodaq_utils.math_utils import normalize

from pymodaq_data import DataToExport, DataCalculated
from pymodaq_data.h5modules.saving import H5SaverLowLevel
from pymodaq_data.h5modules.data_saving import DataSaverLoader

from pymodaq_gui.managers.parameter_manager import Parameter
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_gui.utils.custom_app import CustomApp
from pymodaq_gui.utils.dock import DockArea, Dock
from pymodaq_gui.utils.file_io import select_file
from pymodaq_gui.parameter.ioxml import parameter_to_xml_string
from pymodaq_plugins_optical_2D_shaping import config as plugin_config
from pymodaq_plugins_optical_2D_shaping.field import Field, Q_, LoaderFactory, FieldLoader

from pymodaq_gui.managers.roi_manager import ROI2D_TYPES, ROI


config_utils = Config()
field_loader_factory = LoaderFactory(

)

class FieldLoaderApp(CustomApp):
    settings_name = "LoaderAppSettings"
    params = [
        {'title': 'Loader', 'name': 'loader', 'type': 'list',
         'limits': field_loader_factory.field_loaders,
         'value': field_loader_factory.field_loaders[0]},
        {'title': 'Reload:', 'name': 'reload', 'type': 'bool_push', 'label': 'Reload!',
         'value': False},
        {'title': 'Save:', 'name': 'save', 'type': 'bool_push', 'label': 'Save Field!',
         'value': False},
        {'title': 'Initial size', 'name': 'ini_size', 'type': 'group', 'children': [
            {'title': 'Height', 'name': 'height', 'type': 'int', 'value': 0, 'readonly': True},
            {'title': 'Width', 'name': 'width', 'type': 'int', 'value': 0, 'readonly': True},
        ]},
        {'title': 'Needed size', 'name': 'needed_size', 'type': 'group', 'children': [
            {'title': 'Pixel Height (µm)', 'name': 'pixel_height', 'type': 'float',
             'value': plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'pixel_size'),
             'readonly': False},
            {'title': 'Pixel Width (µm)', 'name': 'pixel_width', 'type': 'float',
             'value': plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'pixel_size'),
             'readonly': False},
            {'title': 'Height', 'name': 'height', 'type': 'int',
             'value': plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
             'readonly': True},
            {'title': 'Width', 'name': 'width', 'type': 'int',
             'value': plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width'),
             'readonly': True},
            {'title': 'Show on Viewer', 'name': 'show_needed_area', 'type': 'bool_push',
             'value': True,},
        ]},
        {'title': 'utils', 'name': 'utils', 'type': 'group', 'children': [
            {'title': 'Show field', 'name': 'show_field', 'type': 'bool_push', 'value': True},

            {'title': 'Flip ud', 'name': 'flipud', 'type': 'bool', 'value': False},
            {'title': 'Flip lr', 'name': 'fliplr', 'type': 'bool', 'value': False},
            {'title': 'Sizing', 'name': 'sizing', 'type': 'group', 'children': [
                {'title': 'Scaling', 'name': 'scaling', 'type': 'float', 'value': 1., 'max': 1.},
                {'title': 'Do Scaling', 'name': 'do_scaling', 'type': 'bool', 'value': True},
                {'title': 'Keep aspect ratio', 'name': 'aspect_ratio', 'type': 'bool',
                 'value': True},
                {'title': 'Height', 'name': 'height', 'type': 'int', 'value': 0, 'readonly': True},
                {'title': 'Width', 'name': 'width', 'type': 'int', 'value': 0, 'readonly': True},
            ]},
            {'title': 'Smoothing', 'name': 'smoothing', 'type': 'group', 'children': [
                {'title': 'Apply:', 'name': 'apply_smoothing', 'type': 'bool', 'value': False},
                {'title': 'Sigma X (pxl)', 'name': 'sigma_x', 'type': 'int', 'value': 10,},
                {'title': 'Sigma Y (pxl)', 'name': 'sigma_y', 'type': 'int', 'value': 10, },
            ]},
            {'title': 'Masking', 'name': 'masking', 'type': 'group', 'children': [
                {'title': 'Mask type', 'name': 'mask_type', 'type': 'list',
                 'limits': ROI2D_TYPES, 'value': ROI2D_TYPES[0]},
                {'title': 'Show On', 'name': 'show_on', 'type': 'list',
                 'value': 'Amplitude', 'limits': ['Amplitude', 'Phase']},
                {'title': 'Show Mask', 'name': 'show_mask', 'type': 'bool', 'value': False},
                {'title': 'Apply Mask', 'name': 'apply_mask', 'type': 'bool', 'value': False},
            ]}
        ]},
    ]

    field_signal = QtCore.Signal(Field)

    def __init__(self, dockarea: DockArea, title: str = None, **kwargs):
        super().__init__(parent=dockarea, title=title)
        self._main_widget: QtWidgets.QWidget = None
        self.field_widget: QtWidgets.QWidget = None
        self.settings_widget: QtWidgets.QWidget = None
        self._loader_settings_widget: QtWidgets.QWidget = None

        self.amp_viewer: Viewer2D = None
        self.phase_viewer: Viewer2D = None

        self.mask: ROI = None

        self._field_loader: FieldLoader = None

        self._ini_field = Field()
        self.field = Field()

        self.setup_ui()

        self.loader_kwargs = kwargs

        self.loader = field_loader_factory.field_loaders[0]

    def update_pixels(self, pixel_sizes: Tuple[Q_, Q_]):
        self.settings.child('needed_size', 'pixel_height').setValue(pixel_sizes[0].m_as('um'))
        self.settings.child('needed_size', 'pixel_width').setValue(pixel_sizes[1].m_as('um'))
        self.load_field()

    def value_changed(self, param: Parameter):
        if param.name() == 'loader':
            self.loader = param.value()

        elif param.name() in ('flipud', 'fliplr', 'do_scaling', 'scaling', 'aspect_ratio', 'apply_mask',
                              'apply_smoothing', 'sigma_y', 'sigma_x'
            ):
            self.field = self._ini_field.deepcopy()
            self.update_final_size()
            self.update_viewers()

        elif param.name() == 'show_field':
            self.field_widget.setVisible(param.value())

        elif param.name() == 'reload':
            if param.value():
                self.load_field()
                param.setValue(False)

        elif param.name() == 'show_needed_area':
            self.show_roi_target(param.value())

        elif param.name() == 'show_mask':
            if self.settings['utils', 'masking', 'show_on'] == 'Amplitude':
                viewer = self.amp_viewer
            else:
                viewer = self.phase_viewer
            if param.value():
                viewer.roi_manager.add_roi_programmatically(
                    self.settings['utils', 'masking', 'mask_type'])
                self.mask = viewer.roi_manager.get_roi_from_index(0)
                self.mask.sigRegionChangeFinished.connect(
                    lambda : self.value_changed(self.settings.child('utils', 'masking', 'apply_mask')))
            else:
                viewer.roi_manager.remove_roi_programmatically(0)
                self.mask.sigRegionChangeFinished.disconnect()
                self.mask = None

        elif param.name() == 'save':
            if param.value():
                self.save_field()
                param.setValue(False)

    def save_field(self):
        pass
        file_name = select_file(start_path=config_utils('data_saving','h5file', 'save_path'),
                                save=True, ext='h5')  # see daq_utils
        if file_name != '':
            dwa = DataCalculated('Field',
                                 data=[
                                     self.field.amplitude,
                                     self.field.phase
                                 ],
                                 labels=['Amplitude', 'Phase'],
                                 axes = self.field.get_axes())

            settings_all = [parameter_to_xml_string(self.settings),
                            parameter_to_xml_string(self._field_loader.settings)]
            settings_str = b'<All_settings title="All Settings" type="group">'
            for set in settings_all:
                if len(settings_str + set) < 60000:
                    # size limit for any object header (including all the other attributes) is 64kb
                    settings_str += set
                else:
                    break
            settings_str += b'</All_settings>'

            with DataSaverLoader(file_name, new_file=True) as saver:
                saver.add_data('/RawData/', dwa, settings=settings_str)


    def update_slm(self, slm_default_name: str):
        self.settings.child('needed_size', 'height').setValue(
            plugin_config('SLM', slm_default_name, 'height'))
        self.settings.child('needed_size', 'width').setValue(
            plugin_config('SLM', slm_default_name, 'width'))

    def show_roi_target(self, show=True):
        self.amp_viewer.roi_target.setVisible(show)
        self.phase_viewer.roi_target.setVisible(show)

    def update_viewers(self):
        needed_shape = (self.settings['needed_size', 'height'],
                        self.settings['needed_size', 'width'],
                        )
        if self.field is not None:
            self.amp_viewer.show_data(self.field.amplitude_as_dwa())
            self.phase_viewer.show_data(self.field.phase_as_dwa())

            pixels_magnitude = np.array([pixel.to_base_units().magnitude for pixel in self.field.pixels_sizes])
            self.amp_viewer.move_roi_target(
                (0, 0),
                (np.array(needed_shape) * pixels_magnitude)[::-1])
            self.phase_viewer.move_roi_target(
                (0, 0),
                (np.array(needed_shape) * pixels_magnitude)[::-1])

    def set_loader_in_settings(self, loader_name: str):
        if loader_name in self.settings.child('loader').opts['limits']:
            self.settings.child('loader').setValue(loader_name)

    def threshold_phase(self, field: Field, threshold=1e-9):
        """ Apply a threshold around 0 and pi to avoid numerical errors"""

        field.phase[np.pi - np.abs(field.phase) < threshold] = np.pi
        field.phase[np.abs(field.phase) < threshold] = 0

    def load_field(self, *args, **kwargs):
        notify = kwargs.pop('notify', False)
        self._ini_field = self._field_loader.load_field(*args, notify=notify, **kwargs)
        self.apply_modifications(self._ini_field)

    def apply_modifications(self, field: Field):
        if field is not None:
            self.threshold_phase(field)
            self.field = field.deepcopy()
            self.update_ini_size()
            self.update_final_size()
            self.update_viewers()

            self.field_signal.emit(self.field)

    def update_field(self, field: Field):
        self._ini_field = field
        self.apply_modifications(field)

    def update_ini_size(self):
        self.settings.child('ini_size', 'height').setValue(self._ini_field.shape[0])
        self.settings.child('ini_size', 'width').setValue(self._ini_field.shape[1])

    def update_final_size(self):
        pixel_ratio = self.settings['needed_size', 'pixel_width'] / self.settings['needed_size', 'pixel_height']
        needed_shape = (self.settings['needed_size', 'height'],
                         self.settings['needed_size', 'width'])
        if self.settings['utils', 'flipud']:
            self.field.field = np.flipud(self.field.field)
        if self.settings['utils', 'fliplr']:
            self.field.field = np.fliplr(self.field.field)

        _field_temp = self.field.deepcopy()
        if self._field_loader.with_physical_pixels_size:
            self.field.amplitude = _field_temp.amplitude
            self.field.phase = _field_temp.phase
        else:
            self.field.amplitude = rescale(_field_temp.amplitude, (pixel_ratio, 1))
            self.field.phase = rescale(_field_temp.phase, (pixel_ratio, 1))

        self.field.calibrate_axes((Q_(self.settings['needed_size', 'pixel_height'], 'um'),
                                   Q_(self.settings['needed_size', 'pixel_width'], 'um')))

        if self.settings['utils', 'sizing', 'do_scaling']:
            if self.settings['utils', 'sizing', 'aspect_ratio']:
                ratio = np.min(np.array(needed_shape) / np.array(self.field.shape))
            else:
                ratio = np.array(needed_shape) / np.array(self.field.shape)
            _field_temp = self.field.deepcopy()

            self.field.amplitude = self.rescale_normalize(_field_temp.amplitude,
                                                          ratio * self.settings['utils', 'sizing', 'scaling'])
            self.field.phase = self.rescale_normalize(_field_temp.phase,
                                                      ratio * self.settings['utils', 'sizing', 'scaling'])

        self.settings.child('utils', 'sizing', 'height').setValue(self.field.shape[0])
        self.settings.child('utils', 'sizing', 'width').setValue(self.field.shape[1])

        npad = self.get_npad_between(needed_shape, self.field.shape)
        if np.any(np.array(npad)):
            _field_temp = self.field.deepcopy()
            self.field.amplitude = np.pad(_field_temp.amplitude, npad)
            self.field.phase = np.pad(_field_temp.phase, npad)

        if self.settings['utils', 'masking', 'apply_mask'] and self.mask is not None:
            QtWidgets.QApplication.processEvents()
            if self.settings['utils', 'masking', 'show_on'] == 'Amplitude':
                viewer = self.amp_viewer
            else:
                viewer = self.phase_viewer
            slices, tr = self.mask.getArraySlice(self.field.amplitude,
                                                 viewer.view.get_image_item(),
                                                 returnSlice=True)

            mask_amp = self.mask.getArrayRegion(self.field.amplitude,
                                                viewer.view.get_image_item())
            amplitude = np.zeros(self.field.amplitude.shape)
            slices_sure = []
            for ind, sl in enumerate(slices):
                slices_sure.append(slice(sl.start, sl.start + mask_amp.shape[ind]))

            amplitude[*slices_sure] = mask_amp
            self.field.amplitude = amplitude

            mask_phase = self.mask.getArrayRegion(self.field.phase,
                                                  viewer.view.get_image_item())
            phase = np.zeros(self.field.amplitude.shape)
            phase[*slices_sure] = mask_phase
            self.field.phase = phase

        if self.settings['utils', 'smoothing', 'apply_smoothing']:
            self.field.amplitude = gaussian_filter(self.field.amplitude, sigma=(
                self.settings['utils', 'smoothing', 'sigma_y'],
                self.settings['utils', 'smoothing', 'sigma_x']
            ))
            self.field.phase = gaussian_filter(self.field.phase, sigma=(
                self.settings['utils', 'smoothing', 'sigma_y'],
                self.settings['utils', 'smoothing', 'sigma_x']
            ))

        #print(self.field.shape)

    def rescale_normalize(self, array_in: np.ndarray, ratio) -> np.ndarray:
        """ Rescale and renormalize the output array to have the same intensity dynamic as the input array"""
        array_out = rescale(array_in, ratio)
        array_normalized = normalize(array_out) * (np.max(array_in) - np.min(array_in)) + np.min(array_in)
        if np.any(np.isnan(normalize(array_normalized))):  # generate nan is the array is constant
            return array_out
        else:
            return array_normalized


    def get_npad_between(self, first_shape, second_shape):
        """ Get the padding necessary to match object shape and image shape

        If positive, the image shape is bigger than the object
        If negative, the object shape is bigger than the image
        """
        npad_before = ((np.array(first_shape) -
                        np.array(second_shape)) // 2).astype(int)
        npad_after = (np.array(first_shape) -
                        np.array(second_shape)) - npad_before
        return (npad_before[0], npad_after[0]), (npad_before[1], npad_after[1])

    @property
    def loader(self):
        return self._field_loader

    @loader.setter
    def loader(self, loader_name: str):
        try:
            self._field_loader: FieldLoader =\
                field_loader_factory.get_loader(loader_name)(
                    parent=self,
                    **self.loader_kwargs
                )

            while True:
                child = self._loader_settings_widget.layout().takeAt(0)
                if not child:
                    break
                child.widget().deleteLater()
                QtWidgets.QApplication.processEvents()

            self._loader_settings_widget.layout().addWidget(self._field_loader.settings_tree)
            self._field_loader.register_listener(self)

        except ValueError as e:
            pass

    def setup_docks(self):
        self.docks['target'] = Dock('Target')
        self._main_widget = QtWidgets.QWidget()
        self._main_widget.setLayout(QtWidgets.QHBoxLayout())

        self.field_widget = QtWidgets.QWidget()
        self.field_widget.setLayout(QtWidgets.QHBoxLayout())
        field_widget_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.field_widget.layout().addWidget(field_widget_splitter)

        amp_widget = QtWidgets.QWidget()
        self.amp_viewer = Viewer2D(amp_widget)
        self.amp_viewer.view.get_action('legend').trigger()

        phase_widget = QtWidgets.QWidget()
        self.phase_viewer = Viewer2D(phase_widget)
        self.phase_viewer.view.get_action('legend').trigger()

        self.show_roi_target(self.settings['needed_size', 'show_needed_area'])

        field_widget_splitter.addWidget(amp_widget)
        field_widget_splitter.addWidget(phase_widget)

        self.dockarea.addDock(self.docks['target'])

        self.settings_widget = QtWidgets.QWidget()
        self.settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self.settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.settings_widget.layout().addWidget(splitter)
        splitter.addWidget(self.settings_tree)

        self._loader_settings_widget = QtWidgets.QWidget()
        self._loader_settings_widget.setLayout(QtWidgets.QVBoxLayout())
        self._loader_settings_widget.layout().setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._loader_settings_widget)

        self._main_widget.layout().addWidget(self.settings_widget)
        self.settings_tree.setMinimumWidth(300)
        self.settings_tree.setMinimumHeight(150)
        self._main_widget.layout().addWidget(self.field_widget)
        self.docks['target'].addWidget(self._main_widget)

    def setup_actions(self):
        ...

    def connect_things(self):
        ...


def main():
    from pathlib import Path
    from pymodaq_gui.qt_utils import mkQApp

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
