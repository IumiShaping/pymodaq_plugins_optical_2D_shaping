from typing import Tuple, Union
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from zernpy import ZernPol

from qtpy import QtWidgets, QtCore, QtGui
from zernpy.calculations.calc_psfs_check import pixel_size

from pymodaq_data import Q_
from pymodaq_data.data import DataCalculated

from pymodaq_gui.utils.custom_app import CustomApp, Dock, DockArea
from pymodaq_gui.utils.widgets.widget_bkg import WidgetWithBkg
from pymodaq_gui.parameter.pymodaq_ptypes.itemselect import ItemSelect
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_gui.utils.widgets.spinbox import SpinBox

from pymodaq_plugins_optical_2D_shaping import config as plugin_config
from pymodaq_plugins_optical_2D_shaping.utilities.zernike import (ZernikeUI, SliderSpinBox,
                                                                  ZernikeCoeffs)



@dataclass()
class CorrectionValues:
    tilt_x: float = 0.
    tilt_y: float = 0.
    focal_length: float = 0.
    zernike: ZernikeCoeffs = ZernikeCoeffs()


class Correction(CustomApp):

    correction_changed = QtCore.Signal(CorrectionValues)
    phase_changed = QtCore.Signal(DataCalculated)

    _plugin_config = plugin_config

    def __init__(self, parent: DockArea, beam_fwhm=Q_(5, 'mm')):
        super().__init__(parent)

        self._zernike_ui: ZernikeUI = None
        self._zernike_coeffs = ZernikeCoeffs()

        self._beam_fwhm = beam_fwhm
        self.zernike_values: list[np.ndarray] = []

        self.setup_ui()

    @property
    def beam_fwhm(self) -> Q_:
        """ Get/Set the beam FWHM in intensity as a Quantity"""
        return self._beam_fwhm

    @beam_fwhm.setter
    def beam_fwhm(self, fwhm: Q_):
        if fwhm.is_compatible_with(self._beam_fwhm):
            self._beam_fwhm = fwhm

    def emit_corrections(self):
        corrections = CorrectionValues(self.tilt_x.value(),
                                       self.tilt_y.value(),
                                       self.focal_length.value(),
                                       self._zernike_coeffs)

        self.correction_changed.emit(corrections)
        self.phase_changed.emit(self.compute_corrections(corrections))

    def update_zernike(self, n, m, value):
        self._zernike_coeffs.set(n, m, value)
        self.emit_corrections()

    def setup_docks(self):

        self.docks['corrections'] = Dock('Corrections')
        self.dockarea.addDock(self.docks['corrections'])

        self._toolbar = QtWidgets.QToolBar()
        self.unit_radius_sb = SpinBox(value=2.0)
        self.unit_radius_sb.setMaximumWidth(50)

        self.beam_fwhm_sb = SpinBox(value=2.0, suffix = 'mm')
        self.beam_fwhm_sb.setMaximumWidth(50)

        main_widget = QtWidgets.QWidget()
        self.docks['corrections'].addWidget(main_widget)

        main_widget.setLayout(QtWidgets.QVBoxLayout())
        main_widget.layout().addWidget(self._toolbar)
        widget_main_correction = QtWidgets.QWidget()
        widget_main_correction.setLayout(QtWidgets.QGridLayout())

        self.tilt_x = SliderSpinBox(value=0., bounds=(-10, 10))
        self.tilt_y = SliderSpinBox(value=0., bounds=(-10, 10))
        self.focal_length = SliderSpinBox(value=0., bounds=(-10, 10))

        self.item_select = ItemSelect()
        self.item_select.set_value(dict(all_items=['un', 'deux', 'trois'],
                                        selected=[]))

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt X'), 0, 0)
        widget_main_correction.layout().addWidget(self.tilt_x, 1, 0)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt Y'), 0, 1)
        widget_main_correction.layout().addWidget(self.tilt_y, 1, 1)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Focal Length'), 0, 2)
        widget_main_correction.layout().addWidget(self.focal_length, 1, 2)

        widget_main_correction.layout().addWidget(self.item_select, 0, 3, 2, 1)

        main_widget.layout().addWidget(widget_main_correction)

        widget_zernike = QtWidgets.QWidget()
        widget_zernike.setLayout(QtWidgets.QHBoxLayout())

        widget = WidgetWithBkg('../resources/zernike.png')
        self._zernike_ui = ZernikeUI(widget)

        main_widget.layout().addWidget(widget_zernike)
        widget_zernike.layout().addWidget(widget)

        self.viewer_widget = QtWidgets.QWidget()
        self.viewer2D = Viewer2D(self.viewer_widget)
        widget_zernike.layout().addWidget(self.viewer_widget)
        self.viewer_widget.setVisible(False)

    def setup_actions(self):
        self.add_action('show_phase', 'Show Phase', 'show', tip='Display the correction phase in a 2D Viewer',
                        checkable=True, toolbar=self._toolbar)
        self.add_widget('beam_fwhm_label', QtWidgets.QLabel('Beam FWHM: '), toolbar=self._toolbar)
        self.add_widget('beam_fwhm', self.beam_fwhm_sb, toolbar=self._toolbar,
                        tip='Input beam FWHM in Intensity')
        self.add_widget('unity_radius_label', QtWidgets.QLabel('Unity Radius: '), toolbar=self._toolbar)
        self.add_widget('unity_radius', self.unit_radius_sb, toolbar=self._toolbar,
                        tip='Unity radius for Zernike polynomials, '
                            'expressed as a multiple factor of the Input Beam FWHM')
        self.add_action('reset', 'Reset', 'Redo')

    def connect_things(self):
        self._zernike_ui.slider_changed_sig.connect(self.update_zernike)
        self.tilt_x.valueChanged.connect(self.emit_corrections)
        self.tilt_y.valueChanged.connect(self.emit_corrections)
        self.focal_length.valueChanged.connect(self.emit_corrections)

        self.connect_action('show_phase', self.viewer_widget.setVisible)
        self.connect_action('reset', self.reset)

    def reset(self):
        self.tilt_y.setValue(0.)
        self.tilt_x.setValue(0.)
        self.focal_length.setValue(0.)
        self._zernike_ui.reset()

    def compute_corrections(self, correction: CorrectionValues) -> DataCalculated:
        quad_phase_array = self.compute_focal_phase(correction.focal_length)
        linear_phase_array = self.compute_linear_phase(correction.tilt_x, correction.tilt_y)
        zernike_phase = self.compute_zernike_phase(correction.zernike)

        phase = DataCalculated('Correction phase',
                               data=[quad_phase_array+linear_phase_array+zernike_phase])

        if self.is_action_checked('show_phase'):
            self.viewer2D.show_data(phase)

        return phase

    def compute_zernike_phase(self, zernike: ZernikeCoeffs) -> np.ndarray[float, float]:
        """

        Parameters
        ----------
        zernike

        Returns
        -------

        """
        pixel_size = Q_(self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'pixel_size'), 'um')
        unity_radius = self.unit_radius_sb.value() * self.beam_fwhm

        polynomials = []
        amplitudes = []
        xlin, ylin = self._get_xy()
        xlin *= pixel_size
        ylin *= pixel_size
        yy, xx = np.meshgrid(ylin, xlin, indexing='ij')

        r = (np.sqrt(xx**2 + yy**2) / unity_radius).to_base_units().magnitude
        r[r>=1.] = 0.
        theta = np.angle(xx.magnitude + 1j*yy.magnitude)

        if len(self.zernike_values) != int(np.sum(np.arange(zernike.order_max+1))):
            compute = True
            self.zernike_values = []
        else:
            compute = False

        for n in range(zernike.order_max):
            for m in range(-n, n+2, 2):
                polynomials.append(ZernPol(n=n, m=m))
                if compute:
                    self.zernike_values.append(polynomials[-1].polynomial_value(r, theta))
                amplitudes.append(zernike.get(n, m))

        return np.sum(list(map(np.multiply, self.zernike_values, amplitudes)), 0)

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

    def compute_focal_phase(self, focal_value: float) -> np.ndarray[float, float]:
        """ compute the phase to send to the SLM to achieve this focal length

        Parameters
        ----------
        focal_value: float
            required focal length value in cm
        """
        if np.abs(focal_value) < 0.01:
            coeff = 0.
        else:
            focal_length = Q_(focal_value, 'cm')
            pixel_size = Q_(self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'pixel_size'), 'um')
            wavelength = Q_(self._plugin_config('wavelength_nm'), 'nm')

            coeff =  float((pixel_size ** 2 / (wavelength * focal_length) * np.pi).to_reduced_units().magnitude)

        xlin, ylin = self._get_xy()
        yquad = coeff * ylin ** 2
        xquad = coeff * xlin ** 2

        yy, xx = np.meshgrid(yquad, xquad, indexing='ij')
        return yy + xx

    def compute_linear_phase(self, tiltx: float, tilty: float) -> np.ndarray[float, float]:
        xlin, ylin = self._get_xy()
        #todo: compute the physical relation between this tilt value and a displacement in mm
        # in the focal plane

        ylin *= tilty
        xlin *= tiltx
        yy, xx = np.meshgrid(ylin, xlin, indexing='ij')

        return yy + xx

    @property
    def shape(self) -> tuple[int, int]:
        """ Get the shape of the configured SLM"""
        return (self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'height'),
                self._plugin_config('SLM', self._plugin_config('SLM', 'default_slm'), 'width'),
                )


def main():
    from pymodaq_gui.utils.utils import mkQApp

    def print_corrections(corrections: CorrectionValues):
        print(corrections)

    app = mkQApp('Optical Shaping')

    area = DockArea()
    area.show()

    correction = Correction(area)
    correction.correction_changed.connect(print_corrections)

    app.exec()


if __name__ == '__main__':
    main()