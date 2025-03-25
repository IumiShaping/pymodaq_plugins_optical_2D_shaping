from typing import Tuple, Union
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from qtpy import QtWidgets, QtCore, QtGui

from pymodaq_gui.utils.custom_app import CustomApp, Dock, DockArea

from pymodaq_plugins_optical_2D_shaping.utilities.zernike import (ZernikeUI, SliderSpinBox,
                                                                  ZernikeCoeffs)
from pymodaq_gui.utils.widgets.widget_bkg import WidgetWithBkg
from pymodaq_gui.parameter.pymodaq_ptypes.itemselect import ItemSelect


@dataclass()
class CorrectionValues:
    tilt_x: float = 0.
    tilt_y: float = 0.
    focale: float = 0.
    zernike: ZernikeCoeffs = ZernikeCoeffs()


class Correction(CustomApp):

    correction_changed = QtCore.Signal(CorrectionValues)

    def __init__(self, parent: DockArea):
        super().__init__(parent)

        self._zernike_ui: ZernikeUI = None
        self._zernike_coeffs = ZernikeCoeffs()

        self.setup_ui()

    def emit_corrections(self):
        self.correction_changed.emit(
            CorrectionValues(self.tilt_x.value(),
                             self.tilt_y.value(),
                             self.focale.value(),
                             self._zernike_coeffs))

    def update_zernike(self, n, m, value):
        self._zernike_coeffs.set(n, m, value)
        self.emit_corrections()

    def setup_docks(self):

        self.docks['corrections'] = Dock('Corrections')
        self.dockarea.addDock(self.docks['corrections'])

        main_widget = QtWidgets.QWidget()
        self.docks['corrections'].addWidget(main_widget)

        main_widget.setLayout(QtWidgets.QVBoxLayout())

        widget_main_correction = QtWidgets.QWidget()
        widget_main_correction.setLayout(QtWidgets.QGridLayout())

        self.tilt_x = SliderSpinBox(value=0., bounds=(-10, 10))
        self.tilt_y = SliderSpinBox(value=0., bounds=(-10, 10))
        self.focale = SliderSpinBox(value=0., bounds=(-10, 10))

        self.item_select = ItemSelect()
        self.item_select.set_value(dict(all_items=['un', 'deux', 'trois'],
                                        selected=[]))

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt X'), 0, 0)
        widget_main_correction.layout().addWidget(self.tilt_x, 1, 0)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt Y'), 0, 1)
        widget_main_correction.layout().addWidget(self.tilt_y, 1, 1)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Focale'), 0, 2)
        widget_main_correction.layout().addWidget(self.focale, 1, 2)

        widget_main_correction.layout().addWidget(self.item_select, 0, 3, 2, 1)

        main_widget.layout().addWidget(widget_main_correction)

        widget_zernike = QtWidgets.QWidget()
        widget_zernike.setLayout(QtWidgets.QHBoxLayout())

        widget = WidgetWithBkg('../resources/zernike.png')
        self._zernike_ui = ZernikeUI(widget)

        main_widget.layout().addWidget(widget_zernike)
        widget_zernike.layout().addWidget(widget)

    def setup_actions(self):
        pass

    def connect_things(self):
        self._zernike_ui.slider_changed_sig.connect(self.update_zernike)
        self.tilt_x.valueChanged.connect(self.emit_corrections)
        self.tilt_y.valueChanged.connect(self.emit_corrections)
        self.focale.valueChanged.connect(self.emit_corrections)


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