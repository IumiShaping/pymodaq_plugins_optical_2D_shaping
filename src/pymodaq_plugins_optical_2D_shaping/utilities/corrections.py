from typing import Tuple, Union
import numpy as np
from pathlib import Path

from qtpy import QtWidgets, QtCore, QtGui

from pymodaq_gui.utils.custom_app import CustomApp

from pymodaq_plugins_optical_2D_shaping.utilities.zernike import ZernikeUI, SliderSpinBox
from pymodaq_gui.utils.widgets.widget_bkg import WidgetWithBkg


class Correction(CustomApp):


    def __init__(self, parent):
        super().__init__(parent)

        self._zernike_ui: ZernikeUI = None

        self.setup_ui()

    def setup_docks(self):
        ver_layout = QtWidgets.QVBoxLayout()
        self.parent.setLayout(ver_layout)

        widget_main_correction = QtWidgets.QWidget()
        widget_main_correction.setLayout(QtWidgets.QGridLayout())

        self.tilt_x = SliderSpinBox(value=0., bounds=(-10, 10))
        self.tilt_y = SliderSpinBox(value=0., bounds=(-10, 10))
        self.focale = SliderSpinBox(value=0., bounds=(-10, 10))

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt X'), 0, 0)
        widget_main_correction.layout().addWidget(self.tilt_x, 1, 0)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Tilt Y'), 0, 1)
        widget_main_correction.layout().addWidget(self.tilt_y, 1, 1)

        widget_main_correction.layout().addWidget(QtWidgets.QLabel('Focale'), 0, 2)
        widget_main_correction.layout().addWidget(self.focale, 1, 2)

        self.parent.layout().addWidget(widget_main_correction)

        widget_zernike = QtWidgets.QWidget()
        widget_zernike.setLayout(QtWidgets.QHBoxLayout())

        widget = WidgetWithBkg('../resources/zernike.png')
        self._zernike_ui = ZernikeUI(widget)

        self.parent.layout().addWidget(widget_zernike)
        widget_zernike.layout().addWidget(widget)



    def setup_actions(self):
        pass

    def connect_things(self):
        pass


def main():
    from pymodaq_gui.utils.utils import mkQApp

    app = mkQApp('Optical Shaping')

    widget = QtWidgets.QWidget()
    widget.show()

    correction = Correction(widget)


    app.exec()


if __name__ == '__main__':
    main()