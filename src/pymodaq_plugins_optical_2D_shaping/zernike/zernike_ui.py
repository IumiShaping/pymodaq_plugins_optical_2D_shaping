from typing import Tuple, Union
import numpy as np

from pathlib import Path

from qtpy import QtWidgets, QtCore, QtGui
from skimage.transform import rescale, resize

from pymodaq_gui.managers.parameter_manager import Parameter
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_gui.utils.custom_app import CustomApp
from pymodaq_gui.utils.dock import DockArea, Dock

from pymodaq_gui.utils.widgets.slider import SliderSpinBox


class WidgetWithBkg(QtWidgets.QWidget):

    def __init__(self, bkg_path: Union[str, Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(bkg_path, str):
            bkg_path = Path(bkg_path)
            if not bkg_path.is_file():
                raise ValueError(f'Unknown background file with path: {bkg_path}')

        self._bkg_path = bkg_path
        self.setup_palette()

    def setup_palette(self):
        pixmap = QtGui.QPixmap(str(self._bkg_path))
        self.setFixedSize(pixmap.size())

        palette = QtGui.QPalette()
        palette.setBrush(palette.ColorRole.Window, QtGui.QBrush(pixmap))
        self.setPalette(palette)


class ZernikeUI(CustomApp):

    slider_changed_sig = QtCore.Signal(int, int, float)

    def __init__(self, parent):
        super().__init__(parent)

        self.order_max = 4
        self.bounds = (-10, 10)

        self.setup_ui()

    def setup_docks(self):
        self.grid_layout = QtWidgets.QGridLayout()
        self.parent.setLayout(self.grid_layout)

        self.sliders = {}
        for n in range(self.order_max):
            for m in range(-n, n+2, 2):
                self.sliders[f'{n}{m}'] = SliderSpinBox(value=0., bounds=self.bounds)
                self.sliders[f'{n}{m}'].valueChanged.connect(self.create_lambda(n, m))
                self.parent.layout().addWidget(self.sliders[f'{n}{m}'], n, m+self.order_max,
                                               QtCore.Qt.AlignmentFlag.AlignHCenter|QtCore.Qt.AlignmentFlag.AlignBottom)

    def create_lambda(self, n, m):
        return lambda: self.slider_changed(n, m)

    def slider_changed(self, n: int, m: int):
        self.slider_changed_sig.emit(n, m, self.sliders[f'{n}{m}'].value())

    def setup_actions(self):
        pass

    def connect_things(self):
        pass


def main():
    from pathlib import Path
    from pymodaq_gui.utils.utils import mkQApp

    def print_info(n, m, value):
        print(f'Slider {n},{m} has a value of  {value}')

    app = mkQApp('Optical Shaping')

    widget = WidgetWithBkg('../resources/zernike.png')
    widget.show()

    zernike_ui = ZernikeUI(widget)

    zernike_ui.slider_changed_sig.connect(print_info)

    app.exec()


if __name__ == '__main__':
    main()