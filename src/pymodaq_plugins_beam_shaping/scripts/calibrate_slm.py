import numpy as np
import sys

from pymodaq.scripting import Detector, Actuator, Dashboard
from pymodaq.utils.data import DataActuator
from pymodaq_data import DataToExport
from pymodaq_gui.plotting.data_viewers import ViewerDispatcher
from pymodaq_gui.utils.utils import mkQApp
from pymodaq_gui.utils.widgets.window import make_window

from pymodaq_plugins_beam_shaping.utilities.sizing import get_slm_size





if __name__ == "__main__":
    dashboard = Dashboard()
    assert 'Shaper' in dashboard.get_devices().result()['actuators']

    qapp = mkQApp('Calibration')

    win, area = make_window(title='Calibration')
    dispatcher = ViewerDispatcher(area)

    shaper = Actuator('Shaper')
    camera = Detector('Camera')

    shaper_shape = get_slm_size()

    for ind in range(256):
        data = np.zeros(shaper_shape)
        data[:, shaper_shape[1] // 2:] = ind
        shaper.move_abs(DataActuator('Shaper', data=[data])).result()

        dte = camera.snap().result()

        dwa_ft = dte[0].ft()


        dte.append(dwa_ft.abs())

        dispatcher.show_data(dte)


    sys.exit(qapp.exec())