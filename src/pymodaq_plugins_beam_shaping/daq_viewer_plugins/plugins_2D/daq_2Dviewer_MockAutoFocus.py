from pathlib import Path

import numpy as np

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq.control_modules.thread_commands import ThreadStatus

from pymodaq_plugins_beam_shaping.hardware.autofocus import Autofocus


class DAQ_2DViewer_MockAutoFocus(DAQ_Viewer_base):
    """
    """
    params = comon_parameters + [
        {'title': 'Blur', 'name': 'blur', 'type': 'int', 'value': 10},
    ]

    def ini_attributes(self):
        self.controller: Autofocus = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == 'blur':
            self.controller.blur = param.value()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        if self.is_master:
            self.controller = Autofocus()
        else:
            self.controller = controller

        self.emit_status(ThreadCommand(ThreadStatus.UPDATE_MAIN_SETTINGS,
                                       [('wait_time', ), 100, 'value']))

        info = "Autofocus initialized"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        self.dte_signal.emit(DataToExport(
            'AutoFocus',
            data=[DataFromPlugins(name='AutoFocus',
                                  data=[self.controller.grab()],
                                  dim='Data2D', labels=['AutoFocused Cat']),
                  ]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        pass
        return ''


if __name__ == '__main__':
    main(__file__)
