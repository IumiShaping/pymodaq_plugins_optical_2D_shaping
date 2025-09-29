from typing import List, Optional, Union
from pymodaq_data.data import DataToExport


import numpy as np
from pymodaq_gui.parameter import Parameter


from pymodaq.extensions.optimizers_base.utils import OptimizerModelGeneric, individual_as_dta
from pymodaq.utils.data import DataToActuators

from pymodaq_plugins_optical_2D_shaping.hardware.autofocus import AutoFocusFactory



class AutoFocus(OptimizerModelGeneric):

    actuators_name: List[str] = []  # to be populated dynamically at instantiation
    detectors_name: List[str] = []  # to be populated dynamically at instantiation

    params = [{'title': 'Optimizing signal', 'name': 'optimizing_signal', 'type': 'group',
               'children': [
                   {'title': 'Get data', 'name': 'data_probe', 'type': 'action'},
                   {'title': 'Optimize 2Ds:', 'name': 'optimize_2d', 'type': 'itemselect',
                             'checkbox': True},
                   {'title': 'BlurMetric', 'name': 'blur_metric', 'type': 'list',
                    'limits': AutoFocusFactory.names(), 'value': AutoFocusFactory.names()[0]},
               ]},]


    def __init__(self, optimization_controller):
        self.actuators_name = optimization_controller.modules_manager.selected_actuators_name
        self.detectors_name = optimization_controller.modules_manager.selected_detectors_name
        super().__init__(optimization_controller)

        self.settings.child('optimizing_signal', 'data_probe').sigActivated.connect(
            self.optimize_from)

    def has_fitness_observable(self) -> bool:
        """ Should return True if the model defined a 0D data to be used as fitness value"""
        return len(self.settings.child('optimizing_signal', 'optimize_2d').value()['selected']) == 1

    def ini_model(self):
        pass

    def optimize_from(self):
        self.modules_manager.get_det_data_list()
        data2D = self.modules_manager.settings['data_dimensions', 'det_data_list2D']
        data2D['selected'] = data2D['all_items']
        self.settings.child('optimizing_signal', 'optimize_2d').setValue(data2D)

    def has_fitness_observable(self) -> bool:
        """ Should return True if the model defined a 0D data to be used as fitness value"""
        return True

    def ini_model(self):
        pass


    def update_settings(self, param: Parameter):
        pass

    def convert_input(self, measurements: DataToExport) -> float:
        """ Convert the measurements in the units to be fed to the Optimisation Controller

        Parameters
        ----------
        measurements: DataToExport
            data object exported from the detectors from which the model extract a float value
            (fitness) to be fed to the algorithm

        Returns
        -------
        float

        """
        data_name: str = self.settings['optimizing_signal', 'optimize_2d']['selected'][0]
        origin, name = data_name.split('/')
        return AutoFocusFactory.get(self.settings['optimizing_signal', 'blur_metric']).compute(
            measurements.get_data_from_name_origin(name, origin).data[0])


    def convert_output(self, outputs: dict[str, Union[float, np.ndarray]],
                       best_individual: Optional[dict[str, float]] = None) -> DataToActuators:
        """ Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: dict with name of the actuator as key and the value to move to as a float (or ndarray)
            output value from the controller from which the model extract a value of the same units as the actuators
        best_individual: dict[str, float]
            the coordinates of the best individual so far
        Returns
        -------
        DataToActuatorOpti: derived from DataToExport. Contains value to be fed to the actuators with a 'mode'
            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        return individual_as_dta(outputs, self.modules_manager.actuators, 'outputs', mode='abs')


