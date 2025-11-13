from pymodaq.extensions.data_mixer.model import DataMixerModel, np  # np will be used in method eval of the formula

from pymodaq_data.data import DataToExport, DataWithAxes, DataRaw
from pymodaq_gui.parameter import Parameter
from pymodaq_gui.config_saver_loader import ConfigSaverLoader

from pymodaq.extensions.data_mixer.parser import (
    extract_data_names, split_formulae, replace_names_in_formula)

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_plugins_optical_2D_shaping.hardware.autofocus import AutoFocusFactory

logger = set_logger(get_module_name(__file__))

class AutoFocus(DataMixerModel):
    params = [
        {'title': 'BlurMetric', 'name': 'blur_metric', 'type': 'list',
         'limits': AutoFocusFactory.names(), 'value': AutoFocusFactory.names()[0]},
        {'title': 'Multiply by:', 'name': 'multiply', 'type': 'float', 'value': 1.},
        {'title': 'Get Data:', 'name': 'get_data', 'type': 'bool_push', 'value': False,
         'label': 'Get Data'},
        {'title': 'Data2D:', 'name': 'data2D', 'type': 'itemselect',
         'value': dict(all_items=[], selected=[])},
    ]

    def ini_model(self):
        self.model_saver_loader = ConfigSaverLoader(self.settings, self.data_mixer.datamixer_config,
                                                    base_path=[self.__class__.__name__])
        self.show_data_list()
        self.model_saver_loader.load_config()

    def update_settings(self, param: Parameter):
        if param.name() == 'get_data':
            self.show_data_list()

    def process_dte(self, dte: DataToExport):
        dte_processed = DataToExport('Computed')
        if len(self.settings['data2D']['selected']) > 0:
            dwa = dte.get_data_from_full_name(
                self.settings['data2D']['selected'][0]
            )
            dte_processed.append(
                DataRaw('AutoFocus',
                        data=[
                            np.atleast_1d([
                                self.settings['multiply'] * AutoFocusFactory.get(
                                self.settings['blur_metric']).compute(dwa[0])])
                        ]))
            dte_processed.append(dwa)

        return dte_processed


    def show_data_list(self):
        dte = self.modules_manager.get_det_data_list()
        data_list2D = dte.get_full_names('data2D')
        self.settings.child('data2D').setValue(dict(all_items=data_list2D, selected=[]))




