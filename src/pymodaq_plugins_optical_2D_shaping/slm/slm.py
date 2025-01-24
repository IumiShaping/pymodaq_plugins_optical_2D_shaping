from pymodaq_gui.managers.parameter_manager import ParameterManager

from pymodaq_plugins_optical_2D_shaping import config as plugin_config

class SLM(ParameterManager):

    params = [
        {'title': 'SLM:', 'name': 'slm', 'type': 'list',
         'limits': plugin_config.get_children('SLM'), 'value': plugin_config('SLM', 'default_slm')},
        {''}
    ]


