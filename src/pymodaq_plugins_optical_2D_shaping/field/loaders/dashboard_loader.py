import numpy as np
from pathlib import Path
from typing import Union, TYPE_CHECKING



from pymodaq_utils.logger import set_logger, get_module_name


from pymodaq_plugins_optical_2D_shaping.field.utils import (FieldLoader, LoadTypeEnum,
                                                            Field)
from pymodaq_plugins_optical_2D_shaping.field.factory import LoaderFactory

if TYPE_CHECKING:
    from pymodaq.utils.modules_manager import ModulesManager

logger = set_logger(get_module_name(__file__))




@LoaderFactory.register_loader()
class DashboardLoader(FieldLoader):

    LOADER_NAME = 'DashboardLoader'

    params = [{'title': 'Detectors', 'name': 'detectors', 'type': 'list', 'limits': []}]

    def __init__(self, **kwargs):
        super().__init__()

        self._listener = dict()
        self.field = Field()

        if 'modules_manager' in kwargs:
            self.modules_manager: ModulesManager = kwargs['modules_manager']
            self.settings.child('detectors').setLimits(self.modules_manager.detectors_name)
        else:
            logger.warning('could not load properly the Dashboard Field Loader as'
                           'no modules manager has been passed')

    def load(self, *args, load_type: LoadTypeEnum = None, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""

        detector = self.modules_manager.get_mod_from_name(self.settings['detectors'])
        dwa = detector.current_data[0]
        self.field.amplitude = dwa[0]
        return self.field
