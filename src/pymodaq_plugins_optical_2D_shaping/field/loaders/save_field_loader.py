import numpy as np
from pathlib import Path


from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_data.h5modules.data_saving import DataLoader
from pymodaq_gui.parameter import Parameter
from pymodaq_data import Q_

from pymodaq_plugins_optical_2D_shaping.field.utils import (FieldLoader, LoadTypeEnum,
                                                            Field)
from pymodaq_plugins_optical_2D_shaping.field.factory import LoaderFactory

resources_path = Path(__file__).parent.parent.parent.joinpath('resources')
#ferris_wheel = resources_path.joinpath('ferris_wheel_3_11_pluto_setup_4f.h5')
ferris_wheel = resources_path.joinpath('ferris_wheel_3_11_pluto_setup_2f.h5')
cat = resources_path.joinpath('cat_field.h5')

logger = set_logger(get_module_name(__file__))


@LoaderFactory.register_loader()
class SavedFieldLoader(FieldLoader):

    LOADER_NAME = 'SavedFieldLoader'
    with_physical_pixels_size = True

    params = FieldLoader.params + \
            [{'title': 'File path:', 'name': 'file', 'type': 'browsepath',
             'value': str(cat), 'filetype': True}
         ]

    def settings_changed(self, param: Parameter):
        """ Don't do anything but only use the reload button otherwise it is too complex to load both phase/amplitude
         images or only one... """
        ...


    def load(self, *args, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""

        with DataLoader(self.settings['file']) as loader:
            dwa = loader.load_data('/RawData/Data00', load_all=True)
            xaxis = dwa.get_axis_from_index(1)[0]
            yaxis = dwa.get_axis_from_index(0)[0]
            self.field = Field(
                'Image',
                amplitude=dwa[0],
                phase=dwa[1],
                pixel_sizes=(Q_(yaxis.scaling, yaxis.units),
                             Q_(xaxis.scaling, xaxis.units))
            )

        return self.field


