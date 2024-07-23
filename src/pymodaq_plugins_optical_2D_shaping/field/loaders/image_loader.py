import numpy as np
from pathlib import Path
from typing import Union

from skimage.io import imread
from skimage.color import rgb2gray

from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.gui_utils.file_io import select_file
from pymodaq.utils.enums import enum_checker
from pymodaq.utils import math_utils as mutils

from pymodaq_plugins_optical_2D_shaping.field.utils import (FieldLoader, LoadTypeEnum,
                                                            Field)
from pymodaq_plugins_optical_2D_shaping.field.factory import LoaderFactory

resources_path = Path(__file__).parent.parent.parent.joinpath('resources')
cheshire_cat_path = resources_path.joinpath('cheshirecat_rect.png')


logger = set_logger(get_module_name(__file__))


@LoaderFactory.register_loader()
class ImageFileLoader(FieldLoader):

    LOADER_NAME = 'ImageFileLoader'

    params = FieldLoader.params + \
        [{'title': 'Amplitude File path:', 'name': 'amp_target_file',
          'type': 'browsepath', 'value': str(cheshire_cat_path), 'filetype': True},
         {'title': 'Phase File path:', 'name': 'phase_target_file', 'type': 'browsepath',
          'value': '', 'filetype': True},
         ]

    def value_changed(self, param: Parameter):
        super().value_changed(param)

        if param.name() == 'amp_target_file':
            if Path(param.value()).is_file():
                self.load_image_from_name(fname=Path(param.value()),
                                          load_type=LoadTypeEnum.AMPLITUDE)
                logger.info(f'Amplitude Image loaded from {param.value()}')

        elif param.name() == 'phase_target_file':
            if Path(param.value()).is_file():
                self.load_image_from_name(fname=Path(param.value()),
                                          load_type=LoadTypeEnum.PHASE)
                logger.info(f'Phase Image loaded from {param.value()}')

    def load_image(self, load_type=LoadTypeEnum.AMPLITUDE):
        file_name = select_file(resources_path, save=False, filter="Images (*.png *.tiff *.jpg)")
        if file_name != '':
            self.load_image_from_name(file_name, load_type=load_type)
            return self.field

    def load_image_from_name(self, fname: Union[str, Path] = cheshire_cat_path,
                             load_type=LoadTypeEnum.AMPLITUDE,
                             notify=True):

        load_type = enum_checker(LoadTypeEnum, load_type)

        fname = Path(fname)
        if fname.is_file():
            try:
                img_array = imread(fname)
                if len(img_array.shape) == 2:
                    pass
                elif len(img_array.shape) == 3:
                    img_array = rgb2gray(img_array[..., 0:3])

                if load_type == LoadTypeEnum.AMPLITUDE:
                    self.field.amplitude = mutils.normalize_to(np.flipud(img_array), 1)
                else:
                    self.field.phase = mutils.normalize_to(np.flipud(img_array), 2 * np.pi)
                if notify:
                    self.notify_listeners(self.field)

            except Exception as e:
                logger.exception(str(e))

    def load(self, *args, load_type: LoadTypeEnum = None, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        if 'fname' in kwargs and load_type is not None:
            self.load_image_from_name(kwargs['fname'], load_type=load_type)
        else:
            self.load_image_from_name(self.settings['amp_target_file'],
                                      load_type=LoadTypeEnum.AMPLITUDE, notify=False)
            self.load_image_from_name(self.settings['phase_target_file'],
                                      load_type=LoadTypeEnum.PHASE)

        return self.field


