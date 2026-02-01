import numpy as np
from pathlib import Path
from typing import Union

from skimage.io import imread
from skimage.color import rgb2gray

from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq_gui.parameter import Parameter
from pymodaq_gui.utils.file_io import select_file
from pymodaq_utils.enums import enum_checker
from pymodaq_utils import math_utils as mutils

from pymodaq_plugins_optical_2D_shaping.field.utils import (FieldLoader, LoadTypeEnum,
                                                            Field)
from pymodaq_plugins_optical_2D_shaping.field.factory import LoaderFactory

resources_path = Path(__file__).parent.parent.parent.joinpath('resources')
cheshire_cat_path = resources_path.joinpath('cheshirecat_rect.png')
cemes_path = resources_path.joinpath('Cemes - Logo - Sigle - Blanc.png')
osama = resources_path.joinpath('osama.png')
one_piece = resources_path.joinpath('one_piece.png')
logger = set_logger(get_module_name(__file__))


@LoaderFactory.register_loader()
class ImageFileLoader(FieldLoader):

    LOADER_NAME = 'ImageFileLoader'

    params = FieldLoader.params + \
        [{'title': 'Amplitude:', 'name': 'amplitude', 'type': 'group', 'children': [
            {'title': 'File path:', 'name': 'file', 'type': 'browsepath',
             'value': str(osama), 'filetype': True},
            {'title': 'Load it:', 'name': 'load', 'type': 'bool', 'value': True},
        ]},
         {'title': 'Phase:', 'name': 'phase', 'type': 'group', 'children': [
             {'title': 'File path:', 'name': 'file', 'type': 'browsepath',
              'value': str(one_piece), 'filetype': True},
             {'title': 'Load it:', 'name': 'load', 'type': 'bool', 'value': True},
         ]},
         ]

    def settings_changed(self, param: Parameter):
        """ Don't do anything but only use the reload button otherwise it is too complex to load both phase/amplitude
         images or only one... """
        ...

    def load_image(self, load_type=LoadTypeEnum.AMPLITUDE):
        file_name = select_file(resources_path, save=False, filter="Images (*.png *.tiff *.jpg)")
        if file_name != '':
            self.load_image_from_name(file_name, load_type=load_type)
            return self.field

    def load_image_from_name(self, fname: Union[str, Path] = cheshire_cat_path,
                             load_type: LoadTypeEnum = LoadTypeEnum.AMPLITUDE,
                             ) -> np.ndarray:

        load_type = enum_checker(LoadTypeEnum, load_type)

        fname = Path(fname)
        if fname.is_file():
            try:
                img_array = imread(fname, as_gray=True)
                if len(img_array.shape) == 2:
                    pass
                elif len(img_array.shape) == 3:
                    img_array = rgb2gray(img_array[..., 0:3])

                if load_type == LoadTypeEnum.AMPLITUDE:
                    img_array = mutils.normalize_to(np.flipud(img_array), 1.)
                else:
                    img_array = mutils.normalize_to(np.flipud(img_array), 2 * np.pi)
                return img_array

            except Exception as e:
                logger.exception(str(e))

    def load(self, *args, load_type: LoadTypeEnum = None, **kwargs) -> Field:
        """ Mandatory reimplemented method. Used to load a target to populate the field attribute"""
        if 'fname' in kwargs and load_type is not None:
            img_array = self.load_image_from_name(kwargs['fname'], load_type=load_type)
            if load_type == LoadTypeEnum.AMPLITUDE:
                self.field = Field('Image', amplitude=img_array)
            else:
                self.field = Field('Image', phase=img_array)
        else:
            amplitude= self.load_image_from_name(self.settings['amplitude', 'file'],
                                                 load_type=LoadTypeEnum.AMPLITUDE)
            phase = self.load_image_from_name(self.settings['phase', 'file'],
                                              load_type=LoadTypeEnum.PHASE)

            self.field = Field(
                'Image',
                amplitude=amplitude if amplitude is not None and self.settings['amplitude', 'load'] else None,
                phase=phase if phase is not None and self.settings['phase', 'load'] else None)

        return self.field


