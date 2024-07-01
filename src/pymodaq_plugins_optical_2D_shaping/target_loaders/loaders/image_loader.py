import numpy as np
from pathlib import Path
from typing import Union

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataFromPlugins, DataToExport, DataRaw
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.gui_utils.file_io import select_file

from pymodaq_plugins_optical_2D_shaping.target_loaders.utils import (TargetLoader, LoadTypeEnum,
                                                                     enum_checker)
from pymodaq_plugins_optical_2D_shaping.target_loaders.factory import TargetLoaderFactory

resources_path = Path(__file__).parent.parent.joinpath('resources')
cheshire_cat_path = resources_path.joinpath('cheshirecat_rect.png')


logger = set_logger(get_module_name(__file__))


@TargetLoaderFactory.register_target_loader()
class ImageFileLoader(TargetLoader):

    LOADER_NAME = 'ImageFileLoader'

    params = TargetLoader.params + \
        [{'title': 'Amplitude File path:', 'name': 'amp_target_file',
          'type': 'browsepath', 'value': '', 'filetype': True},
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
                self.update_viewers()

        elif param.name() == 'phase_target_file':
            if Path(param.value()).is_file():
                self.load_image_from_name(fname=Path(param.value()),
                                          load_type=LoadTypeEnum.PHASE)
                logger.info(f'Phase Image loaded from {param.value()}')
                self.update_viewers()

    def load(self, *args, load_type=LoadTypeEnum.AMPLITUDE, **kwargs):
        self.load_image(load_type)

        return self.field

    def load_image(self, load_type=LoadTypeEnum.AMPLITUDE):
        file_name = select_file(resources_path, save=False, filter="Images (*.png *.tiff *.jpg)")
        if file_name != '':
            self.load_image_from_name(file_name, load_type=load_type)

    def load_image_from_name(self, fname: Union[str, Path] = cheshire_cat_path,
                             load_type=LoadTypeEnum.AMPLITUDE):

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
                    self.field.amplitude = np.flipud(img_array)
                else:
                    self.field.phase = np.flipud(img_array)

            except Exception as e:
                logger.exception(str(e))


if __name__ == '__main__':
    from pymodaq.utils.gui_utils.utils import mkQApp

    app = mkQApp('ImageLoader')
    loader = ImageFileLoader()
    loader.settings_tree.show()
    loader.target_widget.show()

    app.exec()

