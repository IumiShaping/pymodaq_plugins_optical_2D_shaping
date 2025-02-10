from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets

from pymodaq_gui.managers.parameter_manager import ParameterManager, Parameter
from pymodaq_utils.enums import BaseEnum
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_gui.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq_plugins_optical_2D_shaping.field import Field

logger = set_logger(get_module_name(__file__))


class LoadTypeEnum(BaseEnum):

    AMPLITUDE = 'amplitude'
    PHASE = 'phase'


class FieldLoaderParameterManager(ParameterManager):
    settings_name = 'loader_settings'

    def __init__(self):
        super().__init__()
        self.settings_tree.header().setVisible(True)
        self.settings_tree.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.settings_tree.header().setMinimumSectionSize(150)
        self.settings_tree.setMinimumHeight(150)


class FieldLoader(FieldLoaderParameterManager, metaclass=ABCMeta):
    LOADER_NAME = abstractproperty()
    with_physical_pixels_size = False
    params = [
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parent_app = kwargs.pop('parent')

        self._listener = dict()
        self.field = Field()

    @property
    def pixel_width(self):
        return self.parent_app.settings['needed_size', 'pixel_width']

    @property
    def pixel_height(self):
        return self.parent_app.settings['needed_size', 'pixel_height']

    @property
    def n_pixel_width(self):
        return self.parent_app.settings['needed_size', 'width']

    @property
    def n_pixel_height(self):
        return self.parent_app.settings['needed_size', 'height']

    def register_listener(self, object_instance):
        """ Register objects that will be notified when the field attribute has been changed

        The object should have a method named update_field that will be called with the new
        field whenever the :meth:`load_target` method is called
        """
        if hasattr(object_instance, 'update_field'):
            self._listener.update({object_instance.__class__.__name__: object_instance})

    def notify_listeners(self, field: Field):
        for listener in self._listener.values():
            listener.update_field(field)

    def value_changed(self, param: Parameter):
        ...

    def load_field(self, *args, notify=True, **kwargs) -> Field:
        """Load a field to populate the field attribute

        This method should be used from external object
        """
        field = self.load(*args, **kwargs)
        if notify:
            self.notify_listeners(field)
        return field

    def load(self, *args, **kwargs) -> Field:
        """ Abstract method to reimplement. Used to load something
        to populate the field attribute"""
        raise NotImplementedError