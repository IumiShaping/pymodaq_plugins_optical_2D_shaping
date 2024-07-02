from abc import ABCMeta, abstractproperty

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq.utils.managers.parameter_manager import ParameterManager, Parameter
from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.parameter.utils import iter_children
from pymodaq.utils.enums import BaseEnum, enum_checker
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.plotting.data_viewers.viewer2D import Viewer2D
from pymodaq.utils.data import DataRaw
from pymodaq.utils.gui_utils.custom_app import CustomApp
from pymodaq.utils.gui_utils.dock import DockArea, Dock

from pymodaq_plugins_optical_2D_shaping.target_loaders import target_loader_factory, TargetLoader
from pymodaq_plugins_optical_2D_shaping.target_loaders.field import Field

class TargetApp(CustomApp):