from pathlib import Path
from pymodaq_utils.logger import set_logger  # to be imported by other modules.
from .utils import Config
config = Config()  # before field loader factory otherwise it will generate an error

from .field import field_loader_factory

