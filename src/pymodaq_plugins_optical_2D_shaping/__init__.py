from pathlib import Path
from pymodaq.utils.logger import set_logger  # to be imported by other modules.
from .utils import Config
from .field import field_loader_factory

config = Config()


with open(str(Path(__file__).parent.joinpath('resources/VERSION')), 'r') as fvers:
    __version__ = fvers.read().strip()
