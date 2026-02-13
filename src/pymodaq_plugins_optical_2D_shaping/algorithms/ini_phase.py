from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable
from importlib import import_module

import numpy as np

from pymodaq_gui.parameter import Parameter
from pymodaq_plugins_optical_2D_shaping.algorithms import AlgoBase
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import LensSetup

from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))


def register_phases(parent_module_name: str = 'pymodaq_plugins_optical_2D_shaping'):
    """ Browse modules containing initial phases for optical shaping and register them in the factory"""
    phases = []
    try:
        phase_module = import_module(f'{parent_module_name}.algorithms.ini_phases')

        phase_path = Path(phase_module.__path__[0])

        for file in phase_path.iterdir():
            if file.is_file() and 'py' in file.suffix and file.stem != '__init__':
                try:
                    phases.append(import_module(f'.{file.stem}', phase_module.__name__))
                except (ModuleNotFoundError, NotImplementedError) as e:
                    logger.warning(str(e))
    except ModuleNotFoundError as e:
        logger.warning(str(e))
    finally:
        return phases


class PhaseBase(ABC):

    params: list[dict[str, str]] = []  # definition of the specific parameters needed to compute the loss

    def __init__(self, settings: Parameter, algo: AlgoBase):
        self.settings = settings  # attribute used to access specific parameters changed by the user
        self.algo = algo

    @abstractmethod
    def compute_phase(self, **kwargs) -> np.ndarray:
        """ Compute the initial phase to be fed into an algorithm

        To be reimplemented
        """
        ...


class PhaseFactory:
    """ The factory class for creating initial phases """

    _registry = {}

    @classmethod
    def register_phase(cls) -> Callable:
        """Class decorator method to register Ini Phases class to the internal registry. Must be used as
        decorator above the definition of a PhaseBase inherited class.

        The Loss class must implement specific class attributes and methods
        """

        def inner_wrapper(wrapped_class: type[PhaseBase]) -> type[PhaseBase]:
            name = wrapped_class.__name__

            if name not in cls._registry:
                cls._registry[name] = wrapped_class
            # Return wrapped_class
            return wrapped_class

        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_phase(cls, name: str) -> type[PhaseBase]:
        """Factory command to get registered phase class
        .
        This method gets the appropriate executor class from the registry

        Parameters
        ----------
        name: str
            The name of the class as specified during registration

        Returns
        -------
        an instance of the executor created
        """

        if name not in cls._registry:
            raise ValueError(f".{name} is not a supported Initial Phase.")

        return cls._registry[name]

    @property
    def phases(self):
        return list(self._registry.keys())

