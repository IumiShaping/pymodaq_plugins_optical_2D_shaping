from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Callable


from pymodaq_plugins_beam_shaping.algorithms.utils import AlgoType
from pymodaq_utils.logger import set_logger, get_module_name


from pymodaq_gui.parameter import Parameter


logger = set_logger(get_module_name(__file__))


class StoppingBase(ABC):
    ALGOTYPE = AlgoType.AMPLITUDE

    params: list[dict[str, str]] = []  # definition of the specific parameters needed to compute the loss

    def __init__(self, settings: Parameter):
        self.settings = settings  # attribute used to access specific parameters changed by the user

    @abstractmethod
    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        """ evaluate if the algorithm should stop given some parameters and metrics

        Parameters
        ----------
        iterative_index: int
            The current iteration of the algorithm since phase initialization
        fitness: float
            The fitness of the current iteration: either NRMSE or Fidelity Error
        efficiency: float
            The efficiency of the current iteration

        Returns
        -------
        bool: True if the algorithm should stop
        """
        ...


class StoppingFactory:
    """The factory class for creating stopping classes"""

    _registry = {}

    @classmethod
    def register_stop(cls) -> Callable:
        """Class decorator method to register Stopping class to the internal registry. Must be used as
        decorator above the definition of a StoppingBase inherited class.

        The StoppingBase class must implement specific class attributes and methods
        """

        def inner_wrapper(wrapped_class: type[StoppingBase]) -> type[StoppingBase]:
            name = wrapped_class.__name__

            if name not in cls._registry:
                cls._registry[name] = wrapped_class
            # Return wrapped_class
            return wrapped_class
        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_stop_class(cls, name: str) -> type[StoppingBase]:
        """Factory command to get registered stopping class
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
            raise ValueError(f".{name} is not a supported Stopping class.")

        return cls._registry[name]

    @property
    def stops(self):
        return list(self._registry.keys())


def register_stoppings(parent_module_name: str = 'pymodaq_plugins_beam_shaping'):
    """ Browse modules containing stopping criteria for optical shaping and register them in the factory"""
    try:
        stopping_module = import_module(f'{parent_module_name}.algorithms.stoppings')

        stopping_path = Path(stopping_module.__path__[0])

        for file in stopping_path.iterdir():
            if file.is_file() and 'py' in file.suffix and file.stem != '__init__':
                try:
                    import_module(f'.{file.stem}', stopping_module.__name__)
                except (ModuleNotFoundError, NotImplementedError) as e:
                    logger.warning(str(e))
    except ModuleNotFoundError as e:
        logger.warning(str(e))

