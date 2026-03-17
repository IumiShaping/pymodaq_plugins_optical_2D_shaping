from abc import ABC, abstractmethod
from typing import Callable

from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoType


from pymodaq_gui.parameter import Parameter


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


@StoppingFactory.register_stop()
class NoneStop(StoppingBase):
    """ Never automatically stops the algorithm """
    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return False


@StoppingFactory.register_stop()
class Iter(StoppingBase):
    """ Automatically stops the algorithm after it reached a given number of iteration """

    params = [
        {'title': 'Niter max:', 'name': 'niter_max', 'type': 'int', 'value': 100}
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return iterative_index >= self.settings['niter_max']


@StoppingFactory.register_stop()
class FitnessThreshold(StoppingBase):
    """ Automatically stops the algorithm when its fitness is below a given threshold"""

    params = [
        {'title': 'Fitness:', 'name': 'fitness', 'type': 'float', 'value': 0.1}
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return fitness <= self.settings['fitness']


@StoppingFactory.register_stop()
class EfficiencyThreshold(StoppingBase):
    """ Automatically stops the algorithm when its efficiency is above a given threshold and after
    a given number of iteration"""

    params = [
        {'title': 'Efficiency:', 'name': 'efficiency', 'type': 'float', 'value': 0.9},
        {'title': 'Niter max:', 'name': 'niter_max', 'type': 'int', 'value': 100},
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return fitness <= self.settings['fitness'] and iterative_index >= self.settings['niter_max']

@StoppingFactory.register_stop()
class FitnessConvergence(StoppingBase):
    """ Automatically stops the algorithm when its fitness is below a given threshold"""
    """ Never automatically stop the algorithm """

    params = [
        {'title': 'Fitness:', 'name': 'fitness', 'type': 'float', 'value': 0.1}
    ]

    def tell_stop(self,
                  iterative_index: int,
                  fitness: float,
                  efficiency: float,
                  *args, **kwargs) -> bool:
        return fitness <= self.settings['fitness']