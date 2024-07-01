from importlib import import_module
from pathlib import Path

from typing import Callable

from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase


def register_algorithms(parent_module_name: str = 'pymodaq_plugins_optical_2D_shaping'):
    """ Browse modules containing algorithms for optical shaping and register them in the factory"""
    algorithms = []
    try:
        algorithm_module = import_module(f'{parent_module_name}.algorithms.algorithms')

        algorithm_path = Path(algorithm_module.__path__[0])

        for file in algorithm_path.iterdir():
            if file.is_file() and 'py' in file.suffix and file.stem != '__init__':
                try:
                    algorithms.append(import_module(f'.{file.stem}', algorithm_module.__name__))
                except ModuleNotFoundError:
                    pass
    except ModuleNotFoundError:
        pass
    finally:
        return algorithms


class AlgorithmFactory:
    """The factory class for creating Algorithm"""

    algorithms_registry = {}

    @classmethod
    def register_algorithm(cls) -> Callable:
        """Class decorator method to register algo class to the internal registry. Must be used as
        decorator above the definition of an AlgoBase inherited class.

        The Algorithm class must implement specific class attributes and methods

        returns:
            the exporter class
        """

        def inner_wrapper(wrapped_class) -> Callable:
            algo_name = wrapped_class.ALGO_NAME

            if algo_name not in cls.algorithms_registry:
                cls.algorithms_registry[algo_name] = wrapped_class
            # Return wrapped_class
            return wrapped_class

        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_algorithm(cls, algo_name: str) -> AlgoBase:
        """Factory command to get registered algorithms
        .
        This method gets the appropriate executor class from the registry

        Parameters
        ----------
        algo_name: str
            The name of the algorithm as specified during registration

        Returns
        -------
        an instance of the executor created
        """

        if algo_name not in cls.algorithms_registry:
            raise ValueError(f".{algo_name} is not a supported Algorithm.")

        return cls.algorithms_registry[algo_name]

    @property
    def algorithms(self):
        return list(self.algorithms_registry.keys())