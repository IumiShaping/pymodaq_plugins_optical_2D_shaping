from typing import Callable
from pymodaq_plugins_optical_2D_shaping.algorithms.utils import AlgoBase


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
        and instantiates it.

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

        return cls.algorithms_registry[algo_name]()

    @property
    def algorithms(self):
        return list(self.algorithms_registry.keys())