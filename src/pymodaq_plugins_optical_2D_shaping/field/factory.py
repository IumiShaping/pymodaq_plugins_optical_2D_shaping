from pathlib import Path
from importlib import import_module
from typing import Callable
from pymodaq_plugins_optical_2D_shaping.field.utils import FieldLoader

from pymodaq_utils.logger import set_logger, get_module_name

logger = set_logger(get_module_name(__file__))


def register_loaders(parent_module_name: str = 'pymodaq_plugins_optical_2D_shaping'):
    """ Browse modules containing algorithms for optical shaping and register them in the factory"""
    loaders = []
    try:
        loader_module = import_module(f'{parent_module_name}.field.loaders')

        loader_path = Path(loader_module.__path__[0])

        for file in loader_path.iterdir():
            if file.is_file() and 'py' in file.suffix and file.stem != '__init__':
                try:
                    loaders.append(import_module(f'.{file.stem}', loader_module.__name__))
                except Exception as e:
                    logger.exception(str(e))
    except ModuleNotFoundError:
        pass
    finally:
        return loaders


class LoaderFactory:
    """The factory class for creating target loaders"""

    loader_registry = {}

    @classmethod
    def register_loader(cls) -> Callable:
        """Class decorator method to register algo class to the internal registry. Must be used as
        decorator above the definition of an FieldLoader inherited class.

        The Algorithm class must implement specific class attributes and methods

        returns:
            the exporter class
        """

        def inner_wrapper(wrapped_class) -> Callable:
            algo_name = wrapped_class.LOADER_NAME

            if algo_name not in cls.loader_registry:
                cls.loader_registry[algo_name] = wrapped_class
            # Return wrapped_class
            return wrapped_class

        # Return decorated function
        return inner_wrapper

    @classmethod
    def get_loader(cls, loader_name: str) -> FieldLoader:
        """Factory command to get registered target loaders
        .
        This method gets the appropriate executor class from the registry

        Parameters
        ----------
        loader_name: str
            The name of the target loader as specified during registration

        Returns
        -------
        an instance of the executor created
        """

        if loader_name not in cls.loader_registry:
            raise ValueError(f".{loader_name} is not a supported Field loader.")

        return cls.loader_registry[loader_name]

    @property
    def field_loaders(self) -> list[str]:
        return list(self.loader_registry.keys())