from pathlib import Path
from pymodaq_utils.logger import set_logger  # to be imported by other modules.
from .utils import Config
config = Config()  # before field loader factory otherwise it will generate an error

from .field import field_loader_factory, LoaderFactory
from .algorithms.factory import AlgorithmFactory
from .algorithms.utils import LensSetup

algo_factory = AlgorithmFactory()
loader_factory = LoaderFactory()

#check the presence of all registered algorithm in the config
registered_algorithms = algo_factory.algorithms
config_algorithms = config('algo', 'default_algo')

for algo in registered_algorithms:
    if algo not in config_algorithms:
        config_algorithms.append(algo)
config['algo', 'default_algo'] = config_algorithms

#check the presence of all registered field loader in the config
registered_field_loader = loader_factory.field_loaders
config_target_field_loader = config('target', 'default_loader')
config_input_field_loader = config('input', 'default_loader')


for loader in registered_field_loader:
    if loader not in config_target_field_loader:
        config_target_field_loader.append(loader)
    if loader not in config_input_field_loader:
        config_input_field_loader.append(loader)
config['target', 'default_loader'] = config_target_field_loader
config['input', 'default_loader'] = config_input_field_loader
config.save()
