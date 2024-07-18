from .field import Field
from .factory import LoaderFactory, register_loaders, FieldLoader


register_loaders()

field_loader_factory = LoaderFactory()
