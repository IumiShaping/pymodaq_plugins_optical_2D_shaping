from pymodaq.utils.data import DataActuator


class DataShaper(DataActuator):
    def __init__(self, *args, as_grey_levels = False, **kwargs):
        super().__init__(*args, as_grey_levels=as_grey_levels, **kwargs)



