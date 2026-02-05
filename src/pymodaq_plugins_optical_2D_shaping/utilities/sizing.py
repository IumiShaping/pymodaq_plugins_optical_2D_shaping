import numpy as np
from pymodaq_utils.math_utils import greater2n, is_power_of_two
from pymodaq_plugins_optical_2D_shaping import config as plugin_config



def get_effective_needed_field_size() -> int:
    """ Compute from the configuration values the needed square size of the fields to be used"""

    binning = plugin_config('sizing', 'binning')
    if not is_power_of_two(binning):
        raise ValueError('binning must be a multiple of 2')

    needed_field_size = (greater2n(max(plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
                                      plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width'))) //
                         binning)
    return needed_field_size


def get_slm_size() -> tuple[int, int]:
    return (plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
            plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width'))


def get_effective_slm_size() -> tuple[int, int]:
    binning = plugin_config('sizing', 'binning')
    if not is_power_of_two(binning):
        raise ValueError('binning must be a multiple of 2')
    size = (np.array([plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
                                       plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width')]) //
         binning)
    return int(size[0]), int(size[1])


def get_effective_slm_pixel_size() -> float:
    """ Compute from the configuration values the effective SLM pixel size in microns"""

    return (plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'pixel_size') *
            plugin_config('sizing', 'binning'))


def unbin_slm(phase: np.ndarray, binning: int) -> np.ndarray:
    """ "rebin" a given ndarray by repeating its values binning time

    The final shape is initial shape x binning
    """
    return np.repeat(np.repeat(phase, binning, axis=0), binning, axis=1)



if __name__ == '__main__':
    pass