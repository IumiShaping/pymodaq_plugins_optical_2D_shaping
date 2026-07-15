from typing import TYPE_CHECKING, Sequence, Union

import numpy as np
from skimage.transform import downscale_local_mean, rescale

from pymodaq_utils.math_utils import greater2n, is_power_of_two
from pymodaq_plugins_beam_shaping import config as plugin_config

if TYPE_CHECKING:
    from pymodaq_plugins_beam_shaping.field import Field

def get_binning() -> int:
    return plugin_config('sizing', 'binning')


def get_effective_needed_field_size() -> tuple[int, int]:
    """ Compute from the configuration values the needed square size of the fields to be used"""

    binning = plugin_config('sizing', 'binning')
    if not is_power_of_two(binning):
        raise ValueError('binning must be a multiple of 2')
    if plugin_config('sizing', 'square_size'):
        needed_field_size = (greater2n(max(plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'height'),
                                           plugin_config('SLM', plugin_config('SLM', 'default_slm')[0], 'width'))) //
                             binning)
        needed_field_size = (needed_field_size, needed_field_size)
    else:
        needed_field_size = tuple(np.array(get_slm_size()) // binning)
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


def unbin_to_real_slm(phase: np.ndarray, binning: int = None) -> np.ndarray:
    """ "rebin" a given ndarray by repeating its values binning time to fit the pixel arrays of the real SLM

    The final shape is initial shape x binning

    """
    if binning is None:
        binning = plugin_config('sizing', 'binning')
    return np.repeat(np.repeat(phase, binning, axis=0), binning, axis=1)


def get_npad_between(first_shape, second_shape):
    """ Get the padding necessary to match object shape and image shape

    If positive, the image shape is bigger than the object
    If negative, the object shape is bigger than the image
    """
    npad_before = ((np.array(first_shape) -
                    np.array(second_shape)) // 2).astype(int)
    npad_after = (np.array(first_shape) -
                    np.array(second_shape)) - npad_before
    return (npad_before[0], npad_after[0]), (npad_before[1], npad_after[1])


def slice_for_crop(shape: Sequence[int],
                   size: tuple[int, int],
                   center: tuple[int, int] = None) -> tuple[slice, slice]:
    shape = np.atleast_1d(shape)
    if center is None:
        center = tuple(shape // 2)

    return(slice(max(0, center[0] - size[0] // 2), min(shape[0], center[0] + size[0] // 2)),
           slice(max(0, center[1] - size[1] // 2), min(shape[1], center[1] + size[1] // 2)))


def crop_field(field: 'Field',
               size: tuple[int, int],
               center: tuple[int, int] = None):
    cropped_field = field.isig[*slice_for_crop(field.shape, size, center)]

    return cropped_field.pad(get_npad_between(size, cropped_field.shape), mode='edge')


def crop_array(array: np.ndarray,
               size: tuple[int, int],
               center: tuple[int, int] = None):
    cropped_array = array[*slice_for_crop(array.shape, size, center)]
    return np.pad(cropped_array, get_npad_between(size, cropped_array.shape),
                  mode='edge')

def pixelize(data: np.ndarray, bin_factor: int) -> np.ndarray:
    if np.all(np.array(data.shape) // bin_factor == np.array(data.shape) / bin_factor):
        binned_data = downscale_local_mean(data, bin_factor)
    else:
        binned_data = rescale(data, 1 / bin_factor)
    pixelated_array = np.repeat(
        np.repeat(binned_data, bin_factor, axis=0),
        bin_factor, axis=1)
    return crop_array(pixelated_array, data.shape)

def get_effective_area_pos_size_in_pxls() -> tuple[np.ndarray, np.ndarray]:
    """ Get the position and size of the effective area image of the SLM in pixels

    pos is the position of the left, bottom corner of the area
    """
    size = np.array(get_effective_slm_size())
    pos = (np.array(get_effective_needed_field_size()) - size) / 2
    return pos, size



if __name__ == '__main__':
    pass