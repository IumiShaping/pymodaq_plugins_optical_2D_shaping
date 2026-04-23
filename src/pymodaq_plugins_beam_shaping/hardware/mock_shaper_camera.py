import numpy as np

from pymodaq.utils.data import DataActuator

from pymodaq_plugins_beam_shaping.utilities.sizing import get_slm_size, get_effective_slm_pixel_size


def gaussian_fwhm(x: np.ndarray, x0: float, fwhm: float) -> np.ndarray:
    """ Get a Gaussian amplitude distribution centered in x0 with a full width at half maximum in intensity of fwhm

    Parameters
    ----------
    x: np.ndarray
        distribution on which the gaussian is evaluated (in microns)
    x0: float
        center of the gaussian distribution (in micron)
    fwhm: float
        full width at half maximum in intensity expressed (in micron)

    Returns
    -------
    np.ndarray
    """
    return np.exp(- 2 * np.log(2) * ((x - x0) / fwhm) ** 2)


def compute_grid() -> tuple[np.ndarray, np.ndarray]:
    """ Compute the centered positions on the grid in microns given the value of the pixel width"""

    n_pixel_height, n_pixel_width = get_slm_size()
    pixel_height, pixel_width = (get_effective_slm_pixel_size(), get_effective_slm_pixel_size())

    x = np.arange(0, n_pixel_width, 1) * pixel_width   #um
    x = x-np.mean(x)
    y = np.arange(0, n_pixel_height, 1) * pixel_height  #um
    y = y - np.mean(y)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


class ShaperCamera:
    def __init__(self):
        self._slm_values: np.ndarray = np.zeros(get_slm_size())

        xx, yy = compute_grid()
        self._amplitude = (gaussian_fwhm(xx, 0, 7e3) *
                           gaussian_fwhm(yy, 0, 7e3))

    def get_slm(self):
        return self._slm_values

    def apply_grey_scale(self, value: np.ndarray):
        self._slm_values = value

    def get_camera(self) -> np.ndarray:
        return np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self._amplitude * np.exp(1j * self._slm_values)))) /
                      np.sqrt(np.prod(self._slm_values.shape))) ** 2
