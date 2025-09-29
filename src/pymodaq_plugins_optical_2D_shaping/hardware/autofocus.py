from abc import ABC, abstractmethod
from typing import Callable
from pathlib import Path

import numpy as np
import cv2
from skimage.measure import shannon_entropy, blur_effect
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale, resize
from skimage.util import crop

resources_path = Path(__file__).parent.parent.joinpath('resources')
cheshire_cat_path = resources_path.joinpath('cheshirecat_rect.png')

CAT_ARRAY = np.flipud(imread(cheshire_cat_path))
if len(CAT_ARRAY.shape) == 2:
    pass
elif len(CAT_ARRAY.shape) == 3:
    CAT_ARRAY = rgb2gray(CAT_ARRAY[..., 0:3])

class AutoFocusBase(ABC):
    
    name: str

    @abstractmethod
    def compute(self, array: np.ndarray):
        ...


class AutoFocusFactory:
    _builders = {}

    @classmethod
    def register(cls) -> Callable:
        """ To be used as a decorator

        Register in the class registry a new scanner class using its 2 identifiers: scan_type and scan_sub_type
        """

        def inner_wrapper(wrapped_class: AutoFocusBase) -> AutoFocusBase:
            key = wrapped_class.name
            cls._builders[key] = wrapped_class
            return wrapped_class

        return inner_wrapper


    @classmethod
    def get(cls, key : str) -> AutoFocusBase:
        builder = cls._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder

    @classmethod
    def create(cls, key, **kwargs) -> AutoFocusBase:
        return cls._builders.get(key)(**kwargs)

    @classmethod
    def keys(cls) -> list[str]:
        return list(cls._builders.keys())

    @classmethod
    def names(cls) -> list[str]:
        return list(cls.keys())



# Function to compute Local Variance focus measure
# see https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/

@AutoFocusFactory.register()
class AutoFocusLocalVariance:
    name = 'LocalVariance'

    @staticmethod
    def compute(array: np.ndarray, ksize=5):
        mean = cv2.blur(array, (ksize, ksize))
        squared_mean = cv2.blur(array**2, (ksize, ksize))
        variance = squared_mean - (mean**2)
        return np.mean(variance)


@AutoFocusFactory.register()
class AutoFocusShannonEntropy:
    name = 'ShannonEntropy'

    @staticmethod
    def compute(array: np.ndarray):
        return shannon_entropy(array)


@AutoFocusFactory.register()
class AutoFocusTenengrad:
    name = 'Tenengrad'

    @staticmethod
    def compute(array: np.ndarray):
        sobel_x = cv2.Sobel(array, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
        sobel_y = cv2.Sobel(array, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
        tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(tenengrad)  # Return mean gradient magnitude as focus score


@AutoFocusFactory.register()
class AutoFocusBrennerGradient:
    name = 'BrennerGradient'

    @staticmethod
    def compute(array: np.ndarray):
        shifted = np.roll(array, -2, axis=1)  # Shift by 2 pixels horizontally
        diff = (array - shifted) ** 2  # Compute squared difference
        return np.sum(diff)  # Sum all differences as the focus measure


@AutoFocusFactory.register()
class AutoFocusSobelVariance:
    name = 'SobelVariance'

    @staticmethod
    def compute(array: np.ndarray):
        sobel_x = cv2.Sobel(array, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(array, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        variance = np.var(array)  # Compute variance of pixel intensities
        return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance


@AutoFocusFactory.register()
class AutoFocusLaplacian:
    name = 'Laplacian'

    @staticmethod
    def compute(array):
        laplacian = cv2.Laplacian(array, cv2.CV_64F)  # Apply Laplacian filter
        return np.var(laplacian)  # Compute variance of Laplacian


@AutoFocusFactory.register()
class AutoFocusSkImage:
    name = 'SkImage'

    @staticmethod
    def compute(array):
        return blur_effect(array)



class Autofocus:

    _blurr_amplitude = 10

    @property
    def blurr(self):
        return self._blurr_amplitude

    @blurr.setter
    def blurr(self, new_blurr: int):
        self._blurr_amplitude = int(abs(new_blurr))

    def grab(self):
        if self.blurr == 0:
            return CAT_ARRAY
        else:
            array = rescale(cv2.blur(CAT_ARRAY, (self.blurr, self.blurr)),
                            1 + self._blurr_amplitude/100)
            crop_width = (np.array(array.shape) - np.array(CAT_ARRAY.shape)) / 2
            crop_width = tuple(crop_width.astype(int))
            return crop(array, crop_width)