import pytest
import numpy as np
from pymodaq_plugins_beam_shaping.utilities import sizing

@pytest.mark.parametrize('factor_reduced', np.arange(1, 10))
def test_bin_average_array(factor_reduced):
    len = 2
    factor = 4
    array_in = np.linspace(1, 2**(len**2), 2**(len**2)).reshape((2**len, 2**len))
    array_expanded = np.repeat(np.repeat(array_in, factor, axis=0), factor, axis=1)

    binned_array = sizing.bin_average_array(array_expanded, factor_reduced)
    assert array_expanded.shape == binned_array.shape
    if factor_reduced == factor:
        assert np.all(binned_array == array_expanded)

