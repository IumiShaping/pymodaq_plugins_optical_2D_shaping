import pytest
import numpy as np
from pymodaq_plugins_beam_shaping.utilities import sizing

@pytest.mark.parametrize('factor_reduced', (4,))
def test_pixelize(factor_reduced):
    len = 2
    factor = 4
    array_in = np.linspace(1, 2**(len**2), 2**(len**2)).reshape((2**len, 2**len))
    array_expanded = np.repeat(np.repeat(array_in, factor, axis=0), factor, axis=1)

    binned_array = sizing.pixelize(array_expanded, factor_reduced)
    assert array_expanded.shape == binned_array.shape
    if factor_reduced == factor:
        assert np.allclose(binned_array, array_expanded)

