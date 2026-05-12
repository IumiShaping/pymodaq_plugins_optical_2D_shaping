import numpy as np
import pytest
from pymodaq_plugins_beam_shaping.field import Field


class TestField:

    def test_field_init_empty(self):
        NAME = 'myname'
        f = Field(NAME)
        assert f.name == NAME

    def test_field_init_amplitude(self):
        amp_array = np.random.random((10, 15))

        field = Field('amplitude', amplitude=amp_array)
        assert np.allclose(field.amplitude, amp_array)
        assert np.allclose(field.phase, np.zeros_like(amp_array))

    def test_field_init_phase(self):
        phase_array = np.random.random((10, 15))

        field = Field('amplitude', phase=phase_array)
        assert np.allclose(field.phase, phase_array)
        assert np.allclose(field.amplitude, np.ones_like(phase_array))



