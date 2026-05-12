import numpy as np
from numpy.fft import fftshift, fft

from pymodaq_plugins_beam_shaping.field import Field
from time import perf_counter

from pymodaq_gui.qt_utils import mkQApp

app = mkQApp('fft')
N = 2048
M = 1024
box = np.zeros((M, N), dtype='float64')
box[M//2-10: M//2+10, N//2-50: N//2+50] = 1
field = Field('box', amplitude=box)

# using numpy
fft_complex = fftshift(fft(fftshift(field.amplitude, axes=0), axis=0), axes=0)
fft_2_complex = fftshift(fft(fftshift(fft_complex.T, axes=0), axis=0), axes=0).T

field_fft = Field('fft', amplitude=np.abs(fft_complex), phase=np.angle(fft_complex))
field_fft2 = Field('fft', amplitude=np.abs(fft_2_complex), phase=np.angle(fft_2_complex))


field_fft.intensity_as_dwa().plot('qt')
field_fft2.intensity_as_dwa().plot('qt')

# using pytensor
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import fft


start = perf_counter()
x = pytensor.shared(box, 'box')


rfft = fft.rfft(x, norm='ortho')
rfft_complex = (rfft[..., 0] + 1j * rfft[..., 1]).transpose()
rfft2 = fft.rfft(pt.abs(rfft_complex) * pt.cos(pt.angle(rfft_complex)), norm='ortho')
rfft2_complex = (rfft2[..., 0] + 1j * rfft2[..., 1]).transpose()
fft2_complex = pt.abs(rfft2_complex) * pt.exp(1j * pt.angle(rfft2_complex))

f_rfft2 = pytensor.function([], fft2_complex)

compiled = perf_counter() - start
print(f'Compilation time: {compiled}')

calculated = perf_counter()
for ind in range(10):
    out = f_rfft2()

    calculated = perf_counter() - calculated
    print(f'Calculated time: {calculated}')

    print(out.shape)


field_fft2_tensor = Field(amplitude=np.abs(out), phase=np.angle(out))
field_fft2_tensor.intensity_as_dwa('tensor', 'tensor').plot('qt')


app.exec()