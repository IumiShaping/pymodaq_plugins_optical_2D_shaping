import numpy as np
from numpy.fft import fftshift, fft
from pymodaq_plugins_optical_2D_shaping.field import Field
from time import perf_counter

import torch
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.set_default_device(device)

from pymodaq_gui.utils.utils import mkQApp

app = mkQApp('fft')
N = 2048
M = 1024
xx, yy = np.meshgrid(range(N), range(M))

phase = 0.001 * ((xx - M //2)**2 + (yy - N //2)**2)

box = np.zeros((M, N), dtype='float64')
box[M//2-10: M//2+10, N//2-50: N//2+50] = 1
field = Field('box', amplitude=box, phase=phase)

# using numpy
field_fft2 = field.fft2()

field_fft2.intensity_as_dwa().plot('qt')


phase_tensor = torch.tensor(phase, requires_grad=True)

field_tensor = torch.tensor(field.amplitude) * phase_tensor
field_torch_fft2 = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(
    field_tensor)))

print(field_torch_fft2)

loss = torch.sum(torch.abs(field_torch_fft2))
loss.backward()

grad = phase_tensor.grad
print(grad)

field_array_numpy = field_torch_fft2.detach().numpy()

app.exec()