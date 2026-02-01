# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 08:28:00 2026

@author: weber
"""
%gui qt
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
import pymodaq_gui

import torch
from torch.nn import MSELoss, Module

from pymodaq_data.h5modules.data_saving import DataLoader
from pymodaq_data.data import DataCalculated, DataToExport

resources = Path(r'C:\Users\weber\Labo\ProgrammesPython\PyMoDAQ_Git\pymodaq_plugins_folder\pymodaq_plugins_optical_2D_shaping_private\src\pymodaq_plugins_optical_2D_shaping\resources')
onepiece_path = resources.joinpath('onepiece.h5')
gaussian_path = resources.joinpath('gaussian_field.h5')
#%%
def square_mask(shape, X0, Y0, SX, SY, dtype=torch.float32):
    """
    shape: (H, W)
    X0, Y0: center (column, row)
    S: lateral size of the square (pixels)
    """
    H, W = shape
    halfX = SX // 2
    halfY = SY // 2
    y = torch.arange(H).view(-1, 1)
    x = torch.arange(W).view(1, -1)
    mask = ((torch.abs(x - X0) <= halfX) & (torch.abs(y - Y0) <= halfY) )
    return mask.to(dtype)


N_slm_x = 1920
N_slm_y = 1080

#%%
dwa_loader = DataLoader(onepiece_path)


for node in dwa_loader.walk_nodes():
    print(node)
    
target_amplitude = dwa_loader.load_data('/RawData/Data00')
target_phase = dwa_loader.load_data('/RawData/Data01')

target_amplitude.plot()
plt.gca().set_aspect('equal')
plt.colorbar()

target_phase.plot()
plt.gca().set_aspect('equal')
plt.colorbar()
#%%
dwa_input_loader = DataLoader(gaussian_path)


input_amplitude = dwa_input_loader.load_data('/RawData/Data00')
    
input_amplitude.plot()
plt.gca().set_aspect('equal')
plt.colorbar()

#%% Normalisation

target_amplitude[0] = (target_amplitude[0] * 
                       np.sqrt(np.sum(input_amplitude[0]**2) / 
                       np.sum(target_amplitude[0]**2)))

target_field = target_amplitude * np.exp(1j * target_phase)

print(np.sum(target_amplitude[0]))
print(np.sum(input_amplitude[0]))

target_amplitude.plot()
plt.gca().set_aspect('equal')
plt.colorbar()

target_field.real().plot()
plt.gca().set_aspect('equal')
plt.colorbar()


#%% prepare tensors and masking

input_amplitude_tensor = torch.from_numpy(input_amplitude[0])
target_field_tensor = torch.from_numpy(target_field[0])


image_field = torch.fft.fftshift(torch.fft.fft2(input_amplitude_tensor))

ratio = torch.sum(torch.abs(image_field)) / torch.sum(torch.abs(input_amplitude_tensor))

mask = square_mask((N_slm_y, N_slm_x),
                   N_slm_x // 2,
                   N_slm_y // 2,
                   N_slm_x-600, N_slm_y-500)

target_field_masked_dwa = target_field.deepcopy()
target_field_masked_dwa[0] *= mask.numpy()

target_field_masked_dwa.real().plot()
plt.gca().set_aspect('equal')
plt.colorbar()


#%% prepare_live_viewer
dwa_image = target_field.deepcopy()
dte = DataToExport('field', data = [target_field.abs(),
                                    target_field.angle()])
                                    
viewer = dte.plot('qt')
#%% prepare optim
Niter = 500
phase = torch.rand(input_amplitude_tensor.shape, requires_grad=True)
optimizer = torch.optim.LBFGS([phase],
                              lr = 1,
                              tolerance_grad=1e-11,
                              tolerance_change=1e-13,
                              line_search_fn='strong_wolfe')

    
fft_tensor = torch.fft.fftshift(torch.fft.fft2(input_amplitude_tensor, norm='forward'))

ratio = torch.sqrt((torch.sum(torch.abs(input_amplitude_tensor)**2) /
                    torch.sum(torch.abs(fft_tensor)**2)))


def closure():
    optimizer.zero_grad()
    input_field = input_amplitude_tensor * torch.exp(1j * phase)
    image_field = ratio * torch.fft.fftshift(torch.fft.fft2(input_field, norm='forward'))

    #diff = torch.abs(image_field) - torch.abs(target_field_tensor)  #amplitude only
    diff = image_field - target_field_tensor  #amplitude and phase
    
    
    loss = torch.mean(torch.abs(diff * mask)**2)
    print(loss.item())
    loss.backward()
    return loss

for ind in range(Niter):
    optimizer.step(closure)    
    print(ind)
    
    input_field = input_amplitude_tensor * torch.exp(1j * phase.detach())
    image_field = torch.fft.fftshift(torch.fft.fft2(input_field))
    image_field = (image_field * 
                   torch.sum(torch.abs(input_amplitude_tensor)) /
                   torch.sum(torch.abs(image_field)))
    

    viewer.show_data(DataToExport('field', data=[
        DataCalculated('image', data=[np.abs(image_field.detach().numpy())]),
        DataCalculated('image', data=[np.angle(image_field.detach().numpy())])
        ]))