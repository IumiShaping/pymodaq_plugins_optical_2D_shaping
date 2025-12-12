import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

mega =  1E6
centi = 1E-2
milli = 1E-3
micro = 1E-6
nano = 1E-9
pico = 1E-12
femto = 1E-15




def load_and_process_image(path, size = (1920, 1080)):
    image = Image.open(path)
    image_resized = image.resize(size)
    return np.asarray(image_resized.convert('L'))



def generate_checkerboard(Nx=1920, Ny=1080, block_size=20):

    y, x = np.indices((Ny, Nx))
    

    mask = ((x // block_size) + (y // block_size)) % 2
    
    return mask

def generate_circular_aperture_physical(x_F, y_F, r):

    distance = np.sqrt(x_F**2 + y_F**2)
    return (distance <= r).astype(float)



final_target_amplitude = load_and_process_image(r"C:\Users\varollo\Desktop\PhD\Spatial_Light_modulation\Control_Phase_amplitude\main_codes\targets\Einstein.jpg")
final_target_phase = load_and_process_image(r"C:\Users\varollo\Desktop\PhD\Spatial_Light_modulation\Control_Phase_amplitude\main_codes\targets\poulet.png")

final_target_amplitude = final_target_amplitude/np.max(final_target_amplitude)
final_target_phase = final_target_phase/np.max(final_target_phase)

phi = final_target_phase*np.pi # A verifier ?
beta = np.arccos(final_target_amplitude/2)

print(np.min(phi), np.max(phi), np.min(beta), np.max(beta))
theta1 = phi + beta
theta2 = phi - beta

#We generate M1 and M2
p = 5
M1 = generate_checkerboard(1920,1080,p)
M2 = 1 - M1


alpha = (M1*theta1 + M2*theta2)
alpha = np.mod(alpha, 2*np.pi)


Field_at_SLM = np.exp(1j*alpha)
Field_at_FP1 = np.fft.fftshift(np.fft.fft2(Field_at_SLM))

# --------------------------------------------------------------
# Axes du plan image (SLM)
# --------------------------------------------------------------
Nx = 1920
Ny = 1080
apix = 8*micro
x = (np.arange(Nx) - Nx/2) * apix
y = (np.arange(Ny) - Ny/2) * apix

# --------------------------------------------------------------
# Axes du plan de Fourier
# --------------------------------------------------------------
wavelength = 515*nano
# Échantillonnage spatial pour FFT :
dx = apix
dy = apix

# Résolutions fréquentielles
fx = np.fft.fftshift(np.fft.fftfreq(Nx, dx))
fy = np.fft.fftshift(np.fft.fftfreq(Ny, dy))

f1 = 100*milli

x_F = wavelength * f1 * fx
y_F = wavelength * f1 * fy

# On crée grilles pour pouvoir tracer et filtrer
X_F, Y_F = np.meshgrid(x_F, y_F)


# --------------------------------------------------------------
# Filtre physique dans le plan de Fourier
# --------------------------------------------------------------
radius_um = 700*micro  # exemple : r = 50 microns
aperture_at_FP1 = generate_circular_aperture_physical(X_F, Y_F, radius_um)

# Filtrage
Field_filtered = Field_at_FP1 * aperture_at_FP1


# --------------------------------------------------------------
# Retour après la deuxième lentille
# --------------------------------------------------------------
Field_after_lens2 = np.fft.ifft2(np.fft.ifftshift(Field_filtered))

# --------------------------------------------------------------
# AFFICHAGES
# --------------------------------------------------------------

camera_Ny, camera_Nx = 1080, 1440
camera_pixel_size = 3.45 #pixel size in µm

x_camera = np.linspace(-camera_Nx//2, camera_Nx//2, camera_Nx)*camera_pixel_size
y_camera = np.linspace(-camera_Ny//2, camera_Ny//2, camera_Ny)*camera_pixel_size

extent_camera = np.array([np.min(x_camera), np.max(x_camera), np.min(y_camera), np.max(y_camera)])


extent_FourierPlane = [x_F[0]*1e6, x_F[-1]*1e6, y_F[0]*1e6, y_F[-1]*1e6]

plt.figure()
plt.title("Amplitude dans le plan de Fourier (axes physiques)")
plt.imshow(np.abs(Field_at_FP1), extent=extent_FourierPlane, origin='lower', vmin = 0, vmax = 1e4)
plt.xlabel("x_F (µm)")
plt.ylabel("y_F (µm)")
plt.colorbar()
plt.contour(X_F*1e6, Y_F*1e6, aperture_at_FP1, colors='r', levels=[0.5])
plt.show()


plt.figure()
plt.title("Amplitude du champ reconstruit")
plt.imshow(np.abs(Field_after_lens2)/np.max(np.abs(Field_after_lens2)), extent = extent_camera)
plt.colorbar()
plt.show()

plt.figure()
plt.title("Phase du champ reconstruit")
plt.imshow(np.angle(Field_after_lens2), cmap='hsv', extent=extent_camera)
plt.colorbar()
plt.show()

