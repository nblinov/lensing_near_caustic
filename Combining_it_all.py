import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
import gaussian_random_fields as gr

# Phi parameters
phi_11, phi_22 = 0.8, 0.0
phi_111, phi_112, phi_122, phi_222 = 0.6, 0.5, 0.3, 0.5

g = 2 / (phi_11 ** 2 * phi_222)

def lens_mapping(x1, x2, random_field_x1=None, random_field_x2=None):
    y1 = 0.5 * x1 ** 2 * phi_111 + x1 * x2 * phi_112 + x1 * phi_11 + 0.5 * x2 ** 2 * phi_122
    y2 = 0.5 * x1 ** 2 * phi_112 + x1 * x2 * phi_122 + 0.5 * x2 ** 2 * phi_222 + x2 * phi_22
    if random_field_x1 is not None and random_field_x2 is not None:
        y1 += random_field_x1
        y2 += random_field_x2
    return y1, y2

def generate_gaussian_random_field(alpha, size):
    return gr.gaussian_random_field(alpha, size, True)

def pixel_to_pos(pixel, origin, pixel_size):
    return origin + pixel * pixel_size

def pos_to_pixel(coord, origin, pixel_size):
    return np.rint((coord - origin) / pixel_size).astype(int)

# Parameters
nx, ny = 401, 401
xl, yl = 3.0, 3.0
xs, ys = 2.0 * xl / (nx - 1), 2.0 * yl / (ny - 1)
xpos, rad = 0.0, 0.01
ypos_values = np.linspace(-1.0, 1.0, 400)
magnifications, magnifications_field = [], []

# Generate Gaussian random fields
random_field_x1 = generate_gaussian_random_field(1, nx)
random_field_x2 = generate_gaussian_random_field(1, nx)

# Pre-compute grids and Gaussian source
j1, j2 = np.mgrid[0:nx, 0:nx]
x1, x2 = pixel_to_pos(j2, -xl, xs), pixel_to_pos(j1, -xl, xs)
x = np.linspace(-yl, yl, ny)
y = np.linspace(-yl, yl, ny)
X, Y = np.meshgrid(x, y)

for ypos in ypos_values:
    a_gaussian = np.exp(-((X - xpos)**2 + (Y - ypos)**2) / (2 * rad**2)) / (2. * np.pi * rad**2)
    
    # Lens mappings
    y1, y2 = lens_mapping(x1, x2)
    y1_field, y2_field = lens_mapping(x1, x2, 0.01 * random_field_x1, 0.01 * random_field_x2)
    
    # Convert deflected coordinates to source plane pixels
    i1, i2 = pos_to_pixel(y2, -yl, ys), pos_to_pixel(y1, -yl, ys)
    i1_field, i2_field = pos_to_pixel(y2_field, -yl, ys), pos_to_pixel(y1_field, -yl, ys)
    
    # Filter valid rays
    ind = (i1 >= 0) & (i1 < ny) & (i2 >= 0) & (i2 < ny)
    ind_field = (i1_field >= 0) & (i1_field < ny) & (i2_field >= 0) & (i2_field < ny)
    
    # Initialize image planes
    b, b_field = np.zeros((nx, nx)), np.zeros((nx, nx))
    
    # Calculate magnification without random field
    np.add.at(b, (j1[ind], j2[ind]), a_gaussian[i1[ind], i2[ind]])
    
    # Calculate magnification with random field
    np.add.at(b_field, (j1[ind_field], j2[ind_field]), a_gaussian[i1_field[ind_field], i2_field[ind_field]])

    # Calculate fluxes and magnifications
    fa_gaussian = np.sum(a_gaussian) * ys * ys
    fb = np.sum(b) * (xs ** 2)
    fb_field = np.sum(b_field) * (xs ** 2)
    magnifications.append(fb / fa_gaussian)
    magnifications_field.append(fb_field / fa_gaussian)

# Analytic and asymptotic solutions
def gaussian_source_magnification(w):
    print(f"w value: {w}")
    norm = np.pi / 2.
    if w > 0:
        return 0.5 * np.sqrt(w * np.pi / 2.) * np.exp(-0.5 * w ** 2) * kv(0.25, 0.5 * w ** 2) / norm
    else:
        return 0.25 * np.power(np.pi, 1.5) * np.sqrt(-w) * np.exp(-0.5 * w ** 2) * (iv(-0.25, 0.5 * w ** 2) + iv(0.25, 0.5 * w ** 2)) / norm
    
def asymptotic_source_magnification(w):
    return 0. if w > 0 else 1. / np.sqrt(-w)

analytic = [np.sqrt(g / rad) * gaussian_source_magnification(-ypos / rad) for ypos in ypos_values]
asymptotic = [np.sqrt(g / rad) * asymptotic_source_magnification(-ypos / rad) for ypos in ypos_values]

# Plotting
plt.figure()
plt.plot(ypos_values, magnifications, 'b-', label='Magnification without Random Field')
plt.plot(ypos_values, magnifications_field, 'g-', label='Magnification with Random Field')
plt.plot(ypos_values, analytic, '--', label="Analytic solution for Gaussian Source")
plt.plot(ypos_values, asymptotic, 'r--', label="Asymptotic solution for Gaussian Source")
plt.xlabel('Distance from Caustic (ypos)')
plt.ylabel('Value')
plt.title('Total Magnification vs Vertical Distance from Caustic')
plt.legend()
plt.grid(True)
plt.show()
