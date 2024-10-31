import numpy as np
from scipy.special import kv, iv

def gaussian_source(y1, y2, rad, c=[0.,0.]):
    """
    Normalized intensity of a Gaussian source in the source plane
    Args:
        y1, y2 - positions in the source plane
        rad - radius of the source
        c - location of the center of the source
    """
    a_gaussian = np.exp(-((y1 - c[0])**2 + (y2 - c[1])**2) / (rad**2)) / (np.pi * rad**2)
    return(a_gaussian)

def analytic_gaussian_source_magnification(w):
    """
    Analytic form of the magnification of a source near a fold caustic. The actual magnification is obtained by multiplying this function by sqrt(g/R), where g=2 / (phi_11 ** 2 * phi_222) and R is the source radius
    Args: 
        w - distance of the source from the caustic (in the source plane) in units of source radius R; w = -y/R
    """
    norm = np.pi / 2.
    if w > 0:
        return 0.5 * np.sqrt(w * np.pi / 2.) * np.exp(-0.5 * w ** 2) * kv(0.25, 0.5 * w ** 2) / norm
    else:
        return 0.25 * np.power(np.pi, 1.5) * np.sqrt(-w) * np.exp(-0.5 * w ** 2) * (iv(-0.25, 0.5 * w ** 2) + iv(0.25, 0.5 * w ** 2)) / norm

def asymptotic_source_magnification(w):
    """
    Asymptotic magnification far away from the caustic. The actual magnification is obtained by multiplying this function by sqrt(g/R), where g=2 / (phi_11 ** 2 * phi_222) and R is the source radius
    Args: 
        w - distance of the source from the caustic (in the source plane) in units of source radius R; w = -y/R

    """
    return 0. if w > 0 else 1. / np.sqrt(-w)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    R_sun = 695700 #km
    Gpc_in_km = 3.08568e+22
    arcsec = np.pi/(180*60.*60.)
    g = 1e-4
    source_rad = 100.*R_sun / (1.79*Gpc_in_km) / arcsec
    y_list = np.linspace(-5*source_rad, 5*source_rad, 100)
    y1_list, y2_list = np.meshgrid(y_list,y_list)
    plt.imshow(gaussian_source(y1_list, y2_list,source_rad, c=[0.,2.*source_rad]),origin='lower') 
    plt.show()


    analytic = [np.sqrt(g / source_rad) * analytic_gaussian_source_magnification(-y / source_rad) for y in y_list]
    asymptotic = [np.sqrt(g / source_rad) * asymptotic_source_magnification(-y / source_rad) for y in y_list]

    plt.plot(y_list/source_rad, asymptotic, 'g--', label='Asymptotic Magnification')
    plt.plot(y_list/source_rad, analytic, 'b-', lw=2,label='Analytic Magnification')
    plt.legend()
    plt.xlabel(r'$-y/R$')
    plt.ylabel(r'$\mu$')
    plt.title('Analytic Magnification of a Gaussian Source')
    plt.show()
