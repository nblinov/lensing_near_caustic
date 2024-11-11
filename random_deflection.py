import numpy as np 
import power_spectrum as ps


def generate_random_field(params, compute_jacobian=False):
    """
    Generates a random Gaussian lensing deflection field using a physical clump power spectrum
    Args:
        params - dictionary with the following keys:
        pixel_size_in_rad - pixel size in the lens plane in radians
        num_pixel - number of pixels in the lens plane along one direction 
        area_in_rad_sq - physical area of the lens plane in radians squared
        ps_params - dictionary containing parameters needed to evaluate the power spectrum
        compute_jacobian - flag to compute Jacobian matrix of the random deflection
    Returns:
        [delta_alpha1,delta_alpha2] - random deflection fluctuations along the two plane directions. Each of the list elements is a (num_pixel,num_pixel) array. Note that the deflections have units of radians, which might differ from the units used in the inverse ray shooting!
        jacobian - jacobian of the deflections if the compute_jacobian flag is true. The shape of this is (N,N,2,2)
    """


    # Parameters for the DFT
    delta = params['pixel_size_in_rad'] # spatial spacing in the lens plane in radians
    size = params['num_pixel']

    # Fourier frequencies for the DFT
    ell_x_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta) # same units as 1/pixel_size, e.g., 1/rad
    ell_y_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta) # same units as 1/pixel_size, e.g., 1/rad 
    ell_freq = np.meshgrid(ell_x_freq, ell_y_freq)

    # magnitude of the angular FT coord conjugate to position. same units as 1/pixel_size, e.g., 1/rad
    ell_magnitude = np.sqrt(ell_freq[0]**2 + ell_freq[1]**2 + 1e-50)

    # sqrt amplitude of the angular convergence power spectrum times "volume"; the angular PS should have units rad^2; same units is sqrt(area_in_rad_sq rad^2), e.g. rad^2
    # this sets the magnitude of covergence fluctuations in fourier space; note that the PS units are fixed internally by the PS parameters - that is if we used a different unit for the pixel size, amplitude below would have mixed units
    amplitude = np.sqrt(params['area_in_rad_sq'] * ps.Pkappa_angular(ell_magnitude,params['ps_params']))
    #print("Max delta kappa amplitude = ", np.max(amplitude))

    noise = np.random.normal(size = (size, size)) \
            + 1j * np.random.normal(size = (size, size))

    # the deflections with units pixel_size (l/l^2) * pixel_size^2 (amplitude) /pixel_size^2 (delta^2) = pixel_size, e.g., rad
    delta_alpha1 = np.fft.ifft2((2 * 1j * ell_freq[0] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/delta**2
    delta_alpha2 = np.fft.ifft2((2 * 1j * ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/delta**2
    if compute_jacobian:
        j11 = np.fft.ifft2((2 * 1j * ell_freq[0]*ell_freq[0] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/delta**2
        j12 = np.fft.ifft2((2 * 1j * ell_freq[0]*ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/delta**2
        j22 = np.fft.ifft2((2 * 1j * ell_freq[1]*ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/delta**2
        jac = np.array([[j11,j12],[j12,j22]])
        return(np.array([delta_alpha1,delta_alpha2]), np.moveaxis(jac, [0, 1], [2, 3]))
    else:
        return(np.array([delta_alpha1,delta_alpha2]))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    num_pixel_lns, num_pixel_src = 1001, 101 #Number of pixels in both planes
    half_size_lns, half_size_src = 1e-6, 1e-8 # half size of the lens and source planes
    pixel_size_lns, pixel_size_src = 2.0 * half_size_lns / (num_pixel_lns - 1), 2.0 * half_size_src / (num_pixel_src - 1) # horizontal and vertical physical pixel size

    arcmin = np.pi/(180*60.)
    arcsec = np.pi/(180*60.*60.)
    microarcsec = 1e-6*arcsec

    input_params = {'pixel_size_in_rad':pixel_size_lns*arcsec, 'num_pixel':num_pixel_lns, 'area_in_rad_sq': (2.*half_size_lns*arcsec)**2} 
    #Parameters needed for power spectrum here
    ps_params = {'M':2e-5, 'c': 1200, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
    input_params['ps_params'] = ps_params 

    R_sun = 695700 #km
    au_in_km = 1.49598e+08

    source_rad = 100.*R_sun / (ps_params['DS']*ps.Gpc_in_km) / microarcsec
    print("angular source radius [microarcsec] = ", source_rad)

    halo_info = ps.get_halo_information(ps_params['M'], ps_params['c'])
    print("halo parameters: rs [km], rho_s [Msun/km^3] = ", halo_info[0], "\t", halo_info[2])
    print("peak power occurs at q [1/km] = ", ps.peak_power_q(ps_params), " \t length scale [AU] = ", 2.*np.pi/ps.peak_power_q(ps_params)/au_in_km)
    print("peak power occurs at ell = ", ps.peak_power_ell(ps_params), " \t angular scale [microarcsec] = ", 2.*np.pi/ps.peak_power_ell(ps_params)/microarcsec)
    print("peak power and dimensionless power = ", ps.peak_power(ps_params), "\t", ps.peak_dimensionless_power(ps_params))
    print("characteristic size of deflection fluctuations [microarcsec] = ",  np.sqrt(input_params['area_in_rad_sq'])*np.sqrt(ps.peak_dimensionless_power(ps_params))/np.power(2.*np.pi,1.5)/microarcsec)


    ell_x_freq = np.fabs(2.*np.pi*np.fft.fftfreq(input_params['num_pixel'],d=input_params['pixel_size_in_rad']))
    print("DFT [ell_min/ell_peak,ell_max/ell_peak] = ", [np.min(ell_x_freq)/ps.peak_power_ell(ps_params),np.max(ell_x_freq)/ps.peak_power_ell(ps_params)])
    delta_alpha = generate_random_field(input_params)/microarcsec
    print("Max deflection = ", np.max(delta_alpha))
    
    fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
    pos = ax1.imshow(delta_alpha[0])
    ax1.set_title(r'$\delta\alpha_0(x)$ in microarcsec')
    fig.colorbar(pos, ax=ax1)
    pos = ax2.imshow(delta_alpha[1])
    ax2.set_title(r'$\delta\alpha_1(x)$ in microarcsec')
    fig.colorbar(pos, ax=ax2)
    plt.show()
