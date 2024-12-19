import numpy as np 
import power_spectrum as ps
import tqdm


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
    delta = np.atleast_1d(params['pixel_size_in_rad']) # spatial spacing in the lens plane in radians
    # allow for different pixel size in the horizontal and vertical directions
    if len(delta) == 1:
        delta = np.array([delta[0],delta[0]])
    size = params['num_pixel']

    # Fourier frequencies for the DFT
    ell_x_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta[0]) # same units as 1/pixel_size, e.g., 1/rad
    ell_y_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta[1]) # same units as 1/pixel_size, e.g., 1/rad 
    ell_freq = np.meshgrid(ell_x_freq, ell_y_freq)

    # magnitude of the angular FT coord conjugate to position. same units as 1/pixel_size, e.g., 1/rad
    ell_magnitude = np.sqrt(ell_freq[0]**2 + ell_freq[1]**2 + 1e-50)

    # sqrt amplitude of the angular convergence power spectrum times "volume"; the angular PS should have units rad^2; same units as sqrt(area_in_rad_sq rad^2), e.g. rad^2
    # this sets the magnitude of covergence fluctuations in fourier space; note that the PS units are fixed internally by the PS parameters - that is if we used a different unit for the pixel size, amplitude below would have mixed units
    #
    area_in_rad_sq = delta[0]*delta[1]*params['num_pixel']**2
    amplitude = np.sqrt(area_in_rad_sq * ps.Pkappa_angular(ell_magnitude,params['ps_params']))

    #amplitude = np.sqrt(ps.Pkappa_angular(ell_magnitude,params['ps_params'])/params['area_in_rad_sq'])
    #print("Max delta kappa amplitude = ", np.max(amplitude))

    noise = np.random.normal(size = (size, size)) \
            + 1j * np.random.normal(size = (size, size))

    # the deflections with units pixel_size (l/l^2) * pixel_size^2 (amplitude) /pixel_size^2 (delta^2) = pixel_size, e.g., rad
    delta_alpha1 = np.fft.ifft2((2 * 1j * ell_freq[0] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/(delta[0]*delta[1])
    delta_alpha2 = np.fft.ifft2((2 * 1j * ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/(delta[0]*delta[1])
    if compute_jacobian:
        j11 = np.fft.ifft2((2 * 1j * ell_freq[0]*ell_freq[0] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/(delta[0]*delta[1])
        j12 = np.fft.ifft2((2 * 1j * ell_freq[0]*ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/(delta[0]*delta[1])
        j22 = np.fft.ifft2((2 * 1j * ell_freq[1]*ell_freq[1] / np.power(ell_magnitude, 2) ) * noise * amplitude).real/(delta[0]*delta[1])
        jac = np.array([[j11,j12],[j12,j22]])
        return(np.array([delta_alpha1,delta_alpha2]), np.moveaxis(jac, [0, 1], [2, 3]))
    else:
        return(np.array([delta_alpha1,delta_alpha2]))

def generate_random_convergence_field(params):
    """
    Generates a random Gaussian convergence field using a physical clump power spectrum. This function is not used directly to generate lens maps, but rather helps check normalizations of various quantities in testing.
    Args:
        params - dictionary with the following keys:
        pixel_size_in_rad - pixel size in the lens plane in radians
        num_pixel - number of pixels in the lens plane along one direction 
        area_in_rad_sq - physical area of the lens plane in radians squared
        ps_params - dictionary containing parameters needed to evaluate the power spectrum
    Returns:
        delta_kappa - random convergence fluctuations 
    """


    # Parameters for the DFT
    delta = np.atleast_1d(params['pixel_size_in_rad']) # spatial spacing in the lens plane in radians
    # allow for different pixel size in the horizontal and vertical directions
    if len(delta) == 1:
        delta = np.array([delta[0],delta[0]])
    size = params['num_pixel']

    # Fourier frequencies for the DFT
    ell_x_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta[0]) # same units as 1/pixel_size, e.g., 1/rad
    ell_y_freq = 2.*np.pi*np.fft.fftfreq(size,d=delta[1]) # same units as 1/pixel_size, e.g., 1/rad 
    ell_freq = np.meshgrid(ell_x_freq, ell_y_freq)

    # magnitude of the angular FT coord conjugate to position. same units as 1/pixel_size, e.g., 1/rad
    ell_magnitude = np.sqrt(ell_freq[0]**2 + ell_freq[1]**2 + 1e-50)

    # sqrt amplitude of the angular convergence power spectrum times "volume"; the angular PS should have units rad^2; same units as sqrt(area_in_rad_sq rad^2), e.g. rad^2
    # this sets the magnitude of covergence fluctuations in fourier space; note that the PS units are fixed internally by the PS parameters - that is if we used a different unit for the pixel size, amplitude below would have mixed units
    # factor of 0.5 because amplitude sets the standard deviations of the real and imaginary parts of the desnity contrast. 
    area_in_rad_sq = delta[0]*delta[1]*params['num_pixel']**2
    amplitude = np.sqrt(area_in_rad_sq * ps.Pkappa_angular(ell_magnitude,params['ps_params'])/(delta[0]*delta[1])**2)

    """
    # this is a trick to get fourier coefficients that give rise to a real function when inverse transformed
    white_noise_in_position_space = np.random.normal(size = (size, size))
    # after normalizing by the white noise PS, fourier coefficients are Gaussian random vars with variance 1
    noise = np.fft.fft2(white_noise_in_position_space)/np.sqrt(size**2)
    """
    # another way to generate the fourier coefficients of a real function is to generate unconstrained fourier coefficients
    # and then take the real part after the fourier transform
    noise = np.random.normal(size = (size, size)) \
            + 1j * np.random.normal(size = (size, size))


    # the field should be real by definition; take real part to get rid of any numerical error-induced imaginary parts
    delta_kappa = np.fft.ifft2(noise * amplitude).real 
    """
    ell_bins = np.linspace(ps.peak_power_ell(params['ps_params'])/10, ps.peak_power_ell(params['ps_params'])*10,50)
    ell_diff = np.diff(ell_bins)[0]
    #Pell_est = get_ps_estimate(np.abs(noise)*amplitude*(delta[0]*delta[1]),ell_magnitude, ell_bins, ell_diff, area_in_rad_sq )
    Pell_est = get_ps_estimate(np.abs(np.fft.fft2(delta_kappa))*(delta[0]*delta[1]),ell_magnitude, ell_bins, ell_diff, area_in_rad_sq )
    plt.plot(ell_bins,Pell_est * ell_bins**2 / 2 / np.pi)
    plt.plot(ell_bins,ps.Pkappa_angular(ell_bins,params['ps_params']) * ell_bins**2 / 2 / np.pi, '--')
    plt.xscale('log')
    plt.yscale('log')
    """
    return(delta_kappa)

def get_ps_estimate(dk_fourier,ell_magnitude, ell_bins, ell_diff, domain_volume):
    # this assumes that dk_fourier are the dimensionful fourier coefficients, only then does the PS has the right dimensions
    Pell = np.zeros(len(ell_bins))
    for i, ell in enumerate(ell_bins):
        match = np.where((ell_magnitude > ell-ell_diff/2) & (ell_magnitude <= ell+ell_diff/2))
        # The expectation value is < |dk(ell)|^2 > = P(\ell) V, so we need to divide by V 
        Pell[i] =  np.mean(dk_fourier[match] ** 2)/ domain_volume
    return(Pell)

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    num_pixel_lns = 2001 #Number of pixels in lens plane
    half_size_lns = 1e-5 # half size of the lens plane
    pixel_size_lns = 2.0 * half_size_lns / num_pixel_lns # horizontal and vertical physical pixel size

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

    # Reconstruct the convergence power spectrum from the simlation
    # a la https://nkern.github.io/posts/2024/grfs_and_ffts/
    delta_kappa = generate_random_convergence_field(input_params)
    delta_kappa_dft = np.abs(np.fft.fft2(delta_kappa)) * input_params['pixel_size_in_rad']**2 # this factor is needed to get delta(ell) without any prefactors associated with finite box size 
    print(delta_kappa_dft)
    # For white noise, the PS is V, and < |dk(ell)|^2 > = P(\ell) V = V^2, so dk(ell) ~ V
    print("Scale of white noise-only fluctuations of fourier modes = ", input_params['area_in_rad_sq'])
    fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
    dk = ax1.imshow(delta_kappa)
    fig.colorbar(dk)
    ax1.set_title(r"Convergence Fluctuations $\delta\kappa(x)$")
    dk_dft = ax2.imshow(delta_kappa_dft)
    fig.colorbar(dk_dft)
    ax2.set_title(r"Convergence Fluctuations $|\delta\kappa(\ell)|$")
    plt.show()

    ell_x_freq = 2.*np.pi*np.fft.fftfreq(input_params['num_pixel'],d=input_params['pixel_size_in_rad'])
    ell_y_freq = 2.*np.pi*np.fft.fftfreq(input_params['num_pixel'],d=input_params['pixel_size_in_rad'])

    ell_bins = np.linspace(ps.peak_power_ell(ps_params)/10, ps.peak_power_ell(ps_params)*10,50)
    ell_freq = np.meshgrid(ell_x_freq, ell_y_freq)
    ell_magnitude = np.sqrt(ell_freq[0]**2 + ell_freq[1]**2 + 1e-50)

    ell_diff = np.diff(ell_bins)[0]
    print("dl = ", ell_diff)
    #print(ell_bins)
    num_realizations = 10
    Pell_avg = np.zeros(len(ell_bins))
    for i in tqdm.tqdm(range(num_realizations)):
        delta_kappa = generate_random_convergence_field(input_params)
        delta_kappa_dft = np.abs(np.fft.fft2(delta_kappa)) * input_params['pixel_size_in_rad']**2 # this factor is needed to convert the dimensionless DFT coefficient into a Fourier mode delta(ell) with with proper dimensions 
        Pell_avg += get_ps_estimate(delta_kappa_dft,ell_magnitude, ell_bins, ell_diff, input_params['area_in_rad_sq'])
    Pell_avg /= num_realizations

    plt.plot(ell_bins, Pell_avg*ell_bins**2 / 2 / np.pi, label='Estimated PS')
    pc_in_km = 3.08568e+13
    Gpc_in_km = 1e9*pc_in_km
    rs = halo_info[0]
    Pell_input = ps.Pkappa_angular(ell_bins,ps_params)
    plt.plot(ell_bins, Pell_input*ell_bins**2 / 2 / np.pi, '--', label='Input PS')
    plt.axvline(x=2.*np.pi/np.sqrt(input_params['area_in_rad_sq']))
    plt.axvline(x=2.*np.pi/input_params['pixel_size_in_rad'])

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\ell^2 P_\kappa(\ell)/(2\pi)$', fontsize = 20)
    plt.xlabel(r'$\ell$', fontsize = 20)

    plt.legend()
    plt.show()

