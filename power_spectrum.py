import sys, os
import numpy as np
from scipy import special

from astropy.cosmology import Planck18
from astropy import units
from IPython.display import clear_output

from functools import partial

pc_in_km = 3.08568e+13
Gpc_in_km = 1e9*pc_in_km
AU_in_km = 1.49598e+08 

# Present day critical density
rho_c = Planck18.critical_density0.to('M_sun/km^3').value # Solar masses / km^3

def get_halo_information(M, c):
    """
    Args: 
        M - virial mass, or M200. Mass of the region which encloses an average density of 200 times the critical density.
        c - concentration parameter
    Returns:
        rs - the NFW scale radius
        rMax - the virial or maximum radius of the halo, c*rs
        rho_s - scale density 
        
    """
    rs = ((3*M)/(800*c**3*np.pi*rho_c))**(1./3.) # km
    rMax = c*rs # km
    rho_s = ((1 + c)*M)/(16.*np.pi*rs**3*(-c + np.log(1 + c) + c*np.log(1 + c))) # Solar masses/ km^3
    Ms = 8.*np.pi*rs**3*rho_s*(np.log(4.) - 1)
    return rs, rMax, rho_s, Ms # [km, km, Solar Masses / km^3, Solar mass]

def _g_truncated_nfw(x, c, asymptotic_switch=1e5):
    """
    A function needed for the Fourier transform of the truncated NFW density profile as a function of x=c*r_s, with the truncation radius at c*r_s
    """

    if c*x < asymptotic_switch:
        si_x, ci_x = special.sici(x)
        si_1_plus_cx, ci_1_plus_cx = special.sici((1 + c) * x)
        return -np.sin(c * x)/((1 + c) * x) \
            - np.cos(x) * (ci_x - ci_1_plus_cx) \
            - np.sin(x) * (si_x - si_1_plus_cx)
    else:
        return (1.-np.cos(c*x)/(1 + c)**2) / x**2
g_truncated_nfw = np.vectorize(_g_truncated_nfw)

def rhoTildeSq(q, M, c):
    """
    Fourier transform of the truncated NFW profile
    Args:
        q: Fourier wavenumber, conjugate to the physical length r
        M: virial mass of the minihalo in solar masses
        c: concentration parameter
    """

    rs, rMax, rho_s, Ms = get_halo_information(M, c)
    x = q * rs

    result = (16 * np.pi * rs**3 * rho_s * g_truncated_nfw(x, c))**2

    return result

def Pkappa(q, params):
    M = params['M']
    c = params['c']
    DL = params['DL'] # Gpc
    DS = params['DS'] # Gpc
    DLS = params['DLS'] # Gpc
    f = params['dm_mass_fraction']



    Sigma_cr = 1.74e-24 / (DL*DLS/DS) # [M_Sun / km^2]
    Sigma_cl = 0.83*Sigma_cr # [M_Sun / km^2]

    return f * Sigma_cl / Sigma_cr**2  * rhoTildeSq(q, M=M, c = c)/M

def Pkappa_angular(l,params):
    DL = params['DL']*Gpc_in_km
    return(Pkappa(l/DL, params)/DL**2)

def prefactor(params):
    """
    Debugging function to compare certain quantities to mathematica verions
    Note that there can be differences between halo mass definitions that lead to O(few) differences in power spectrum differences
    """
    rs, rMax, rho_s, Ms = get_halo_information(params['M'], params['c'])
    q = peak_power_q(params)
    # should be about 0.35
    #return(q * rs * g_truncated_nfw(q*rs, c))
    DL = params['DL'] # Gpc
    DS = params['DS'] # Gpc
    DLS = params['DLS'] # Gpc
    f = params['dm_mass_fraction']



    Sigma_cr = 1.74e-24 / (DL*DLS/DS) # [M_Sun / km^2]
    Sigma_cl = 0.83*Sigma_cr # [M_Sun / km^2]

    return(np.sqrt(f * Sigma_cl / Sigma_cr**2  * (16 * np.pi * rs**3 * rho_s)**2 / (2.*np.pi) / rs**2)/params['M'])
    
def peak_power_q(params):
    """
    Returns the location of the peak power in 1/km
    """
    rs = get_halo_information(params['M'], params['c'])[0]
    return(0.77/rs)

def peak_power_ell(params):
    """
    Returns the angular location of the peak power in radians
    """
    rs = get_halo_information(params['M'], params['c'])[0]
    return(0.77*params['DL']*Gpc_in_km/rs)

def peak_power(params):
    """
    Returns the power at the peak of the dimensionless power spectrum q^2 P(q)/2pi
    """
    qpeak = peak_power_q(params)
    return(Pkappa(qpeak, params))

def peak_dimensionless_power(params):
    """
    Returns the value of the peak of the dimensionless powerspectrum q^2 P(q)/2pi
    """
    qpeak = peak_power_q(params)
    return(qpeak**2 * Pkappa(qpeak, params)/(2.*np.pi))

def Pkappa_angular_gaussian(l,params):
    """
    Toy power spectrum consisting of a Gaussian at the "right" location
    """

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams['text.usetex'] = True

    for c in [10, 100, 1000]:
        params = {'M':1e-6, 'c': c, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
        qPerp = np.geomspace(1e-20, 1e11, 1000)
        pkappa = [q**2 / 2 / np.pi * Pkappa(q,params) for q in qPerp]
        halo_info = get_halo_information(params['M'], params['c'])
        k_rs = halo_info[0]*qPerp
        plt.plot(k_rs, pkappa, label = '$c = '+str(c)+'$')
        print(prefactor(params))
        print("Halo params rs [AU], rhos [M_Sun/pc^3], Ms [MSun] = ", halo_info[0]/AU_in_km, "\t", halo_info[2]*pc_in_km**3, "\t", halo_info[3])
        print("Peak occurs at q [1/AU] = ", peak_power_q(params)*AU_in_km)
        print("Peak sqrt(q^2 P(q)/2pi) = ", np.sqrt(peak_dimensionless_power(params)))

    plt.ylim(float(pkappa[0]), 1e-5)
    plt.xlim(k_rs[0], 1e11)
    plt.axvline(1, color = 'black', ls = '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$q^2 P_\kappa/(2\pi)$', fontsize = 20)
    plt.xlabel(r'$q r_s$', fontsize = 20)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for c in [10, 100, 1000]:
        params = {'M':1e-6, 'c': c, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
        rs = get_halo_information(params['M'], params['c'])[0]
        lPerp = np.geomspace(1e-10*params['DL']*Gpc_in_km/rs, 1e10*params['DL']*Gpc_in_km/rs, 1000)
        l_rs_over_DL = lPerp*rs/(params['DL']*Gpc_in_km)
        pkappa = [l**2 / 2 / np.pi * Pkappa_angular(l,params) for l in lPerp]
        pkappa = [1 / 2 / np.pi * Pkappa_angular(l,params) for l in lPerp]
        plt.plot(l_rs_over_DL, pkappa, label = '$c = '+str(c)+'$')


    #plt.ylim(float(pkappa[0]), 1e-5)
    # #plt.xlim(l_rs_over_DL[0], 1e11)
    plt.axvline(1, color = 'black', ls = '--')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$\ell^2 P_\kappa(\ell)/(2\pi)$', fontsize = 20)
    plt.xlabel(r'$\ell r_s / D_L$', fontsize = 20)
    plt.legend()
    plt.tight_layout()

    # #plt.savefig("graphics/angular_convergence_ps.pdf", bbox_inches='tight')


    plt.show()
