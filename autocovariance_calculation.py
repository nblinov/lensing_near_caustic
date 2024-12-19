import numpy as np
import matplotlib.pyplot as plt
import os

from multiprocessing import Pool, shared_memory
from functools import partial
import tqdm


from lens_map import *
import gaussian_source as src
import random_deflection


arcmin = np.pi/(180*60.)
arcsec = np.pi/(180*60.*60.)
microarcsec = 1e-6*arcsec

pc_in_km = 3.08568e+13
Gpc_in_km = 1e9*pc_in_km
AU_in_km = 1.49598e+08 


# Parameters that determine the lens equation near the caustic - see Schneider, Ehlers and Falco Ch. 6
phi_11, phi_22 = 0.8, 0.0 # This makes the caustic a horizontal line in the source plane

# These are chosen such that the caustic is well approximated by a parabola, i.e. so that we can compare with analytics
phi_111, phi_112, phi_122, phi_222 = 0.06/5., 0.05/5., 0.03/5., 0.05/5.

# This parameter controls the magnification
g = 2 / (phi_11 ** 2 * phi_222)

caustic_params = {'phi_11':phi_11, 'phi_22':phi_22, 'phi_111':phi_111, 'phi_112':phi_112, 'phi_122':phi_122, 'phi_222':phi_222}
ps_params = {'M':2e-6, 'c': 1200, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}



def mardia_skewness(data):
    """
    Compute Mardia's multivariate skewness for a dataset.
    
    Parameters:
    data (numpy.ndarray): A 2D array of shape (n_samples, n_features),
                          where each row is a sample, and each column is a feature.
    
    Returns:
    float: Mardia's skewness measure.
    """
    # Ensure input is a NumPy array
    data = np.asarray(data)
    n_samples, n_features = data.shape
    print("n_samples, n_features = ", n_samples, "\t", n_features)
    
    # Compute the mean vector
    mean_vector = np.mean(data, axis=0)
    #plt.plot(mean_vector)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)
    
    # Compute the inverse of the covariance matrix
    cov_inv = np.linalg.inv(covariance_matrix)
    
    # Center the data
    centered_data = data - mean_vector
    
    # Compute Mardia's skewness
    skewness_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            mahalanobis_ij = centered_data[i] @ cov_inv @ centered_data[j]
            skewness_sum += mahalanobis_ij**3
    
    mardia_skewness = skewness_sum / (n_samples**2)

    chi_squared_mean = n_features*(n_features+1)*(n_features+2)/n_samples
    return mardia_skewness/chi_squared_mean



def pool_func(shared_y1_info, shared_y2_info, pixel_size_lns, rad, src_pos):

    shared_y1_name, y1_shape = shared_y1_info
    shared_y2_name, y2_shape = shared_y2_info
    existing_shm_y1 =  shared_memory.SharedMemory(name=shared_y1_name)
    existing_shm_y2 =  shared_memory.SharedMemory(name=shared_y2_name)
    
    y1 =  np.ndarray(y1_shape,dtype=np.float64,buffer=existing_shm_y1.buf)
    y2 =  np.ndarray(y2_shape,dtype=np.float64,buffer=existing_shm_y2.buf)
    
    lensed_source = src.gaussian_source(y1, y2, rad, c=src_pos)

    # Calculate magnifications
    intensity_norm = 1. # assume it is normalized #np.sum(a_gaussian) * ys * ys
    mag = np.sum(lensed_source) * (pixel_size_lns ** 2) / intensity_norm
    return(mag)

def generate_lightcurve(num_pixel=1001, include_substructure=True):

    # Size and discretization of the lens and source planes
    num_pixel_lns, num_pixel_src = num_pixel, num_pixel # Number of pixels in both planes
    half_size_lns, half_size_src = 4.0, 0.1 # Half size of the lens and source planes
    pixel_size_lns, pixel_size_src = 2.0 * half_size_lns /num_pixel_lns, 2.0 * half_size_src/num_pixel_src # horizontal and vertical physical pixel size
    
    domain_size = 2*half_size_lns
    
    # Random noise - make sure parameters supplied to power spectrum are in radians
    deflection_params = {'pixel_size_in_rad':pixel_size_lns*microarcsec, 'num_pixel':num_pixel_lns, 'area_in_rad_sq': (2.*half_size_lns*microarcsec)**2} 
    #Parameters needed for power spectrum here

    deflection_params['ps_params'] = ps_params
    if include_substructure:
        # the output is in rad, so convert it to microarcsec
        caustic_params['random_deflection'] = random_deflection.generate_random_field(deflection_params)/microarcsec
    else:
        caustic_params['random_deflection'] = None
    
    # indices for the lens plane pixels.
    # Note: arrays are indexed as by [j1,j2], so j2 (j1) indexes horizontal (vertical) directions
    j1, j2 = np.mgrid[0:num_pixel_lns, 0:num_pixel_lns] 
    
    # this maps pixels indices to physical coord as [0,0] -> (-L,L), [0,N-1]-> (L,L) etc
    x1, x2 = pixel_to_pos(j2, -half_size_lns, pixel_size_lns), pixel_to_pos(j1, half_size_lns, -pixel_size_lns)
    
    y1, y2 = lens_mapping(x1, x2, caustic_params)
    
    # Convert deflected coordinates to source plane pixels
    i1, i2 = pos_to_pixel(y2, half_size_src, -pixel_size_src), pos_to_pixel(y1, -half_size_src, pixel_size_src)


    xpos, rad = 0.0, 0.01/3
    ypos_values = np.linspace(0.05,-0.01,400)
    src_positions = np.array([[xpos,ypos] for ypos in ypos_values])
    magnifications = np.zeros_like(ypos_values)
    
    
    # create shared memory so that large static arrays don't get copied to workers
    shm_y1 = shared_memory.SharedMemory(create=True, size=y1.nbytes)
    shm_y2 = shared_memory.SharedMemory(create=True, size=y2.nbytes)
    y1_shared = np.ndarray(y1.shape, dtype=y1.dtype, buffer=shm_y1.buf)
    y1_shared[:] = y1[:]
    y2_shared = np.ndarray(y2.shape, dtype=y2.dtype, buffer=shm_y2.buf)
    y2_shared[:] = y2[:]
    y1_shape = y1.shape
    y2_shape = y2.shape

    magnifications = []
    with Pool(os.cpu_count()) as p:
        #for mag in tqdm.tqdm(p.imap(func=partial(pool_func, [shm_y1.name, y1_shape], [shm_y2.name,y2_shape], rad), iterable=src_positions), total=len(src_positions)):    
        for mag in p.imap(func=partial(pool_func, [shm_y1.name, y1_shape], [shm_y2.name,y2_shape], pixel_size_lns, rad), iterable=src_positions):        
            magnifications.append(mag)
    magnifications = np.array(magnifications)

    return(magnifications)


# ## Autocovariance with substructure


lightcurve_sample = [generate_lightcurve(num_pixel=1501) for i in tqdm.tqdm(range(100))]


ypos_values = np.linspace(0.05,-0.01,400)
rad = 0.01/3
analytic = [np.sqrt(g / rad) * src.analytic_gaussian_source_magnification(-ypos / rad) for ypos in ypos_values]

total_outer = np.zeros((len(lightcurve_sample[0]),len(lightcurve_sample[0])))
means = np.mean(lightcurve_sample,axis=0)
for mag in lightcurve_sample:
    total_outer += np.outer(mag,mag)/len(lightcurve_sample)
autocov_mat = total_outer - np.outer(means,means)

plt.matshow(np.abs(autocov_mat))
plt.colorbar()
plt.title('Autocovariance Matrix')
plt.savefig("autocov.pdf", format="pdf", bbox_inches="tight")
print([np.min(autocov_mat),np.max(autocov_mat)])

# Compute Mardia's skewness
skewness = mardia_skewness(lightcurve_sample)
print(f"Mardia's Skewness: {skewness}")


