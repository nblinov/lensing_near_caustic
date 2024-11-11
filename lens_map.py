import numpy as np
def lens_mapping(x1, x2, params):
    """
    Args:
        x1, x2: lens plane coordinates
        params: dictionary parameters of the fold caustic; these are given by derivatives of the Fermat potential phi, see Ch. 6 in  Schneider, Ehlers and Falco 
    Returns:
        y1, y2: source plane coordinates
    """
    p = params
    y1 = 0.5 * x1 ** 2 * p['phi_111'] + x1 * x2 * p['phi_112'] + x1 * p['phi_11']+ 0.5 * x2 ** 2 * p['phi_122']
    y2 = 0.5 * x1 ** 2 * p['phi_112'] + x1 * x2 * p['phi_122'] + 0.5 * x2 ** 2 * p['phi_222'] + x2 * p['phi_22']
    if params['random_deflection'] is not None:
        y1 += params['random_deflection'][0]
        y2 += params['random_deflection'][1]
    return y1, y2

def lens_mapping_jacobian(x1, x2, params):
    """
    Analytically-computed Jacobian of len_mapping. The returned shape is (len(x1),len(x2),2,2).
    """
    p = params
    jac = np.array([[p['phi_11'] + x1*p['phi_111'] + x2*p['phi_112'], x1*p['phi_112'] + x2*p['phi_122']],
                     [x1*p['phi_112'] + x2*p['phi_122'], x1*p['phi_122'] + p['phi_22'] + x2*p['phi_222']]])
    return(np.moveaxis(jac, [0, 1], [2, 3]))

def pixel_to_pos(pixel, origin, pixel_size):
    return origin + pixel * pixel_size

def pos_to_pixel(coord, origin, pixel_size):
    return np.rint((coord - origin) / pixel_size).astype(int)
