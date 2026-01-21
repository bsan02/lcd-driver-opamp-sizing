"""
Python based Gm/Id lookup
Aviral Pandey, 2023

Built from look_up function by Boris Murmuann
"""

from scipy.io import loadmat
import scipy.interpolate as interp
import numpy as np
import numbers


# Output vars that are in the table
_out_vars = ['W', 'ID', 'VT', 'IGD', 'IGS', 'GM', 'GMB', 'GDS', 
             'CGG', 'CGS', 'CGD', 'CDG', 'CGB', 'CDD', 'CSS', 'STH', 'SFL']
# Input vars that are swept
_in_vars = ['L', 'VGS', 'VDS', 'VSB']
# Misc info vars
_info_vars = ['INFO', 'CORNER', 'TEMP', 'NFING']
_ignore_keys = ['__version__', '__header__', '__globals__']


def _build_interpolator(var, l, vgs, vds, vsb):
    return interp.RegularGridInterpolator((l, vgs, vds, vsb), var)


def importdata(filename):
    """
    Imports the data similar to the Matlab function. 
    The resulting data is a dictionary from key to numpy arrays
    """
    out_dict = loadmat(filename, simplify_cells=True)
    new_dict = dict()
    for key in out_dict:
        if key not in _ignore_keys and isinstance(new_dict, dict):
            new_dict = out_dict[key]
            break
    if new_dict is None:
        raise ValueError("Could not load mat")
    new_dict['W'] = new_dict['W'] * np.ones(new_dict['ID'].shape)

    l = new_dict['L']
    vgs = new_dict['VGS']
    vds = new_dict['VDS']
    vsb = new_dict['VSB']
    for out_var in _out_vars:
        new_dict[out_var] = _build_interpolator(
                new_dict[out_var], l, vgs, vds, vsb)
    return new_dict


def _correct_arr(var, default):
    """
    The inputs could be specified as a number, a list, or a numpy array
    This function turns any of those into a numpy array
    """
    var = default if var is None else var
    if isinstance(var, np.ndarray):
        return var
    elif isinstance(var, list):
        return np.array(var)
    elif isinstance(var, numbers.Number):
        return np.array([var])
    else:
        raise ValueError(f"Invalid value specified {var}")


def _look_up_basic(data_dict, out_var, vgs=None, vds=None, vsb=None, l=None):
    # Default values for all the inputs and convert to numpy arrays
    l = _correct_arr(l, min(data_dict['L']))
    vgs = _correct_arr(vgs, data_dict['VGS'])
    vds = _correct_arr(vds, max(data_dict['VDS'])/2)
    vsb = _correct_arr(vsb, 0)

    # Setup the x values that we'll interpolate over
    x_shape = (l.shape[0], vgs.shape[0], vds.shape[0], vsb.shape[0], 4)
    x = np.zeros(x_shape)
    # TODO There's gotta be a better way to do this
    for l_idx, l_val in enumerate(l):
        for vgs_idx, vgs_val in enumerate(vgs):
            for vds_idx, vds_val in enumerate(vds):
                for vsb_idx, vsb_val in enumerate(vsb):
                    x[l_idx, vgs_idx, vds_idx, vsb_idx] = np.array(
                            [l_val, vgs_val, vds_val, vsb_val])
    
    # Check if out_var is supposed to be a ratio
    if '_' in out_var:
        out_vars = out_var.split('_')
        num = data_dict[out_vars[0]](x)
        den = data_dict[out_vars[1]](x)
        return num / den
    else:
        return data_dict[out_var](x)


def look_up_basic(data_dict, out_var, vgs=None, vds=None, vsb=None, l=None):
    return np.squeeze(_look_up_basic(data_dict, out_var, vgs, vds, vsb, l))


def _look_up_vs_ratio(data_dict, yvar, x_name, x_desired, 
                      vgs=None, vds=None, vsb=None, l=None, lim_left=False, lim_right=False):
    # First, turn gm_id into numpy array if it's not already
    x_desired = _correct_arr(x_desired, None)
    # x_calc is an array of x_name vs vgs
    x_calc = look_up_basic(data_dict, x_name, vgs, vds, vsb, l) 
    y_calc = look_up_basic(data_dict, yvar, vgs, vds, vsb, l)
    # Make sure it's possible to achieve the targets
    if min(x_desired) < min(x_calc) or max(x_desired) > max(x_calc):
        raise ValueError(
            f"Unable to achieve a {x_name} over vgs search that meets target")
    # Now we interpolate across the values
    if lim_left:
        max_idx = np.argmax(x_calc)
        x_calc = x_calc[:max_idx+1]
        y_calc = y_calc[:max_idx+1]
    if lim_right:
        max_idx = np.argmax(x_calc)
        x_calc = x_calc[max_idx:]
        y_calc = y_calc[max_idx:]
    interp_func = interp.interp1d(x_calc, y_calc, kind='cubic')
    return interp_func(x_desired)


def _look_up_vs_ratio_swp(data_dict, yvar, x_name, x_desired, 
                          vgs=None, vds=None, vsb=None, l=None, **kwargs):
    # Set defaults and make arrays if required
    l = _correct_arr(l, min(data_dict['L']))
    vds = _correct_arr(vds, max(data_dict['VDS'])/2)
    vsb = _correct_arr(vsb, 0)
    x_desired = _correct_arr(x_desired, x_desired)
    # This will be the points we search  between
    vgs = data_dict['VGS']
    # Iterate over all the search points
    out = np.zeros((len(l), len(vds), len(vsb), len(x_desired)))
    for l_idx, l_val in enumerate(l):
        for vds_idx, vds_val in enumerate(vds):
            for vsb_idx, vsb_val in enumerate(vsb):
                out[l_idx, vds_idx, vsb_idx, :] = _look_up_vs_ratio(
                        data_dict, yvar, x_name, x_desired, vgs, vds_val, vsb_val, l_val, **kwargs)
    return np.squeeze(out)


def look_up_vs_gm_id(data_dict, yvar, gm_id,
                     vgs=None, vds=None, vsb=None, l=None):
    return _look_up_vs_ratio_swp(
            data_dict, yvar, "GM_ID", gm_id, vgs, vds, vsb, l, lim_right=True)


def look_up_vs_gm_cgg(data_dict, yvar, gm_cgg,
                      vgs=None, vds=None, vsb=None, l=None):
    return _look_up_vs_ratio_swp(
            data_dict, yvar, "GM_CGG", gm_cgg, vgs, vds, vsb, l, lim_left=True)


def look_up_vs_id_w(data_dict, yvar, id_w,
                    vgs=None, vds=None, vsb=None, l=None):
    return _look_up_vs_ratio_swp(
            data_dict, yvar, "ID_W", id_w, vgs, vds, vsb, l)


def look_up_vgs_vs_gm_id(data_dict, gm_id, vds=None, vsb=None, l=None):
    # Set defaults and make arrays if required
    l = _correct_arr(l, min(data_dict['L']))
    vds = _correct_arr(vds, max(data_dict['VDS'])/2)
    vsb = _correct_arr(vsb, 0)
    gm_id = _correct_arr(gm_id, gm_id)
    # This will be the points we search  between
    vgs = data_dict['VGS']
    # Iterate over all the search points
    out = np.zeros((len(l), len(vds), len(vsb), len(gm_id)))
    for l_idx, l_val in enumerate(l):
        for vds_idx, vds_val in enumerate(vds):
            for vsb_idx, vsb_val in enumerate(vsb):
                gm_id_calc = look_up_basic(data_dict, 'GM_ID', vgs, vds_val, vsb_val, l_val) 
                interp_func = interp.interp1d(gm_id_calc, vgs, kind='cubic')
                out[l_idx, vds_idx, vsb_idx, :] = interp_func(gm_id)
    return np.squeeze(out)

