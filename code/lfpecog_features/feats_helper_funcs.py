"""
Helper function for Data Management and
Feature Extraction
"""

# Import public packages and functions
import numpy as np
# from scipy.ndimage import uniform_filter1d


def normalize_var_fts(values):
    """
    Normalise list or (nd)-array of values
    """
    if type(values) == list:
        np.array(values)

    if len(values.shape) == 1:
        ft_max = np.nanmax(values)
        ft_out = values / ft_max
    
    elif len(values.shape) == 2:
        ft_max = np.nanmax(values, axis=1)
        ft_out = values / ft_max
    
    return ft_out


def nan_array(dim: list):
    """Create 2 or 3d np array with nan's"""
    if len(dim) == 2:
        arr = np.array(
            [[np.nan] * dim[1]] * dim[0]
        )
    else:
        arr = np.array(
            [[[np.nan] * dim[2]] * dim[1]] * dim[0]
        ) 

    return arr


def custom_round_array(
    array, resolution
):
    """
    Round an array on a custom
    resolution of choice.
    Works as well for single values
    
    Input:
        - array: array, list or single
            value to round
        - resolution: resolution to
            round on
    
    Returns:
        - round_array: resulting
            rounded array
    """
    if type(array) == list:
        array = np.array(array)
    
    round_array = np.around(
        array / resolution
    ) * resolution

    return round_array


def spaced_arange(
    start, step, num
):
    arr = np.arange(num) * step + start

    return arr


# ### SMOOTHING FUNCTION WITH NP.CONVOLVE or
# from scipy.ndimage import uniform_filter1d


# sig = accDat['40'].On
# dfsig = np.diff(sig)

# kernel_size = 10
# kernel = np.ones(kernel_size) / kernel_size
# sigSm = np.convolve(sig, kernel, mode='same')
# dfSm = np.convolve(dfsig, kernel, mode='same')

# count = 0
# for i, df in enumerate(dfSm[1:]):
#     if df * dfSm[i] < 0: count += 1

# plt.plot(sigSm)
# plt.plot(dfSm)

# plt.xlim(1000, 1500)


# print(count)