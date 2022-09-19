"""
Helper function for Data Management and
Feature Extraction
"""

# Import public packages and functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, variation
from itertools import compress
from os.path import join

def aggregate_arr_fts(
    method, arr
):
    """
    Aggregate array-features (calculated
    per tap in block) to one value per
    block.
    """
    assert method in [
        'mean', 'median', 'stddev', 'sum',
        'coefVar', 'trend_slope', 'trend_R'
    ], f'Inserted method "{method}" is incorrect'

    if np.isnan(arr).any():

        arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        print('artificial 0 added')  # TODO: fill unknwons with 0 or nan?
        return 0  # was inside if statement

    if method == 'allin1':

        if np.isnan(arr).any():
            arr = arr[~np.isnan(arr)]

        return arr  # all in one big list

    elif method == 'mean':
        
        return np.nanmean(arr)
    
    elif method == 'median':
        
        return np.nanmedian(arr)

    elif method == 'stddev':

        arr = normalize_var_fts(arr)
        
        return np.nanstd(arr)

    elif method == 'sum':
        
        return np.nansum(arr)

    elif method == 'coefVar':

        arr = normalize_var_fts(arr)

        return variation(arr)

    elif method[:5] == 'trend':

        try:
            linreg = linregress(
                np.arange(arr.shape[0]),
                arr
            )
            slope, R = linreg[0], linreg[2]

            if np.isnan(slope):
                slope = 0

            if method == 'trend_slope': return slope
            if method == 'trend_R': return R

        except ValueError:
            
            return 0


def normalize_var_fts(values):

    ft_max = np.nanmax(values)
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



# ### SMOOTHING FUNCTION WITH NP.CONVOLVE

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