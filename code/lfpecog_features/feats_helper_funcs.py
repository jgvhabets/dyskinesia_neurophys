"""
Helper function for Data Management and
Feature Extraction
"""

# Import public packages and functions
import numpy as np
from scipy.stats import variation
# from scipy.ndimage import uniform_filter1d

from utils.utils_fileManagement import get_avail_ssd_subs

def get_indiv_peak_freqs(
    psd_dict,
    STATE: str = 'all',
    BANDS: dict = {'narrow_gamma': [62, 89],
                   'lo_beta': [13, 20],
                   'hi_beta': [21, 24]},
    SOURCE_SEL: str = 'lfp',
    verbose: bool = False,
    DATA_VERSION='v4.0',
    FT_VERSION='v6',
):
    
    assert STATE in ['rest', 'dyskmove', 'tap', 'all'], (
        f'incorrect STATE ({STATE})'
    )
    if STATE == 'all': sel_states = ['rest', 'dyskmove', 'tap']
    else: sel_states = [STATE,]

    lid_states = ['nolidbelow30', 'nolidover30', 'nolid',
                  'mildlid', 'moderatelid', 'severelid']

    SUBS = get_avail_ssd_subs(DATA_VERSION=DATA_VERSION,
                              FT_VERSION=FT_VERSION,)
    sub_psds = {s: [] for s in SUBS}

    # COLLECT ALL PSD 1-sec ROWS
    for state in psd_dict.keys():
        # only continue with selected condition data
        if not any([s in state.lower() for s in sel_states]):
            if verbose: print(f'...SKIP state (SEL): {state}')
            continue

        if not any([l in state for l in lid_states]):
            if verbose: print(f'...SKIP state (lid state): {state}')        
            continue
        
        if verbose: print(f'continue with {state}')
        freqs = psd_dict[state].freqs

        # only continue with subject data
        for k in vars(psd_dict[state]).keys():

            s = k.split(state)[0]
            if not s.endswith('_'): continue
            src = s[:-5]
            sub = s[-4:-1]

            if not SOURCE_SEL in src: continue
            
            # print(f'sub-{sub}, data from {src}   ({k})')
            
            # get psd array (samples, freqs)
            psx = getattr(psd_dict[state], k)
            

            # add sub-id code for every added sample
            for l in psx: sub_psds[sub].append(l)

    # CALCULATE INDIV VARIANCES for freq bands
    ind_peaks = {s: {} for s in SUBS}

    for sub in SUBS:
        sub_psx = np.array(sub_psds[sub])
        if verbose: print(f'\n...SUB-{sub} (n = {len(sub_psx)} 1-sec-samples)')

        for bw, f_range in BANDS.items():
            # select relevant freq range from psd
            f_sel = np.logical_and(freqs >= f_range[0],
                                   freqs < f_range[1])
            # get power mean over freq-range, per 1-sec epoch
            try:
                powers = sub_psx[:, f_sel]
            except IndexError:
                if len(sub_psx) == 0: continue  # skips empty psx
                else: powers = sub_psx[:, f_sel]  # raises IndexError in case of different cause
            
            pow_fs = freqs[f_sel]
            variations = variation(powers, axis=0)
            i_max_var = np.argmax(variations)
            f_max_var = pow_fs[i_max_var]
            if verbose: print(f'\tmax {bw} variation in {f_max_var}')
            ind_peaks[sub][bw] = f_max_var

    return ind_peaks



def baseline_zscore(
    arr_to_zscore, bl_mean, bl_std
):
    """
    Performs a z-score with previously determined
    mean and std-dev (from a baseline)
    """
    assert type(arr_to_zscore) == np.ndarray, (
        'arr_to_zscore has to be array dtype'
    ) 

    new_arr = (arr_to_zscore - bl_mean) / bl_std

    return new_arr



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


from scipy.ndimage import uniform_filter1d

def smoothing(
    sig, win_samples=None, win_ms=None, fs=None,
):
    """
    smoothens a signal, either on window length in
    millisec or in samples.
    NEEDS EITHER: win_samples, OR: win_ms AND fs

    Inputs:
        - sig: 1d array
        - win_samples: n samples to use for smoothing
        - win_ms: millisecs to smooth
        - fs: fs (only needed when win_ms given)
    
    Returns:
        - sig: smoothened signal
    """
    assert win_samples or win_ms, (
        'define smoothing window samples or ms'
    )
    if win_ms:
        assert fs, 'define fs if windowing on millisec'
        win_samples = int(fs / 1000 * win_ms)  # smoothing-samples in defined ms-window
    
    # smooth signal
    sig = uniform_filter1d(sig, win_samples)

    return sig
    

def check_matrix_properties(M, verbose=True):
    """
    Input:
        - M: np.ndarray, to check
    """
    # Check if the matrix is singular
    rank = np.linalg.matrix_rank(M)
    if verbose: print(f'shape {M.shape}, rank: {rank}')
    
    # Compute the singular values
    singular_values = np.linalg.svd(M, compute_uv=False)
    
    # Check the magnitude of the smallest singular value
    smallest_singular_value = singular_values[-1]
    tolerance = 1e-6  # Define a tolerance for singularity
    # calculate condition number
    condition_number = np.max(singular_values) / np.min(singular_values)
    if verbose: print(f'matrix-condition number is {condition_number}')
    
    if smallest_singular_value < tolerance:
        if verbose:
            print('The matrix is nearly singular (smallest '
                  f'sing-value: {smallest_singular_value}).')


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def regularize_matrix(M, lasso_alpha=1e-3,):
    """
    """

    # Perform column-wise Lasso regularization
    lasso = Lasso(alpha=lasso_alpha)  # Specify the regularization strength (alpha)
    scaler = StandardScaler()

    M_regularized = np.zeros_like(M)

    for i in range(M.shape[1]):
        column = M[:, i]
        column_scaled = scaler.fit_transform(column.reshape(-1, 1))  # Reshape to a column vector and scale
        lasso.fit(column_scaled, column_scaled)  # Perform Lasso regularization
        column_regularized = lasso.coef_.flatten()  # Retrieve the regularized column
        M_regularized[:, i] = column_regularized
    
    return M_regularized