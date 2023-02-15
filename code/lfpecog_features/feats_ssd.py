"""
Perform spectral spatial decomposition
"""

# Import public packages and functions
import numpy as np

from mne import create_info
from mne.decoding import SSD

def SSD_on_array(
    array, fs, freqband_to_optim
):
    """
    Spatial Signal Decompositino increases
    the signal to noise ratio within a specific
    freq-bandwidth by correcting the signal
    for the 'noise-floor' measured in
    neighbouring frequency bins

    Based on mne.Decoding.SDD:
    (https://mne.tools/stable/generated/mne.decoding.SSD.html#mne.decoding.SSD)

    TODO: eigenvector failure (positive definitive error)
    when performed on 2d data array
    
    Input:
        - array to transform: should be without nans
        - fs
        - freqband_to_optim: tuple or list
            with lower and higher freq border
    
    Returns:
        - ssd_array: array with optimised SNR
            for defined freq-band
    
    Raises:
        - Assertion if array contains NaNs
    """
    assert ~np.isnan(array).any(), (
        'array to perform SSD on cannot contain NaNs'
    )
    if len(array.shape) == 1: array = np.array([array])  # transform to 2d array 1 x samples
    if array.shape[0] > array.shape[1]: array = array.T  # transpose in horizontal format
    
    # create mne info object
    ch_names=[f'n{n}' for n in range(array.shape[0])]
    mne_info = create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types=['eeg'] * len(ch_names)
    )
    
    freqs_noise = (freqband_to_optim[0] - 4, freqband_to_optim[1] + 4)
    # freqs_noise = (1, 50)

    ssd_obj = SSD(
        info=mne_info,
        reg='oas',
        sort_by_spectral_ratio=False,  # False for purpose of example.
        filt_params_signal=dict(l_freq=freqband_to_optim[0],h_freq=freqband_to_optim[1],
                                l_trans_bandwidth=1, h_trans_bandwidth=1),
        filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                               l_trans_bandwidth=1, h_trans_bandwidth=1),
        n_components=len(array),
        n_fft=len(array),
    )
    # perform SSD
    ssd_obj.fit(array)

    ssd_sources = ssd_obj.transform(X=array)

    return ssd_obj, ssd_sources


