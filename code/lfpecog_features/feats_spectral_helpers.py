"""
Contains helper-functions to analyse
spectral features of STN and/or ECoG
data
"""

# Import public packages and functions
import numpy as np

def resample_spectral_freqs(
    psd, freqs, newBinWidth, method: str = 'mean'
):
    """
    Inputs:
        - psd: psd-array
        - freqs: corresponding frequency array
        - newBinWidth (float): new frequency 
            bin width (defines the resolution
            of the psd-freqs)
        - meth: method to merge
    """
    assert method in ['sum', 'mean'], print(
        f'method variable should be "mean" or "sum"'
    )

    oldBinW = np.diff(freqs)[0]  # old bin width
    oldHz = 1 / oldBinW  # old frequency (Hz)
    newHz = 1 / newBinWidth  # new frequency (Hz)

    assert oldHz > newHz, print(
        'New Frequency-Resolution (Hz) cannot'
        ' be larger then current Resolution (Hz)'
    )

    HzDif = oldHz / newHz

    new_freqs = np.arange(
        freqs[0],
        freqs[-1] + newBinWidth,
        newBinWidth
    )

    # transform psd to new resolution
    new_psd = []
    if method == 'sum': meth = np.sum
    elif method == 'mean': meth = np.mean

    for i, f in enumerate(new_freqs):

        old_i0 = int(i * HzDif)
        old_i1 = int((i + 1) * HzDif)
        new_psd.append(
            meth(psd[old_i0:old_i1])
        )

    return new_psd, new_freqs


def select_bandwidths(
    values, freqs, f_min, f_max
):
    """
    Select specific frequencies in PSD
    or coherence outcomes

    Inputs:
        - values: array with values, can be one
            or two-dimensional (containing)
            different windows
        - freqs: corresponding array with frequecies
        - f_min (float): lower cut-off of
            frequencies to select
        - f_max (float): higher cut-off of
            frequencies to select
    
    Returns:
        - values: 1- or 2-d array with spectral values
        - freqs: 1d-array with corresponding frequencies
    """
    sel = [f_min <= f <= f_max for f in freqs]

    if len(values.shape) == 1:

        values = values[sel]
    
    elif len(values.shape) == 2:

        if values.shape[1] != len(freqs):

            values = values.T

        values = values[:, sel]
    
    freqs = freqs[sel]

    return values, freqs


def relative_power(psd):
    """
    Convert original power spectral
    density values in relative power.
    Meaning that every PSD-freq shows
    the part of the total PSD it
    represents in this window
    
    Input:
        - psd (array): original psd, can
            be uni-dimensional and two-
            dimonesional
    
    Return:
        -relPsd (array): converted values
            between 0 and 1
    """
    if len(psd.shape) == 1:
        # if psd is one-dimensional
        sumPsd = np.nansum(psd)
        relPsd = psd / sumPsd

    elif len(psd.shape) == 2:
        # if psd is two-dimensional
        sumsVector = np.nansum(psd, axis=1)  # sum for every single-row (psd-window)
        relPsd = np.divide(psd.T, sumsVector).T  # vector-wise division per row


    return relPsd


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

    TODO: discuss why timeseries is still identical, only
    larger order of magnitude?
    
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
    if len(array) == 1: array = np.array([array])  # transform to 2d array 1 x samples
    if array.shape[0] > array.shape[1]: array = array.T  # transpose in horizontal format
    
    # create mne info object
    ch_names=[f'n{n}' for n in range(array.shape[0])]
    mne_info = create_info(
        ch_names=ch_names,
        sfreq=fs,
        ch_types=['eeg'] * len(ch_names)
    )
    
    freqs_noise = (freqband_to_optim[0] - 1, freqband_to_optim[1] + 1)

    ssd = SSD(info=mne_info,
        reg='oas',
        sort_by_spectral_ratio=False,  # False for purpose of example.
        filt_params_signal=dict(l_freq=freqband_to_optim[0],h_freq=freqband_to_optim[1],
                                l_trans_bandwidth=1, h_trans_bandwidth=1),
        filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                               l_trans_bandwidth=1, h_trans_bandwidth=1),
    )
    # perform SSD
    ssd.fit(array)
    ssd_array = ssd.transform(array)

    return ssd_array
