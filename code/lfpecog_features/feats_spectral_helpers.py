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