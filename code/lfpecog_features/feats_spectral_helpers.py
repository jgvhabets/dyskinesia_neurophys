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
    sel = [f_min < f < f_max for f in freqs]

    freqs = freqs[sel]

    if len(values.shape) == 1:

        values = values[sel]
    
    elif len(values.shape) == 2:

        values = values[:, sel]

    return values, freqs