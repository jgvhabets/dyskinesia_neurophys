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


def correct_notch_throughs(
    freqs, psd, notches,
):
    """
    smoothen out linenoise removal notch
    throughs, applied before fooof fitting
    """
    for notch in notches:
        # find closest index to 6 before and after notch-center
        i = np.argmin(abs(np.array(freqs) - (notch - 6)))
        # use as reference values
        before = psd[i]
        i = np.argmin(abs(np.array(freqs) - (notch + 6)))
        after = psd[i]
        # select values to correct (+/- 5 Hz)
        sel = np.where(np.logical_and(
            (notch - 5) < np.array(freqs),
            np.array(freqs) < (notch + 5)))[0]
        # linear line between references to fill
        fill = np.linspace(before, after, len(sel))
        # fill with mean between original value and linear line
        for i, v in zip(sel, fill): psd[i] = (psd[i] + v) / 2
    
    return psd


def get_empty_spectral_ft_dict(
    lists_or_means: str,
    sources_incl = ['LFP_L', 'LFP_R', 'ECOG'],
    bw_feats = ['peak_freq', 'peak_size'],
    ap_feats = ['ap_off', 'ap_exp'],
    ranges = ['alpha', 'lo_beta', 'hi_beta', 'midgamma'],
):
    assert lists_or_means in ['lists', 'means'], (
        'lists_or_means must be "lists" or "means"'
    )

    feat_outs = {}

    for src in sources_incl:
        feat_outs[src] = {}

        for ft in bw_feats:
            feat_outs[src][ft] = {}

            if lists_or_means == 'lists':
                for bw in ranges: feat_outs[src][ft][bw] = []
            
            elif lists_or_means == 'means':
                for bw in ranges:
                    feat_outs[src][ft][bw] = {}
                    feat_outs[src][ft][bw]['mean'] = np.nan
                    feat_outs[src][ft][bw]['cv'] = np.nan
                    feat_outs[src][ft][bw]['sd'] = np.nan
        
        for ft in ap_feats:
            feat_outs[src][ft] = {}

            if lists_or_means == 'lists':
                feat_outs[src][ft] = []
            
            elif lists_or_means == 'means':
                feat_outs[src][ft]['mean'] = np.nan
                feat_outs[src][ft]['cv'] = np.nan
                feat_outs[src][ft]['sd'] = np.nan
            

    return feat_outs