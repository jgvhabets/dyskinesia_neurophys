"""
Create and/or Plot Bandpowers per Frequency-
Bandwidth.
"""

# import public packages
import os
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

# import own functions
import lfpecog_features.feats_spectral_helpers as specHelpers
import lfpecog_features.feats_helper_funcs as ftHelpers


def get_FreqBandPowers_array(
    segmPsdArray, psdFreqs,
    to_Zscore=False,
    to_Smooth=False,
    smoothWin_sec=10,

):
    """
    Makes an array with mean-PSDs over time
    per freq-band, out of an array containing
    full PSDs over time
    """
    freq_bandwidths = {
        'theta-alpha': (4, 12),
        'low-beta': (13, 20),
        'high-beta': (20, 30),
        'broadband gamma': (50, 200),
        'narrow-band gamma': (60, 90)
    }
    # define resulting variables
    segLen_sec = .5
    smooth_nSegs = int(smoothWin_sec / segLen_sec)
    # create nan-array to fill
    bp_array = ftHelpers.nan_array(
        [len(freq_bandwidths), segmPsdArray.shape[0]]
    )

    for ax_n, f_band in enumerate(
        freq_bandwidths.keys()
    ):

        flow, fhigh = freq_bandwidths[f_band]

        bandpowers, _ = specHelpers.select_bandwidths(
            segmPsdArray,
            freqs=psdFreqs,
            f_min=flow, f_max=fhigh
        )
        bp_sig = np.nanmean(bandpowers, axis=1)

        if to_Zscore:
            bp_Z_mean = np.nanmean(bp_sig)
            bp_Z_sd = np.nanstd(bp_sig)
            bp_sig = (bp_sig - bp_Z_mean) / bp_Z_sd

        if to_Smooth:
            bp_sig = uniform_filter1d(bp_sig, smooth_nSegs)

        bp_array[ax_n, :] = bp_sig
    
    return bp_array, freq_bandwidths.keys()


def plot_bandPower_colormap(
    inputChannels, 
    to_Zscore=True,
    to_Smooth=False,
    smoothWin_sec=10,
    fsize=14,
    to_show=True,
    to_save=False,
):
    try:  # if list is given
        nrows = len(inputChannels)
    except TypeError:  # if class is given
        nrows = 1

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(16, 4 * nrows),
        sharex=True
    )

    for i in np.arange(nrows):
        # set variables, makes it feasible for single contact-fts and list of contact-fts
        if nrows == 1:
            ax = axes
            try:
                ch_fts = inputChannels[0]
            except TypeError:
                ch_fts = inputChannels

        else:
            ax = axes[i]
            ch_fts = inputChannels[i]


        bp_array, freqBandNames = get_FreqBandPowers_array(
            segmPsdArray=ch_fts.segmPsds,
            psdFreqs=ch_fts.psdFreqs,
            to_Zscore=to_Zscore,
            to_Smooth=to_Smooth,
            smoothWin_sec=smoothWin_sec,
        )

        if to_Zscore:
            map_params = {
                'cmap': 'coolwarm',  # blue-white-red: bwr, yellow-orange-red: YlOrRd
                'vmin': -2.5, 'vmax':2.5
            }
        else:
            map_params = {
                'cmap': 'viridis',
                'vmin': 0, 'vmax':1e-12
            }

        im = ax.pcolormesh(
            bp_array,
            **map_params,
        )
        cb = fig.colorbar(im, ax=ax)
        for tick in cb.ax.get_yticklabels():
            tick.set_fontsize(fsize)

        # set Freq-Band names as y-ticks
        ax.set_yticks(np.arange(.5, bp_array.shape[0], 1))
        ax.set_yticklabels(freqBandNames, size=fsize,)

        # set dopa-times correpsonding to segments as x-ticks
        ax.set_xlabel(
            'Time after LT intake (min)',
            fontsize=fsize,
        )
        ax.set_xticks(ch_fts.winStartIndices,)
        ax.set_xticklabels(
            np.array(ch_fts.winStartTimes) / 60,
            fontsize=fsize,
        )
        # PLOT WINDOW INDICATORS (gray line where temporal interruption is)
        for xline in ch_fts.winStartIndices[1:]:
            ax.axvline(
                xline,
                ymin=0, ymax=5, color='k', lw=3, alpha=.8,
            )

        title = 'Bandpowers'
        if to_Zscore: title = 'Z-scored ' + title
        if nrows > 1: title = f'{ch_fts.contactName}: ' + title
        ax.set_title(title, size=fsize + 4)

        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)

    if to_show: plt.show()
    if to_save: print('TODO: make saving func')

    plt.close()
