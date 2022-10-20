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
import lfpecog_preproc.preproc_import_scores_annotations as importClin


def get_FreqBandArray_fromSpecFts(
    segmFtArray, ftFreqs,
    to_Zscore=False,
    to_Smooth=False,
    smoothWin_sec=10,

):
    """
    Transforms an array of spectral features
    over time into an array containing mean-
    features per pre-defined freq-bandwidth.

    Input:
        - segmFtArray: 2d array of spectral features
            over time, shape: freqs, time
        - ftFreqs: array with corr freqs to features
        - to_Zscore: boolean to apply z-scoring
            per freq-bandwidth
        - to_Smooth: boolean to apply smoothing
            within freq-bandwidth
        - smoothWin_sec: if smoothing is applied,
            this defines the window in seconds
    
    Returns:
        - bp_array: new array with mean-features per
            bandwidth, shape: bandwidths x time
        - freq_bandwidth.keys: names of bandwidths
    """
    # define bandwidths: names are used for output,
    # frequencies for feature-selection
    freq_bandwidths = {
        'theta-alpha': (4, 12),
        'low-beta': (12, 20),
        'high-beta': (20, 30),
        'broadband gamma': (50, 200),
        'narrow-band gamma': (60, 90)
    }
    # define resulting variables
    segLen_sec = .5
    smooth_nSegs = int(smoothWin_sec / segLen_sec)
    # create nan-array to fill
    bp_array = ftHelpers.nan_array(
        [len(freq_bandwidths), segmFtArray.shape[0]]
    )

    for ax_n, f_band in enumerate(
        freq_bandwidths.keys()
    ):

        flow, fhigh = freq_bandwidths[f_band]

        bandpowers, _ = specHelpers.select_bandwidths(
            segmFtArray,
            freqs=ftFreqs,
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
    ft_type: str,
    to_Zscore=True,
    to_Smooth=False,
    smoothWin_sec=10,
    fsize=14,
    to_show=True,
    to_save=False,
    fig_name=None,
    fig_dir=None,
    sub=None, # TODO: PUT SUB IN CLASS
):
    """
    Takes one or multiple array with spectral
    features over time and plots colormaps

    Input:
        - inputChannels: containing spectral features
            of channels to plot, list of classes or
            one class; one channel(combi) per class
        - ft_type: describes features inserted, should
            be in ['Spectral Power', Imag-Coherence']
        
    """
    # adjust for input as 1 Class or list of Classes
    try:  # if list is given
        nrows = len(inputChannels)
    except TypeError:  # if class is given
        nrows = 1
    
    list_of_ftTypes = [
        'Spectral Power', 'Imag-Coherence'
    ]
    assert ft_type in list_of_ftTypes, print(
        f'ft_type ({ft_type}) not in list'
    )

    # set up figure
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(16, 4 * nrows),
        sharex=True
    )

    for i in np.arange(nrows):
        # set variables (1 contact-Class or list of contact-Classes)
        if nrows == 1:
            ax = axes
            try:
                ch_fts = inputChannels[0]
            except TypeError:
                ch_fts = inputChannels

        else:
            ax = axes[i]
            ch_fts = inputChannels[i]

        # create actual freq bandpower arrays
        bp_array, freqBandNames = get_FreqBandArray_fromSpecFts(
            segmFtArray=ch_fts.segmPsds,
            ftFreqs=ch_fts.psdFreqs,
            to_Zscore=to_Zscore,
            to_Smooth=to_Smooth,
            smoothWin_sec=smoothWin_sec,
        )

        # set correct figure-parameters for ft-type and settings
        if ft_type == 'Spectral Power':
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
        elif ft_type == 'Imag-Coherence':
            map_params = {
                    'cmap': 'PiYG',  # coolwarm
                    'vmin': -1, 'vmax':1
                }


        # PLOT freqband-array
        im = ax.pcolormesh(
            bp_array,
            **map_params,
        )
        cb = fig.colorbar(im, ax=ax)
        for tick in cb.ax.get_yticklabels():
            tick.set_fontsize(fsize)

        # PLOT LID-timings (observed Start and Peak)
        lid_timings = importClin.get_seconds_of_LID_start()[sub]
        lid_clrs = {
            'start': 'green',
            'peak': 'orange'
        }
        for timing in lid_clrs:
            lid_i = np.argmin(abs(
                ch_fts.segmDopaTimes -
                getattr(lid_timings, f"t_{timing}")
            ))
            axes[i].scatter(
                lid_i, len(freqBandNames) - .2,
                color=lid_clrs[timing],
                s=500, marker='*',
                label=f'LID-{timing}',
            )

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

        title = 'Freq-Band' + ft_type
        if to_Zscore: title = 'Z-scored ' + title
        if nrows > 1: title = f'{ch_fts.contactName}: ' + title
        ax.set_title(title, size=fsize + 4)

        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)

    plt.suptitle(
        f'sub-{sub}', color='gray',
        x=.05, y=.9, size=fsize + 12)

    if to_save:
        nameCode = {
            'Spectral Power': 'Powers',
            'Imag-Coherences': 'ICOH'
        }
        plt.savefig(
            os.path.join(
                fig_dir, 'ft_exploration', 'rest',
                f'freqBand{nameCode[ft_type]}',
                fig_name + f'_smooth{smoothWin_sec}'
            ), dpi=150, facecolor='w',
        )
    
    if to_show: plt.show()



    plt.close()
