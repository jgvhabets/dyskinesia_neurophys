"""
Plot features per segment over time.
"""

# import public packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import own functions
import lfpecog_features.feats_main as ftsMain
import lfpecog_features.feats_spectral_helpers as specHelpers
import lfpecog_features.featsExtract_perSegment as getFtsSegm
import lfpecog_preproc.preproc_import_scores_annotations as importClin


def run_segmentFts(
    sub_dfs,
    fs,
    subs_incl,
    ephysSources_incl,
    fig_dir: str,
    winLen_sec: int = 180,
    segLen_sec = .5,
    part_segOverlap = 0,
    plot_norm_method = None,
    return_class: bool = False,
    to_save_timeFreqPlot = False,
    to_show_timeFreqPlot = False,
):
    """
    Get features calculated per segment, returned
    in array per ephysGroup and per contact.
    Currently returning PSDs (welch).
    """
    segmFts = {}  # consider segmFts as subclasses

    for sub in subs_incl:

        data_arr, data_keys, dataWinTimes = ftsMain.get_windows(
            sub_dfs[sub],
            fs=fs,  
            ch='none',  # obsolete in current usage
            winLen_sec=winLen_sec
        )
        lid_timings = importClin.get_seconds_of_LID_start()

        segmFts[sub] = {}

        for ephySource in ephysSources_incl:

            print(f'start ft extraction {ephySource} ({sub})')
        
            segmFts[sub][ephySource] = getFtsSegm.segmentFeatures(
                sub=sub,
                data_arr=data_arr,
                fs=fs,
                data_keys=data_keys,
                winTimes=dataWinTimes,
                ephyGroup=ephySource,
                segLen_sec=segLen_sec,
                part_overlap=part_segOverlap,
            )

            if np.logical_or(
                to_save_timeFreqPlot,
                to_show_timeFreqPlot
            ):
                print('plotting...')

                plot_timeFreqSeg_perGroup(
                    segmFtsClass=segmFts[sub][ephySource],
                    fig_dir=fig_dir,
                    LID_timings=lid_timings[sub],
                    to_save=to_save_timeFreqPlot,
                    to_show=to_show_timeFreqPlot,
                    segLen_sec=segLen_sec,
                    winLen_sec=winLen_sec,
                    norm_method=plot_norm_method,
                )
    if return_class:

        return segmFts




def plot_timeFreqSeg_perGroup(
    segmFtsClass,
    fig_dir,
    winLen_sec=180,
    segLen_sec=.5,
    norm_method=True,
    LID_timings=None,
    to_save=False,
    to_show=False,
):
    """
    Plots spectral features per segment over time.

    Input:
        - segmFtsClass: class containing all Feature- and
            segment info
        - fig_dir: figures-path
        - winLen_sec: seconds of window length
        - segLen_sec: seconds of psd-segment length
        - norm_method: how to normalise PSD values,
            - 'relativePsx' for relative powers
                within each PSD
            - default is None, gives raw PSDs, w/o normalisation
            - 'freq_spec_Zscore': calculates z-scores
                per freq-bin, over whole time-course
    """
    fmin = 1
    fmax = 100

    sub = segmFtsClass.sub
    ephysSource = segmFtsClass.ephyGroup

    if ephysSource == 'ECOG':
        nrows = len(segmFtsClass.ephyCols) // 2
        ncols = 2
    
    else:
        nrows = len(segmFtsClass.ephyCols)
        ncols = 1
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(int(8 * ncols), int(nrows * 4))
    )
    axes = axes.flatten()

    for n, col in enumerate(segmFtsClass.ephyCols):

        pxsel, fsel = specHelpers.select_bandwidths(
            getattr(segmFtsClass, col).segmPsds,
            getattr(segmFtsClass, col).psdFreqs,
            fmin, fmax
        )

        if norm_method == 'relativePsx':
            pxsel = specHelpers.relative_power(pxsel)
            max_cbar = .1
        else:
            max_cbar = np.percentile(pxsel, 95)

        # PLOT FEATURE VALUES
        im = axes[n].imshow(
            pxsel.T, cmap='viridis',
            vmin=0, vmax=max_cbar,
            aspect=24,  # change horizontal/vertical proportion
        )
        
        # PLOT WINDOW INDICATORS (gray line where temporal interruption is)
        for xline in getattr(segmFtsClass, col).winStartIndices[1:]:

            axes[n].axvline(
                x=xline, ymin=0, ymax=1, c='gray', lw=3,
            )

        # PLOT DYSKINESIA TIMINGS
        if LID_timings:  # add clinical DYSK-timings
            lid_clrs = {
                'start': 'green',
                'peak': 'orange'
            }
            for timing in lid_clrs:
                lid_i = np.argmin(
                    abs(np.array(
                        getattr(segmFtsClass, col).segmentTimes
                    ) - getattr(LID_timings, f"t_{timing}"))
                )
                axes[n].scatter(
                    lid_i, pxsel.shape[1] * .95,
                    color=lid_clrs[timing],
                    s=300, marker='*',
                    label=f'LID-{timing}',
                )

        # CREATE AND SET XTICKLABELS WITH SEGMENT DOPA-TIMES
        axes[n].set_xlim(0, pxsel.shape[0])
        axes[n].set_xlabel(
            'Time after LT intake (min)',
            fontsize=16,
        )
        axes[n].set_xticks(
            getattr(segmFtsClass, col).winStartIndices      
        )
        axes[n].set_xticklabels(
            np.array(getattr(segmFtsClass, col).winStartTimes) / 60,
        )

        yticklabs = list(np.around(
            np.arange(fmin, fmax, 10), -1
        ))
        yticklabs[0] = fmin
        yticklabs += [fmax]
        yticks = np.linspace(
            0,
            pxsel.shape[1],
            len(yticklabs)
        )
        axes[n].set_yticks(yticks)
        axes[n].set_yticklabels(yticklabs)
        axes[n].set_ylabel(
            f'{col}\n\nFrequency (Hz)',
            fontsize=16,
        )
        axes[n].set_ylim(0, pxsel.shape[1])

    # set title of total plot
    if norm_method == 'relativePsx':
        title = (f'{sub}: Relative power distribution '
                'over time (in rest)')
    else:
        title = f'{sub}: Power spectra over time (in rest)'
    plt.suptitle(
        title,
        x=.5, ha='center', y=.99,
        fontsize=20,
    )

    plt.tight_layout()
    # create colorbar for time-freq plot
    fig.colorbar(im, ax=axes.ravel().tolist())

    if to_save: 
        fname = (
            f'TimeFreqSegm_{sub}_{ephysSource}'
            f'_win{winLen_sec}s_'
            f'seg{int(segLen_sec * 1000)}ms'
        )
        save_folder = 'timeFreq'

        if norm_method == 'relativePsx':
            fname += '_relPSD'
            save_folder += '_relPSD'

        plt.savefig(
            os.path.join(
                fig_dir, 'ft_exploration',
                'rest', save_folder,
                fname
            ), dpi=150, facecolor='w',
        )

    if to_show: plt.show()

    plt.close()

