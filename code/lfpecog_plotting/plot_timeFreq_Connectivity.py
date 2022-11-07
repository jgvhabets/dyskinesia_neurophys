"""Plot time-frequency Plots for connectivity measures"""

# import
import numpy as np
import matplotlib.pyplot as plt

import lfpecog_preproc.preproc_import_scores_annotations as importClin


def plot_timeFreq_Coherence(
    inputChannels,
    ft_type: str,
    to_Zscore=False,
    to_Smooth=False,
    smoothWin_sec=10,
    fsize=14,
    to_show=True,
    to_save=False,
    fig_name=None,
    fig_dir=None,
    sub=None, # TODO: PUT SUB IN CLASS
    to_include_clinScores=True,
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
        'Imag-Coherence',
        'abs-Imag-Coherence',
        'Squared-Coherence',
    ]
    assert ft_type in list_of_ftTypes, print(
        f'ft_type ({ft_type}) not in list'
    )

    ft_params = {
        'Squared-Coherence': {'ft_attr': 'sqCOH',
                           'freq_attr': 'freqs'},
        'Imag-Coherence': {'ft_attr': 'ICOH',
                           'freq_attr': 'freqs'},
        'abs-Imag-Coherence': {'ft_attr': 'absICOH',
                           'freq_attr': 'freqs'}
    }

    fmin = 1
    fmax = 100

    # set up figure
    if nrows > 6: fig_width = 32
    else: fig_width = 16
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(fig_width, 4 * nrows),
        sharex=False,
    )
    
    # LOOP OVER INPUTCHANNELS
    for i in np.arange(nrows):
        
        if nrows == 1:
            ax = axes
            try:
                ch_fts = inputChannels[0]
            except TypeError:
                ch_fts = inputChannels

        else:
            ax = axes[i]
            ch_fts = inputChannels[i]

        # create actual time-coherence arrays
        pxsel, fsel = specHelpers.select_bandwidths(
            getattr(segmFtsClass, col).segmPsds,
            getattr(segmFtsClass, col).psdFreqs,
            fmin, fmax
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
        elif  ft_type == 'Imag-Coherence':
            map_params = {
                'cmap': 'PiYG',  # coolwarm
                'vmin': -1, 'vmax': 1
            }
        elif ft_type == 'abs-Imag-Coherence':
            map_params = {
                'cmap': 'BuPu',
                'vmin': 0, 'vmax': .5
            }
        elif ft_type == 'Squared-Coherence':
            map_params = {
            'cmap': 'BuPu',
                'vmin': 0, 'vmax': .3
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
            try:
                lid_i = np.argmin(abs(
                    ch_fts.segmDopaTimes -
                    getattr(lid_timings, f"t_{timing}")
                ))
            except AttributeError:
                lid_i = np.argmin(abs(
                    ch_fts.epoch_times -
                    getattr(lid_timings, f"t_{timing}")
                ))

                if ft_type[:2]=='Sp':
                    print('REMOVE segmDopaTimes!! line 214')

            ax.scatter(
                lid_i, len(freqBandNames) - .2,
                color=lid_clrs[timing],
                s=500, marker='*',
                label=f'LID-{timing}',
            )
        
        # PLOT CDRS SCORES
        if to_include_clinScores:
            try:
                scores, _, _ = importClin.run_import_clinInfo(sub=sub)
                # get closest CDRS score to epoch_time
                if type(scores) == type(None):
                    raise ValueError('None scores')
                epoch_clin_scores = [scores.iloc[
                    np.argmin(abs(m - scores['dopa_time']))
                ]['CDRS_total'] for m  in (ch_fts.epoch_times / 60)]

                clinAx = ax.twinx()
                clinAx.plot(
                    epoch_clin_scores,
                    color='darkblue', alpha=.2, lw=5,)
                clinAx.set_ylim(0, 20)
                clinAx.set_ylabel('CDRS total score')

            except FileNotFoundError:
                print(f'No clin scores found for sub {sub}')
            # except ValueError:
            #     print(f'Incorrect clin scores found for sub {sub}')

        # set Freq-Band names as y-ticks
        ax.set_yticks(np.arange(.5, bp_array.shape[0], 1))
        ax.set_yticklabels(freqBandNames, size=fsize,)

        # set dopa-times correpsonding to segments as x-ticks
        ax.set_xlabel(
            'Time after LT intake (min)',
            fontsize=fsize,
        )
        ax.set_xticks(
            np.arange(
                0, ch_fts.epoch_times.shape[0], 3
            )
        )
        ax.set_xticklabels(
            np.around(ch_fts.epoch_times[::3] / 60, 1),
            fontsize=fsize,
        )
        # PLOT WINDOW INDICATORS (gray line where temporal interruption is)
        for i_pre, x_t in enumerate(ch_fts.epoch_times[1:]):
            # if epochs are more than 5 minutes separated
            if x_t - ch_fts.epoch_times[i_pre] > 300:

                ax.axvline(
                    i_pre + 1,
                    ymin=0, ymax=5, color='k', lw=3, alpha=.8,
                )

        title = 'Freq-Band ' + ft_type
        if to_Zscore: title = 'Z-scored ' + title
        if nrows > 1: title = f'{ch_fts.channelName}: ' + title
        ax.set_title(title, size=fsize + 4)

        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)

    plt.suptitle(
        f'sub-{sub}', color='gray',
        x=.05, y=.97, size=fsize + 12)
    plt.tight_layout()

    if to_save:
        nameCode = {
            'Spectral Power': 'Powers',
            'Squared-Coherence': 'sqCOH',
            'Imag-Coherence': 'ICOH',
            'abs-Imag-Coherence': 'absICOH'
        }
        if to_Smooth: fig_name += f'_smooth{smoothWin_sec}'
        else: fig_name += f'_noSmooth'

        if to_Zscore: fig_name += f'_zScored'

        plt.savefig(
            os.path.join(
                fig_dir, 'ft_exploration', 'rest',
                f'freqBand_{nameCode[ft_type]}',
                fig_name
            ), dpi=150, facecolor='w',
        )
    
    if to_show: plt.show()



    plt.close()

