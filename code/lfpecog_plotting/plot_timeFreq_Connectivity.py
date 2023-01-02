"""Plot time-frequency Plots for connectivity measures"""

# import
from os.path import join, exists
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

import lfpecog_preproc.preproc_import_scores_annotations as importClin
import lfpecog_features.feats_spectral_helpers as specHelpers
from utils.utils_fileManagement import get_project_path

def plot_mvc(
    sub, plot_data, plot_freqs, plot_times,
    fs=16, cmap='viridis', mvc_method='mic',
    to_save=False, save_path=None, fname=None,
):
    ### TODO: INSERT 3 EMPTY ROWS ON TIME_JUMP MOMENTS TO INSERT GRAY BAR

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # plot colormap
    im = ax.imshow(
        plot_data.T,
        cmap=cmap, vmin=0, vmax=.6,
        aspect=.5
    )

    # plot colorbar
    fig.colorbar(im, ax=ax)#axes.ravel().tolist())

    # PLOT WINDOW INDICATORS (gray line where temporal interruption is)
    for i_pre, x_t in enumerate(plot_times[1:]):
        # if epochs are more than 5 minutes separated
        if x_t - plot_times[i_pre] > 300:

            ax.axvline(
                i_pre + 1,
                ymin=0, ymax=50, color='lightgray', lw=3, alpha=.8,
            )

    # set correct frequencies on Y-axis
    ytickhop = 8
    ax.set_ylim(0, plot_data.shape[1])
    ax.set_yticks(range(plot_data.shape[1])[::ytickhop])
    ax.set_yticklabels(plot_freqs[::ytickhop])
    ax.set_ylabel('Frequency (Hz)', size=fs + 2)
    # set correct times on X-axis
    xtickhop = 6  #int(len(plot_times) / 9)
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    ax.set_xticks(np.linspace(0, plot_data.shape[0] - 1, len(xticklabs)))
    ax.set_xticklabels(np.around(xticklabs / 60, 1))
    ax.set_xlabel('Time after LDopa (minutes)', size=fs + 2)

    if mvc_method.lower() == 'mic':
        ax.set_title(
            f'sub-{sub}  -  abs. imaginary-Coherence (multivariate)',
            size=fs + 6)
    elif mvc_method.lower() == 'mim':
        ax.set_title(
            f'sub-{sub}  -  abs. Multivariate Interaction Measure',
            size=fs + 6)

    plt.tick_params(axis='both', labelsize=fs, size=fs,)
    plt.tight_layout()
    
    if to_save:
        if not exists(save_path): makedirs(save_path)

        plt.savefig(join(save_path, f'{fname}.png'),
                    dpi=150, facecolor='w',)
        print(f'...Figure {fname} saved to {save_path}')

    plt.close()


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

    fmin = 5
    fmax = 98

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
        coh_sel, f_sel = specHelpers.select_bandwidths(
            values=getattr(ch_fts, ft_params[ft_type]['ft_attr']),
            freqs=getattr(ch_fts, ft_params[ft_type]['freq_attr']),
            f_min=fmin,
            f_max=fmax
        )
        # print(coh_sel.shape, coh_sel,)
        # print(f_sel.shape, f_sel)
        # print(ch_fts.epoch_times.shape, ch_fts.epoch_times)
        # break  

        # set correct figure-parameters for ft-type and settings
        if  ft_type == 'Imag-Coherence':
            map_params = {
                'cmap': 'PiYG',  # coolwarm
                'vmin': -1, 'vmax': .3
            }
        elif ft_type == 'abs-Imag-Coherence':
            map_params = {
                'cmap': 'BuPu',
                'vmin': 0, 'vmax': .3
            }
        elif ft_type == 'Squared-Coherence':
            map_params = {
                'cmap': 'BuPu',
                'vmin': 0, 'vmax': .3
            }

        # if coh_sel.shape[0] <= 10: aspect_ratio = .05
        # else: aspect_ratio = .08
        aspect_ratio = .25 / (coh_sel.shape[1] / coh_sel.shape[0])
        if nrows > 6: aspect_ratio / 2
        # PLOT FEATURE VALUES
        im = ax.imshow(
            coh_sel.T, cmap='viridis',
            vmin=map_params['vmin'],
            vmax=map_params['vmax'],
            aspect=aspect_ratio,  # change horizontal/vertical proportion
        )

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
                lid_i, coh_sel.shape[0] - .2,
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
                    epoch_clin_scores, ls='dotted',
                    color='orange', alpha=.6, lw=3,)
                clinAx.set_ylim(0, 20)
                clinAx.set_ylabel('CDRS total score')

            except FileNotFoundError:
                print(f'No clin scores found for sub {sub}')
            # except ValueError:
            #     print(f'Incorrect clin scores found for sub {sub}')

        # # set Freq-Band names as y-ticks
        yticklabs = list(np.arange(fmin, fmax, 10))
        yticklabs[0] = fmin
        yticklabs.append(fmax)
        yticks = np.linspace(0, coh_sel.shape[1], len(yticklabs))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabs)
        ax.set_ylabel(f'Frequency (Hz)', fontsize=16, )
        ax.set_ylim(0, coh_sel.shape[1])

        # set dopa-times correpsonding to segments as x-ticks
        ax.set_xlabel(
            'Time after LT intake (min)',
            fontsize=fsize,
        )
        ax.set_xticks(
            np.arange(
                0, ch_fts.epoch_times.shape[0], 2
            )
        )
        ax.set_xticklabels(
            np.around(ch_fts.epoch_times[::2] / 60, 1),
            fontsize=fsize,
        )
        # PLOT WINDOW INDICATORS (gray line where temporal interruption is)
        for i_pre, x_t in enumerate(ch_fts.epoch_times[1:]):
            # if epochs are more than 5 minutes separated
            if x_t - ch_fts.epoch_times[i_pre] > 300:

                ax.axvline(
                    i_pre + 1,
                    ymin=0, ymax=50, color='lightgray', lw=3, alpha=.8,
                )

        title = f'Freq-Band {ft_type}'
        if to_Zscore: title = 'Z-scored ' + title
        if nrows > 1: title = f'{ch_fts.channelName}: ' + title
        ax.set_title(title, size=fsize + 4)

        for side in ['top','right','bottom','left']:
            ax.spines[side].set_visible(False)

    plt.suptitle(
        f'sub-{sub}', color='gray',
        x=.05, y=.97, size=fsize + 12)
    plt.tight_layout()
    # create colorbar for time-freq plot
    fig.colorbar(im, ax=axes.ravel().tolist())

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
            join(
                fig_dir, 'ft_exploration', 'rest',
                f'timeFreq_{nameCode[ft_type]}',
                fig_name
            ), dpi=150, facecolor='w',
        )
    
    if to_show: plt.show()



    plt.close()

