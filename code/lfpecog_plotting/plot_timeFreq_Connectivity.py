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
    add_CDRS=True, add_ACC=False,
):
    # define general plot settings
    num_axes = 1
    if add_CDRS: num_axes += 1
    if add_ACC: num_axes += 1
    fig_height = 8 + (4 * (num_axes - 1))

    fig, axes = plt.subplots(num_axes, 1, figsize=(16, fig_height))
    
    # Plot MVC features in subplot
    fig, axes = subplot_mvc(fig=fig, axes=axes, fs=fs, cmap=cmap,
                            num_axes=num_axes, plot_values=plot_data,
                            plot_freqs=plot_freqs,
                            plot_times=plot_times, mvc_method=mvc_method,)

    # Plot clinical LID scores (CDRS) in subplot
    if add_CDRS:
        fig, axes = subplot_cdrs(fig=fig, axes=axes, fs=fs,
                                 sub=sub, plot_times=plot_times)

    # Plot Accelerometer features in subplot
    if add_ACC:
        fig, axes = subplot_acc(fig=fig, axes=axes,)
    

    plt.tick_params(axis='both', labelsize=fs, size=fs,)
    plt.tight_layout()
    
    if to_save:
        if not exists(save_path): makedirs(save_path)

        plt.savefig(join(save_path, f'{fname}.png'),
                    dpi=150, facecolor='w',)
        print(f'...Figure {fname} saved to {save_path}')

    plt.close()


def subplot_mvc(
    fig, axes, num_axes, fs, cmap, mvc_method,
    plot_values, plot_freqs, plot_times,
):
    """
    Create subplot with Multivariate Connectivity
    features

    Input:
        - fig, axes: original fig, axes
        - cmap: colormap to use
        - fs: fontsize
        - plot_values: feature values to plot,
            as 2d-array [n-times, n-freqs]
        - plot_freqs, plot_times: corr to values
            
    """
    # define which ax to plot mvc
    if num_axes == 1: mvc_ax = axes
    else: mvc_ax = axes[1]

    # plot colormap
    vmax = .5
    im = mvc_ax.imshow(plot_values.T, vmin=0, vmax=vmax,
                       cmap=cmap, aspect=.5)
    # plot colorbar
    cbar = fig.colorbar(im, ax=mvc_ax, pad=.01)#axes.ravel().tolist())
    cbar.ax.set_yticks(np.arange(0, vmax, .1), size=fs)
    cbar.ax.set_yticklabels(np.around(np.arange(0, vmax, .1), 1), size=fs)
    if mvc_method.lower() == 'mic':
        cv_title = 'abs. Max. Imaginary Coherence'
    elif mvc_method.lower() == 'mim':
        cv_title = 'Multivariate Interaction Measure'
    cbar.ax.set_ylabel(cv_title, rotation=270, size=fs + 2)
    cbar.ax.get_yaxis().labelpad = 20

    # PLOT JUMP IN TIME INDICATORS (gray line where temporal interruption is)
    for i_pre, x_t in enumerate(plot_times[1:]):
        # if epochs are more than 5 minutes separated
        if x_t - plot_times[i_pre] > 300:
            mvc_ax.axvline(i_pre + 1, ymin=0, ymax=50,
                       color='lightgray', lw=5, alpha=.8,)

    # set correct frequencies on Y-axis
    ytickhop = 8
    mvc_ax.set_ylim(0, plot_values.shape[1])
    mvc_ax.set_yticks(range(plot_values.shape[1])[::ytickhop])
    mvc_ax.set_yticklabels(plot_freqs[::ytickhop])
    mvc_ax.set_ylabel('Frequency (Hz)', size=fs + 2)
    # set correct times on X-axis
    xtickhop = 6  #int(len(plot_times) / 9)
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    mvc_ax.set_xticks(np.linspace(0, plot_values.shape[0] - 1, len(xticklabs)))
    mvc_ax.set_xticklabels(np.around(xticklabs / 60, 1))
    mvc_ax.set_xlabel('Time (minutes after LDOPA)', size=fs + 2)

    return fig, axes


def subplot_cdrs(fig, axes, fs, sub, plot_times, n_plot_ax=0):
    """
    reate subplot with Levodopa Induced Dyskinesia scores
    rated with the Clinical Dyskinesia Rating Scale
    
    In current workflow: always plot in first subplot
    """
    ax = axes[n_plot_ax]
    # Plot CDRS every 10 minutes
    try:
        scores, _, _ = importClin.run_import_clinInfo(sub=sub)
        # get closest CDRS score to epoch_time
        if type(scores) == type(None):
            raise ValueError('None scores')
        epoch_clin_scores = [scores.iloc[
            np.argmin(abs(m - scores['dopa_time']))
        ]['CDRS_total'] for m  in (plot_times / 60)]

        ax.plot(epoch_clin_scores, color='orange',
                     alpha=.6, lw=3,)
        ax.set_ylim(0, 20)
        ax.set_ylabel('CDRS (bilateral sum)')

    except FileNotFoundError:
        print(f'No clin scores found for sub {sub}')
    
    # PLOT LID-timings (observed Start and Peak)
    lid_timings = importClin.get_seconds_of_LID_start()[sub]
    lid_clrs = {'start': 'green', 'peak': 'orange'}
    for timing in lid_clrs:
        lid_i = np.argmin(abs(
            plot_times -
            getattr(lid_timings, f"t_{timing}")
        ))
        ax.axhline(lid_i, y0=0, y1=20, ls='--', lw=5,
                        color=lid_clrs[timing], alpha=.5,
                        label=f'LID-{timing}',
        )
    
    xtickhop = 6
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    ax.set_xticks(np.linspace(0, len(plot_times) - 1, len(xticklabs)))

    return fig, axes


def subplot_acc(fig, axes):
    """
    Create subplot with Accelerometer activity
    """


    return fig, axes

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

