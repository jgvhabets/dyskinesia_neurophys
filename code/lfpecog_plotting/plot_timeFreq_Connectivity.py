"""Plot time-frequency Plots for connectivity measures"""

# import
from os.path import join, exists
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

import lfpecog_preproc.preproc_import_scores_annotations as importClin
from lfpecog_plotting.timefreqSubplot_Acc import subplot_acc
from lfpecog_plotting.timefreqSubplot_Clinical import subplot_cdrs


def plot_mvc(
    sub, plot_data, plot_freqs, plot_times,
    fs=16, cmap='viridis', mvc_method='mic',
    to_save=False, save_path=None, fname=None,
    add_CDRS=True, add_ACC=True, add_task=False, acc_plottype='bars',
    data_version=None, mvc_ax=1, winLen_sec=None,
    plot_params={'sharex': False}, grid_params={},
):
    # define general plot settings
    num_axes = 1
    if add_CDRS: num_axes += 1
    if add_ACC: num_axes += 1

    if num_axes > 1:
        plot_params['sharex']: True
        if num_axes == 2:
            if add_CDRS: grid_params['height_ratios'] = [1, 3]  # MVC and CDRS
            else:
                grid_params['height_ratios'] = [3, 1]  # MVC and ACC
                mvc_ax = 0
        elif num_axes == 3: grid_params['height_ratios'] = [1, 3, 1]  # MVC and ACC and CDRS

    fig, axes = plt.subplots(
        num_axes, 1, gridspec_kw=grid_params, **plot_params,
        figsize=(16, 8 + (4 * (num_axes - 1))),
        constrained_layout=True,)
    
    # Plot MVC features in subplot
    fig, axes = subplot_mvc(fig=fig, axes=axes, num_axes=num_axes,
                            mvc_ax=mvc_ax, fs=fs, cmap=cmap,
                            plot_values=plot_data,
                            plot_freqs=plot_freqs, plot_times=plot_times,
                            sub=sub, mvc_method=mvc_method,)

    # Plot clinical LID scores (CDRS) in subplot
    if add_CDRS:
        fig, axes = subplot_cdrs(fig=fig, axes=axes, fs=fs,
                                 sub=sub, plot_times=plot_times)

    # Plot Accelerometer features in subplot
    if add_ACC:
        fig, axes = subplot_acc(fig=fig, axes=axes, fs=fs,
                                sub=sub, plot_times=plot_times,
                                data_version=data_version,
                                winLen_sec=winLen_sec,
                                plot_task=add_task,
                                plot_type=acc_plottype,)
    
    
    if to_save:
        if not exists(save_path): makedirs(save_path)
        plt.savefig(join(save_path, f'{fname}.png'),
                    dpi=300, facecolor='w',)
        print(f'...Figure {fname} saved to {save_path}')

    plt.close()


def subplot_mvc(
    fig, axes, mvc_ax, num_axes, fs, cmap, mvc_method,
    plot_values, plot_freqs, plot_times, sub,
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
        
    Returns:
        - fig, axes with added subplot      
    """
    # define which ax to plot mvc
    if num_axes == 1: ax = axes  # only MVC
    else: ax = axes[mvc_ax]

    ecog_side = importClin.get_ecog_side(sub)

    # plot colormap
    vmax = .51
    im = ax.imshow(plot_values.T, vmin=0, vmax=vmax,
                       cmap=cmap, aspect='auto') # .5
    # plot colorbar
    cbar = fig.colorbar(im, ax=ax, pad=.01, extend='max',)#axes.ravel().tolist())
    cbar.ax.set_yticks(np.arange(0, vmax, .1), size=fs)
    cbar.ax.set_yticklabels(np.around(np.arange(0, vmax, .1), 1), size=fs)
    if mvc_method.lower() == 'mic':
        cv_title = f'abs. Max. Imaginary Coherence ({ecog_side})'
    elif mvc_method.lower() == 'mim':
        cv_title = f'Multivariate Interaction Measure ({ecog_side})'
    cbar.ax.set_ylabel(cv_title, rotation=270, size=fs + 6)
    cbar.ax.get_yaxis().labelpad = 25

    # PLOT JUMP IN TIME INDICATORS (gray line where temporal interruption is)
    for i_pre, x_t in enumerate(plot_times[1:]):
        # if epochs are more than 5 minutes separated
        if x_t - plot_times[i_pre] >= 120:
            ax.axvline(i_pre + .5, ymin=0, ymax=50,
                       color='lightgray', lw=5, alpha=.8,)

    # set correct frequencies on Y-axis
    ytickhop = 8
    ax.set_ylim(0, plot_values.shape[1])
    ax.set_yticks(range(plot_values.shape[1])[::ytickhop])
    ax.set_yticklabels(plot_freqs[::ytickhop])
    ax.set_ylabel('Frequency (Hz)', size=fs + 6)
    # set correct times on X-axis
    xtickhop = 6  #int(len(plot_times) / 9)
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    ax.set_xticks(np.linspace(0, plot_values.shape[0] - 1, len(xticklabs)))
    ax.set_xticklabels(np.around(xticklabs / 60, 1), size=fs + 2)
    ax.set_xlabel('Time (minutes after LDOPA)', size=fs + 6)
    ax.tick_params(axis='both', size=fs, labelsize=fs)
    for side in ['right', 'bottom', 'top']:
        getattr(ax.spines, side).set_visible(False)

    return fig, axes



