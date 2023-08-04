"""
Plot overview TimeFreqs based on
PSDs from SSD timeseries
"""

# import public functions
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# import own functions
import lfpecog_features.get_ssd_data as ssd
import lfpecog_analysis.ft_processing_helpers as ftProc
import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq

from lfpecog_plotting.plotHelpers import get_colors, remove_duplicate_legend
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_cdrs_specific, get_ecog_side
)
from utils.utils_fileManagement import get_project_path


def plot_indiv_ssd_timefreq_allSources(
    sub,
    fig_name = 'ssdTimeFreq_ALL_vsLID',
    LOG_POWER = False,
    BASELINE_CORRECT = False,
    PERC_CHANGE = False,
    ZSCORE = True,
    CAT_CDRS = False,
    DATA_VERSION='v4.0',
    FT_VERSION='v4',
    SAVE_PLOT = False,
    SHOW_PLOT = False,
):
    # adjust 
    fig_name += f'_sub{sub}'
    if LOG_POWER: fig_name += '_logPow'
    if BASELINE_CORRECT: fig_name += '_blCorr'
    if PERC_CHANGE: fig_name += 'PerCh'
    if ZSCORE: fig_name += '_Z'

    ssd_subClass = ssd.get_subject_SSDs(
        sub=sub,
        incl_stn=True,
        incl_ecog=True,
        ft_setting_fname=f'ftExtr_spectral_{FT_VERSION}.json',)

    n_sources = len(ssd_subClass.ephys_sources)
    fsize=16

    # CREATE FIGURE
    h_ratios = [1, .1, 1, .4] * n_sources
    h_ratios = h_ratios[:-1]
    w_ratios = [.01, 1, 0.01, .05]
    gridspec = dict(hspace=0.0, height_ratios=h_ratios, width_ratios=w_ratios,)

    fig, axes = plt.subplots(nrows=n_sources*4 - 1,
                            ncols=len(w_ratios),
                            gridspec_kw=gridspec,
                            figsize=(18, 3*n_sources),
                            sharex='col',
                            # constrained_layout=True,
                            )
    tf_col=1



    for i_s, source in enumerate(ssd_subClass.ephys_sources):

        subSourceSSD = getattr(ssd_subClass, source)

        # define COLOBAR settings based on data settings
        # if LOG_POWER:
        #     if BASELINE_CORRECT:
        #         if 'lfp' in source: vmin, vmax = -5, 5
        #         elif 'ecog' in source: vmin, vmax = -2.5, 2.5
        #         if PERC_CHANGE: vmin, vmax, cmap = -10, 10, 'PuOr_r'
        #     else: vmin, vmax, cmap = -10, 0, 'plasma'

        # elif ZSCORE: vmin, vmax, cmap = -2, 2, 'PuOr_r'

        # else:
        #     if BASELINE_CORRECT:
        #         vmin, vmax, cmap = -.2, .2, 'PuOr_r'
        #         if PERC_CHANGE: vmin, vmax = -10, 10
        #     else: vmin, vmax, cmap = 0, .2, 'viridis'
        vmin, vmax, cmap = -4, 4, 'PuOr_r'


        # get COMBINED SSDd SPECTRAL timeseries
        (timefreqArr, tf_times,
        min_f, max_f) = ssd_TimeFreq.create_SSD_timeFreqArray(subSourceSSD)
        tf_values = timefreqArr.values

        if ZSCORE:
            for row_i, row_data in enumerate(tf_values):
                tf_values[row_i] = (tf_values[row_i] - np.nanmean(tf_values[row_i])
                                    ) / np.nanstd(tf_values[row_i])

        if BASELINE_CORRECT:
            tf_values = ssd_TimeFreq.correct_timeFreq_baseline(
                tf_values, tf_times, perc_change=PERC_CHANGE
            )
        if LOG_POWER: tf_values = np.log10(tf_values)

        # get nearest CDRS values for ephys-timings
        cdrs_values = {}
        for bodyside in ['bilat', 'left', 'right']:
            _, cdrs_values[bodyside] = ftProc.find_select_nearest_CDRS_for_ephys(
                sub=subSourceSSD.sub,
                ft_times=tf_times / 60,
                side=bodyside,
                cdrs_rater='Patricia',
            )

            # convert CDRS labels into categories
            if CAT_CDRS:
                cdrs_values[bodyside] = ftProc.categorical_CDRS(
                    y_full_scale=cdrs_values[bodyside],
                    time_minutes=tf_times / 60,
                    preLID_minutes=0,
                    preLID_separate=False,
                    convert_inBetween_zeros='mild',
                )

        # PLOT TIMEFREQ AND DYSKIENSIA SCORES
        (axes[2 + (i_s*4), tf_col],
         axes[0 + (i_s*4), tf_col],
         im_cbar, legend_hands_labels) = plot_splitted_timefreq_ax(
            ax_down=axes[2 + (i_s*4), tf_col], ax_up=axes[0 + (i_s*4), tf_col],
            tf_values=tf_values,
            tf_times=tf_times,
            tf_freqs=timefreqArr.index,
            cmap=cmap, vmin=vmin, vmax=vmax,
            cdrs_values=cdrs_values,
        )
        
        # PLOT Y LABEL FOR 2nd Y AXIS DYSKINESIA
        if isinstance(cdrs_values, np.ndarray) or isinstance(cdrs_values, dict):
            dys_ylabel = 'Clinical Dyskinesia'
            if CAT_CDRS: dys_ylabel += ' (categorical)'
            elif not CAT_CDRS: dys_ylabel += ' (CDRS scores)'
            fig.text(x=.9, y=.5, s=dys_ylabel,
                    va='center', ha='left', size=fsize+4,
                    rotation=-90, weight='bold',)

        axes[0 + (i_s*4), tf_col].set_title(f'{source}',
                                            size=fsize+4,
                                            weight='bold',
                                            loc='left',)

    # plot colorbar
    gs = axes[0, -1].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar_label = 'Power (a.u.)'
    if BASELINE_CORRECT and PERC_CHANGE: cbar_label = 'Power %-change vs pre L-DOPA' + cbar_label[5:]
    elif BASELINE_CORRECT and PERC_CHANGE and ZSCORE: cbar_label = 'Z-scored Power %-change vs pre L-DOPA (%)'
    elif ZSCORE and BASELINE_CORRECT: cbar_label = 'Z-scored Power vs pre L-DOPA (a.u.)'
    elif ZSCORE: cbar_label = f'Z-scored {cbar_label}'

    cbar = fig.colorbar(im_cbar, ax=cbar_ax, pad=.12,
                        use_gridspec=True, fraction=.9, aspect=50,)
    # cbar = fig.colorbar(im_cbar, ax=axes.ravel().tolist(), pad=.1)  # ravel axes to plot on full figure
    cbar.ax.set_yticks(np.linspace(vmin, vmax, 5), size=fsize)
    cbar.ax.set_yticklabels(np.around(np.linspace(vmin, vmax, 5), 1), size=fsize)
    cbar.ax.set_ylabel(cbar_label, rotation=270, size=fsize + 4, weight='bold',)
    cbar.ax.get_yaxis().labelpad=10
    for r in ['bottom', 'left', 'right', 'top']:
        cbar_ax.spines[r].set_visible(False)
        cbar_ax.set_xticks([],)
        cbar_ax.set_yticks([],)

    # Plot y-label Frequency left
    fig.text(x=.02, y=.5, s='Frequency (Hz)',
                va='center', ha='right', size=fsize+4,
                rotation=90, weight='bold',)

    # remove axis and labels from white space axes
    other_axes_rows = [1+(i*4) for i in np.arange(n_sources)] +[
        3+(i*4) for i in np.arange(n_sources)][:-1]
    for ax_r in other_axes_rows:
        axes[ax_r, tf_col].set_xticks([],)
        axes[ax_r, tf_col].set_yticks([],)
        for r in ['bottom', 'left', 'right', 'top']:
            axes[ax_r, tf_col].spines[r].set_visible(False)
    other_axes_cols = [0, -2]
    for ax_c in other_axes_cols:
        for ax_r in np.arange(len(h_ratios)):
            axes[ax_r, ax_c].set_xticks([],)
            axes[ax_r, ax_c].set_yticks([],)
            for r in ['bottom', 'left', 'right', 'top']:
                axes[ax_r, ax_c].spines[r].set_visible(False)

    # Plot xlabel at bottom
    axes[-1, tf_col].set_xlabel('Time (minutes after L-Dopa intake)',
                        size=fsize+4, weight='bold',)
    xticks = np.linspace(round(tf_times[0] / 60, -1), round(tf_times[-1] / 60, -1), 5)
    if 0 not in xticks: xticks = sorted(np.append(xticks, 0))
    axes[-1, tf_col].set_xticks(np.array(xticks) * 60, size=fsize,)
    axes[-1, tf_col].set_xticklabels(np.around(xticks), size=fsize)

    hands, labels = remove_duplicate_legend(legend_hands_labels)
    axbox = axes[0, tf_col].get_position()
    axes[0, tf_col].legend(hands, labels, ncol=3, frameon=False,
               fontsize=fsize, loc='lower left',
               bbox_to_anchor=(.25, 1.01),
            #    bbox_to_anchor=[0.5*axbox.width,
            #                    axbox.y1*.99],
            #    bbox_transform=fig.transFigure,
               )

    plt.tight_layout()
    if SAVE_PLOT:
        plt.savefig(join(get_project_path('figures'),
                        'ft_exploration', DATA_VERSION,
                        'ssd_timeFreqs', fig_name),
                    facecolor='w', dpi=300,)
        print(f'sub-{sub}, figure saved: {fig_name}')

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


def plot_splitted_timefreq_ax(
    ax_down, ax_up, tf_values, tf_times, tf_freqs,
    cmap, vmin, vmax, win_spacing=1, fsize=16,
    f_lims={'down': (4, 35), 'up': (60, 90)},
    cdrs_values=False,
):
    """
        - win_spacing: in seconds
    """
    # prepare timefreq array data for colormesh
    C = tf_values.copy()
    win_spacing = 1  # 5 seconds in between windows
    tdiffs_idx = [(t, i) for i, t in enumerate(np.diff(tf_times)) if t>win_spacing]
    for t, i in tdiffs_idx[::-1]:  # in reverse order to not change the relevant indices on the go
        pad = np.array([[np.nan] * C.shape[0]] * int((t - 1) / win_spacing))
        C = np.insert(C, obj=i, values=pad, axis=1)
        # add nans as well to cdrs-array
        if isinstance(cdrs_values, np.ndarray):
            cdrs_values = np.insert(cdrs_values, obj=i, values=pad[:, 0], )
        elif isinstance(cdrs_values, dict):
            for side in cdrs_values.keys():
                cdrs_values[side] = np.insert(cdrs_values[side], obj=i, values=pad[:, 0], )
    # create x and y
    x = np.arange(tf_times[0], tf_times[-1] + win_spacing, win_spacing)
    y = tf_freqs
    # blank non-SSD freqs
    f_sel = np.logical_and(y > 35, y < 60)
    C[f_sel] = np.nan

    # PLOT COLORMESHES
    im_down = ax_down.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_down.set_ylim(f_lims['down'])
    im_up = ax_up.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_up.set_ylim(f_lims['up'])

    for i_ax, ax in enumerate([ax_down, ax_up]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(size=fsize, labelsize=fsize, axis='y')
        
        # ADD CDRS
        if (not isinstance(cdrs_values, np.ndarray) and
            not isinstance(cdrs_values, dict)): continue

        ax_cdrs = ax.twinx()
        if cmap in ['viridis', 'plasma']: cdrs_col = ['lightgray',]
        elif cmap in ['PuOr', 'PuOr_r']: cdrs_col = ['green', 'blue', 'turquoise']
        ax_cdrs.tick_params(size=fsize, labelsize=fsize, axis='y')
        for r in ['top', 'bottom', 'right']: ax_cdrs.spines[r].set_visible(False)
        
        if isinstance(cdrs_values, dict):
            for i_side, side in enumerate(cdrs_values.keys()):
                cdrs_values[side][cdrs_values[side] > 5] = 5
                ax_cdrs.plot(x, cdrs_values[side],
                             color=cdrs_col[i_side], lw=5, alpha=.5,
                             label=f'Dyskinesia {side}')
        else:
            cdrs_values[cdrs_values > 5] = 5
            ax_cdrs.plot(x, cdrs_values, color=cdrs_col[0], lw=5, alpha=.5)

        ax_cdrs.set_xticks([])

        if i_ax == 0:
            ax_cdrs.set_ylim([0, 2.5])
            ax_cdrs.set_yticks([0, 1, 2])
            
        elif i_ax == 1:
            ax_cdrs.set_ylim([2.5, 5])
            ax_cdrs.set_yticks([3, 4, 5])
            ax_cdrs.set_yticklabels(['3', '4', '>=5'])
            

    ax_up.spines['bottom'].set_visible(False)
    ax_up.tick_params(axis='x', size=4, labelsize=4)  # decrease spacing between plots
    # xticks = np.linspace(round(tf_times[0] / 60, -1), round(tf_times[-1] / 60, -1), 5)
    # if 0 not in xticks: xticks = sorted(np.append(xticks, 0))
    # ax_down.set_xticks(np.array(xticks) * 60, size=fsize,)
    # ax_down.set_xticklabels(np.around(xticks), size=fsize)
    ax_down.set_yticks([10, 20, 30])
    ax_up.set_yticks([60, 70, 80, 90])

    leg_hands_labs = plt.gca().get_legend_handles_labels()


    return ax_down, ax_up, im_up, leg_hands_labs