"""
Plot overview TimeFreqs based on
PSDs from SSD timeseries
"""

# import public functions
from os.path import join, exists
from os import makedirs
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# import own functions
import lfpecog_features.get_ssd_data as ssd
import lfpecog_analysis.ft_processing_helpers as ftProc
import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq
from lfpecog_analysis.get_acc_task_derivs import (
    get_raw_acc_traces, signalvectormagn
)

from lfpecog_plotting.plotHelpers import get_colors, remove_duplicate_legend
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_cdrs_specific, get_ecog_side
)
from utils.utils_fileManagement import get_project_path


def plot_indiv_ssd_timefreq_allSources(
    sub,
    fig_name_base = 'ssdTimeFreq_ALL_vsLID',
    LOG_POWER = False,
    BASELINE_CORRECT = False,
    PERC_CHANGE = False,
    ZSCORE = True,
    CAT_CDRS = False,
    DATA_VERSION='v4.0',
    FT_VERSION='v4',
    SAVE_PLOT = False,
    SHOW_PLOT = False,
    CDRS_RATER='Jeroen',
    FORCE_PSD_CREATION=False,
    FOR_FIG=False,
):
    # adjust
    fig_name = f'{fig_name_base}_sub{sub}'
    if LOG_POWER: fig_name += '_logPow'
    if BASELINE_CORRECT: fig_name += '_blCorr'
    if PERC_CHANGE: fig_name += 'PerCh'
    if ZSCORE: fig_name += '_Z'

    PSD_ssd_dict = ssd_TimeFreq.get_SSD_timeFreq(
        sub=sub, DATA_VERSION=DATA_VERSION,
        FT_VERSION=FT_VERSION,
        FORCE_PSD_CREATION=FORCE_PSD_CREATION,)
    sources = PSD_ssd_dict.keys()
    
    # CREATE FIGURE
    h_ratios = [1, .1, 1, .4] * len(sources)
    h_ratios = h_ratios[:-1]
    w_ratios = [.01, 1, 0.01, .05]
    gridspec = dict(hspace=0.0, height_ratios=h_ratios, width_ratios=w_ratios,)
    fsize=16

    fig, axes = plt.subplots(nrows=len(sources)*4 - 1,
                            ncols=len(w_ratios),
                            gridspec_kw=gridspec,
                            figsize=(18, 3*len(sources)),
                            sharex='col',
                            # constrained_layout=True,
                            )
    tf_col=1

    if FOR_FIG:
        alltimes = [PSD_ssd_dict[s]['times'] for s in sources]
        uniq_bool = np.in1d(alltimes[0], alltimes[1])
        uniq_times = np.array(alltimes[0])[uniq_bool]
        uniq_bool = np.in1d(uniq_times, alltimes[2])
        uniq_times = uniq_times[uniq_bool]
    else:
        uniq_times = False


    for i_s, source in enumerate(sources):

        if ZSCORE: vmin, vmax, cmap = -4, 4, 'bwr'  #'afmhot'  'hot'  'bwr'
        if LOG_POWER: vmin, vmax, cmap = -3, 0, 'viridis'

        # get COMBINED SSDd SPECTRAL timeseries
        psd_source_dict = PSD_ssd_dict[source]
        tf_values = psd_source_dict['values'].copy()
        tf_times = psd_source_dict['times'].copy()
        tf_freqs = psd_source_dict['freqs'].copy()
        if isinstance(tf_times, list): tf_times = np.array(tf_times)
        if isinstance(tf_values, list): tf_values = np.array(tf_values)
        if isinstance(tf_freqs, list): tf_freqs = np.array(tf_freqs)

        # CHECK TIMESTAMPS TO EXCLUDE DOUBLE DATA
        if sum(np.diff(tf_times) < 0) > 0:
            # find negative time jumps and get n-doubled samples (double data)
            neg_ch = np.where(np.diff(tf_times) < 0)[0]
            idx_sizes = [(i, abs(np.diff(tf_times)[i]) + 1) for i in neg_ch]
            # create bool selector that is 0 for double values
            sel = np.ones_like(tf_times)
            for i,n in idx_sizes: sel[int(i):int(i+n+1)] = 0
            # use bool-selector to drop double samples
            tf_times = tf_times[sel.astype(bool)]
            tf_values = tf_values[:, sel.astype(bool)]
        # double check
        assert sum(np.diff(tf_times) < 0) == 0, 'tf_times contain double data'
        
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
        for bodyside in ['full',]: #['bilat', 'left', 'right']:
            # try:
            print(f'load {bodyside} cdrs for {sub}, {source}')
            cdrs_values[bodyside] = ftProc.find_select_nearest_CDRS_for_ephys(
                sub=sub,
                ft_times=tf_times / 60,
                side=bodyside,
                cdrs_rater=CDRS_RATER,
                INCL_CORE_CDRS=True,
            )

            # convert CDRS labels into categories
            if CAT_CDRS:
                cdrs_values[bodyside] = ftProc.categorical_CDRS(
                    y_full_scale=cdrs_values[bodyside],
                    time_minutes=tf_times / 60,
                    preLID_separate=False,
                    convert_inBetween_zeros=False,
                )
            # except ValueError:
            #     print(f'Error during importing CDRS scores sub {sub}')
        
        # PLOT TIMEFREQ AND DYSKIENSIA SCORES
        (axes[2 + (i_s*4), tf_col],
         axes[0 + (i_s*4), tf_col],
         im_cbar, legend_hands_labels) = plot_splitted_timefreq_ax(
            ax_down=axes[2 + (i_s*4), tf_col], ax_up=axes[0 + (i_s*4), tf_col],
            tf_values=tf_values,
            tf_times=tf_times.copy(),
            tf_freqs=tf_freqs,
            cmap=cmap, vmin=vmin, vmax=vmax,
            cdrs_values=cdrs_values,
            FOR_FIG=FOR_FIG,
            uniq_times=uniq_times,
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
    if LOG_POWER: cbar_label = f'Log {cbar_label}'
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
    other_axes_rows = [1+(i*4) for i in np.arange(len(sources))] +[
        3+(i*4) for i in np.arange(len(sources))][:-1]
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
    
    if not FOR_FIG:
        xticks = np.linspace(round(tf_times[0] / 60, -1), round(tf_times[-1] / 60, -1), 5)
        if 0 not in xticks: xticks = sorted(np.append(xticks, 0))
        axes[-1, tf_col].set_xticks(np.array(xticks) * 60, size=fsize,)
        axes[-1, tf_col].set_xticklabels(np.around(xticks), size=fsize)

    elif FOR_FIG:
        xticks=[0, 10, 20, 30, 40, 60]
        xtick_idx = [np.argmin(abs(tf_times - (t * 60))) for t in xticks]
        axes[-1, tf_col].set_xticks(xtick_idx, size=fsize,)
        axes[-1, tf_col].set_xticklabels(xticks, size=fsize)


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
        path = join(get_project_path('figures'),
                        'ft_exploration',
                        f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                        'ssd_timeFreqs')
        if not exists(path): makedirs(path)
        if FOR_FIG:
            fig_name += '.pdf'
            plt.savefig(join(path, fig_name), facecolor='w',)
            
        else:
            plt.savefig(join(path, fig_name),
                    facecolor='w', dpi=300,)
        

        print(f'sub-{sub}, figure saved: {fig_name}')

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()
    del(fig_name)



def plot_splitted_timefreq_ax(
    ax_down, ax_up, tf_values, tf_times, tf_freqs,
    cmap, vmin, vmax, win_spacing=1, fsize=16,
    f_lims={'down': (4, 35), 'up': (60, 90)},
    cdrs_values=False, SKIP_BILAT_CDRS=True,
    FOR_FIG: bool = False, uniq_times=False,
    no_return=False, only_return_cbar=False,
    FOR_FIG1=False,  # plot in minutes on x axis
):
    """
        - win_spacing: in seconds
    """
    # prepare timefreq array data for colormesh
    C = tf_values.copy()

    if FOR_FIG:
        t_sel = np.in1d(tf_times, uniq_times)
        C = C[:, t_sel]
        tf_times = tf_times[t_sel]
        cdrs_values['full'] = cdrs_values['full'][t_sel]


    # to fix correct timescale (x-axis), add nan-data blocks
    if FOR_FIG1:
        x = tf_times
        y = tf_freqs
    
    elif not FOR_FIG:
        tdiffs_idx = [(t, i) for i, t in enumerate(np.diff(tf_times)) if t>win_spacing]

        for t, i in tdiffs_idx[::-1]:  # in reverse order to not change the relevant indices on the go
            pad = np.array([[np.nan] * C.shape[0]] * int((t - win_spacing) / win_spacing))
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
    
    elif FOR_FIG:
        # create x and y
        x = np.arange(len(tf_times))
        y = tf_freqs
        

    # blank non-SSD freqs
    f_sel = np.logical_and(y > 35, y < 60)
    C[f_sel] = np.nan

    # PLOT COLORMESHES
    im_down = ax_down.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_down.set_ylim(f_lims['down'])
    im_up = ax_up.pcolormesh(x, y, C, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_up.set_ylim(f_lims['up'])

    linestyles = ['solid', 'dashed', 'dotted']
    for i_ax, ax in enumerate([ax_down, ax_up]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(size=fsize, labelsize=fsize, axis='y')
        
        # ADD CDRS
        if (not isinstance(cdrs_values, np.ndarray) and
            not isinstance(cdrs_values, dict)): continue

        ax_cdrs = ax.twinx()
        if cmap in ['viridis', 'plasma']: cdrs_col = ['lightgray', 'firebrick', 'darkorange']
        elif cmap in ['bwr', 'PuOr', 'PuOr_r']: cdrs_col = ['gray', 'orange', 'darkolivegreen']
        ax_cdrs.tick_params(size=fsize, labelsize=fsize, axis='y')
        for r in ['top', 'bottom', 'right']: ax_cdrs.spines[r].set_visible(False)
        
        try:
            if isinstance(cdrs_values, dict):
                for i_side, side in enumerate(cdrs_values.keys()):
                    if side == 'bilat' and SKIP_BILAT_CDRS: continue
                    cdrs_values[side][cdrs_values[side] > 5] = 5
                    ax_cdrs.plot(x, cdrs_values[side], ls=linestyles[i_side],
                                color=cdrs_col[i_side], lw=5, alpha=.8,
                                label=f'Dyskinesia {side}')
            else:
                cdrs_values[cdrs_values > 5] = 5
                ax_cdrs.plot(x, cdrs_values, color=cdrs_col[0], lw=5, alpha=.5)
        except:
            print(f'CDRS not plotted')
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


    if only_return_cbar: return im_up
    
    elif not no_return: return ax_down, ax_up, im_up, leg_hands_labs



def plot_indiv_timefreq_FIG1(
    sub,
    plot_tf_dict, tf_freqs,
    t_real_minutes, t_cont,
    task_win, acc_win, cdrs_win,
    xticks, xtlabels,
    fig_name = 'FIG1_0000',
    LOG_POWER = True,
    BASELINE_CORRECT = False,
    PERC_CHANGE = False,
    ZSCORE = False,
    SAVE_PLOT = False,
    SHOW_PLOT = False,
    FOR_FIG=False,
    MAX_CDRS: int = 12,
):
    import matplotlib
    task_clrs = matplotlib.cm.get_cmap('Set2')
    
    sources = plot_tf_dict.keys()
  
    ### CREATE FIGURE
    std_rows = [.35, .35, .35, .05]
    h_ratios = std_rows + [1, .1, 1, .1] * len(sources)
    h_ratios = h_ratios[:-1]
    w_ratios = [.01, 1, 0.01, .05]
    gridspec = dict(hspace=0.0, height_ratios=h_ratios, width_ratios=w_ratios,)
    fsize=16

    fig, axes = plt.subplots(nrows=len(h_ratios),
                            ncols=len(w_ratios),
                            gridspec_kw=gridspec,
                            figsize=(12, 6),
                            constrained_layout=True,
                            )
    tf_col=1
    # share all axes
    for ax in axes[1:, tf_col]: ax.sharex(axes[0, tf_col])

    if FOR_FIG:
        alltimes = [plot_tf_dict[s]['times'] for s in sources]
        uniq_bool = np.in1d(alltimes[0], alltimes[1])
        uniq_times = np.array(alltimes[0])[uniq_bool]
        uniq_bool = np.in1d(uniq_times, alltimes[2])
        uniq_times = uniq_times[uniq_bool]
    else:
        uniq_times = False
    
    ### PLOT TASK ROW
    heat_task = axes[0, 1].pcolor(
        t_cont, np.arange(2), np.atleast_2d([np.zeros(len(t_cont)),
                                             task_win.ravel()]),
        vmin=0, vmax=3, cmap='Set2',
    )
    axes[0, 1].set_ylim(.5, 1)
    for i_col, task_name in zip([0, 2, 5],
                                ['rest', 'tap', 'pause']):
        axes[0, 1].scatter([], [], color=task_clrs(i_col),
                           s=100, marker='s',
                           label=task_name,)
    axes[0, 1].legend(ncol=3, bbox_to_anchor=(.5, .95),
                      loc='lower center', fontsize=fsize,
                      frameon=False,)


    ### PLOT ACC ROW
    axes[1, 1].plot(t_cont, acc_win, color='rosybrown', alpha=.9, )
    axes[1, 1].set_ylim(-.5, 3)


    ### PLOT DYSKINESIA ROW HEATMAP
    heat_lid = axes[2, 1].pcolor(t_cont, np.arange(2),
                                 np.atleast_2d([np.zeros(len(t_cont)),
                                                cdrs_win.astype(float)]),
                                 vmin=0, vmax=MAX_CDRS, cmap='Reds',)
    axes[2, 1].set_ylim(0.5, 1)
    

    ### Plot heatmaps for TimeFreqs
    for i_s, source in enumerate(sources):

        if ZSCORE: vmin, vmax, cmap = -4, 4, 'bwr'  #'afmhot'  'hot'  'bwr'
        if LOG_POWER: vmin, vmax, cmap = -3, 0, 'viridis'

        tf_values = plot_tf_dict[source]

        if ZSCORE:
            for row_i, row_data in enumerate(tf_values):
                tf_values[row_i] = (tf_values[row_i] - np.nanmean(tf_values[row_i])
                                    ) / np.nanstd(tf_values[row_i])

        if BASELINE_CORRECT:
            tf_values = ssd_TimeFreq.correct_timeFreq_baseline(
                tf_values, t_cont, perc_change=PERC_CHANGE
            )
        if LOG_POWER: tf_values = np.log10(tf_values)
            
        # PLOT TIMEFREQ AND DYSKIENSIA SCORES
        ax_low = axes[len(std_rows) + 2 + (i_s*4), tf_col]
        ax_high=axes[len(std_rows) + (i_s*4), tf_col]
        tf_cbar = plot_splitted_timefreq_ax(
            ax_down=ax_low, ax_up=ax_high,
            tf_values=tf_values.copy().T,
            tf_times=t_cont.copy(),
            tf_freqs=tf_freqs,
            cmap=cmap, vmin=vmin, vmax=vmax,
            only_return_cbar=True,
            FOR_FIG1=True,
        )
        ax_low.set_yticks([10, 20, 30], size=fsize-8,)
        ax_low.set_yticklabels([10, 20, 30], size=fsize-2,)
        ax_high.set_yticks([65, 75, 85], size=fsize-8,)
        ax_high.set_yticklabels([65, 75, 85], size=fsize-2,)
        
        # PLOT SOURCE LEFT OF TF
        slab = source.replace('_', ' ')
        slab = slab.replace('lfp', 'STN')
        slab = slab.replace('ecog', 'ECoG')
        slab = slab.replace('left', 'L')
        slab = slab.replace('right', 'R')
        fig.text(x=.07, y=.65-(i_s * .22), s=slab,
             va='center', ha='right', size=fsize+4,
             rotation=90,  # weight='bold',
             )

    # plot colorbar
    gs = axes[0, -1].get_gridspec()
    for ax in axes[:, -1]: ax.remove()
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar_label = 'Power (a.u.)'
    if LOG_POWER: cbar_label = f'Log {cbar_label}'
    if BASELINE_CORRECT and PERC_CHANGE: cbar_label = 'Power %-change vs pre L-DOPA' + cbar_label[5:]
    elif BASELINE_CORRECT and PERC_CHANGE and ZSCORE: cbar_label = 'Z-scored Power %-change vs pre L-DOPA (%)'
    elif ZSCORE and BASELINE_CORRECT: cbar_label = 'Z-scored Power vs pre L-DOPA (a.u.)'
    elif ZSCORE: cbar_label = f'Z-scored {cbar_label}'
    cbar = fig.colorbar(tf_cbar, ax=cbar_ax, pad=.12,
                        use_gridspec=True, fraction=.9, aspect=50,)
    # cbar = fig.colorbar(im_cbar, ax=axes.ravel().tolist(), pad=.1)  # ravel axes to plot on full figure
    # cbar.ax.set_yticks(np.linspace(vmin, vmax, 5), size=fsize)
    # cbar.ax.set_yticklabels(np.around(np.linspace(vmin, vmax, 5), 1), size=fsize)
    cbar.ax.set_yticks([])
    cbar.ax.set_ylabel(cbar_label, rotation=270, size=fsize + 4, weight='bold',)
    cbar.ax.get_yaxis().labelpad=20
    for r in ['bottom', 'left', 'right', 'top']:
        cbar_ax.spines[r].set_visible(False)
        cbar_ax.set_xticks([],)
        cbar_ax.set_yticks([],)

    # Plot LID
    fig.text(x=.12, y=.78, s='LID',
             va='center', ha='right', size=fsize-4,
             rotation=0, weight='bold',)
    # Plot MOVE
    fig.text(x=.12, y=.82, s='MOVE',
             va='center', ha='right', size=fsize-4,
             rotation=0,  weight='bold',
             )
    # Plot TASK
    fig.text(x=.12, y=.86, s='TASK',
             va='center', ha='right', size=fsize-4,
             rotation=0,  weight='bold',
             )
    # Plot y-label Frequency left
    fig.text(x=.03, y=.5, s='Frequency (Hz)',
             va='center', ha='right', size=fsize+4,
             rotation=90, weight='bold',)


    # REMOVE A LOT OF TICKS
    for ax in axes.flatten():
        for r in ['bottom', 'left', 'right', 'top']:
            ax.spines[r].set_visible(False)
    for ax in axes[0, :]:
        ax.set_xticks([],)
        ax.set_yticks([],)
    for ax in axes[:, 0]:
        ax.set_xticks([],)
        ax.set_yticks([],)
    for ax in axes[:, -2]:
        ax.set_xticks([],)
        ax.set_yticks([],)
    for i_ax,ax in enumerate(axes[:, tf_col]):
        if i_ax in np.arange(4, 16, 2): continue
        ax.set_xticks([],)
        ax.set_yticks([],)
    

    # Plot xlabel at bottom
    axes[-1, tf_col].set_xlabel('Time (minutes after L-Dopa intake)',
                        size=fsize+4, weight='bold',)
    axes[-1, tf_col].set_xticks(xticks, size=fsize+4,)
    axes[-1, tf_col].set_xticklabels(xtlabels, size=fsize+4,)
    
   
    plt.subplots_adjust(wspace=0, hspace=0)


    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                        'final_Q1_2024',
                        'FIG1_timefreq')
        if FOR_FIG:
            fig_name += '.pdf'
            plt.savefig(join(path, fig_name), facecolor='w',)
            
        else:
            plt.savefig(join(path, fig_name),
                    facecolor='w', dpi=300,)
        

        print(f'sub-{sub}, figure saved: {fig_name}')

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


def prep_cont_tf_arrays(sub_ssdPsd_dict, win_min=.166):
    
    t_overall =  np.arange(0, 66, win_min)

    new_tf = {}

    for s in sub_ssdPsd_dict.keys():

        source_times = np.array(sub_ssdPsd_dict[s]['times']) / 60
        source_tf = np.array(sub_ssdPsd_dict[s]['values'])
        tf_freqs = np.array(sub_ssdPsd_dict[s]['freqs'])

        tftemp = []

        for t0 in t_overall:
            t_sel = np.logical_and(source_times >= t0,
                                   source_times < (t0 + win_min))
            mean_tf = np.mean(source_tf[:, t_sel], axis=1)
            tftemp.append(mean_tf)
        
        new_tf[s] = np.array(tftemp)

    nans = np.ones(new_tf[s].shape[0])

    for s in sub_ssdPsd_dict.keys():
        tempnan = np.any(np.isnan(new_tf[s]), axis=1)
        # print(len(tempnan), sum(tempnan))
        nans[tempnan] = 0

    # IF SUBJECT 008: remove min 19-22
    sel = np.logical_and(t_overall > 19.5, t_overall < 23)
    nans[sel] = 0

    # print(len(nans), sum(nans))
    t_overall = t_overall[nans.astype(bool)]
    for s in sub_ssdPsd_dict.keys():
        new_tf[s] = new_tf[s][nans.astype(bool), :]

    # New x_axis to prevent blurring due to missing data
    x_cont = np.arange(len(t_overall))

    # Create new xticks
    xtlabels = np.arange(0, 65, 10)
    xticks = [np.argmin(abs(t_overall - t)) for t in xtlabels]

    return new_tf, x_cont, tf_freqs, t_overall, xticks, xtlabels



def get_task_acc_for_tfPlot(
    sub='008', plot_x=False, win_min=.16,
    return_task=False, TASK_NUM = True):
    """
    returns smoothed and windowed acc, times, and parallel tasks
    """
    accl = get_raw_acc_traces(sub=sub, side='left', data_version='v4.0')
    accfs = accl.fs
    acctimes = accl.data[:, 0] / 60
    tasks = accl.data[:, 4]
    accl = signalvectormagn(accl.data[:, 1:4])
    accl = (accl - np.mean(accl)) / np.std(accl)

    accr = get_raw_acc_traces(sub=sub, side='right', data_version='v4.0')
    accr = signalvectormagn(accr.data[:, 1:4])
    accr = (accr - np.mean(accr)) / np.std(accr)

    accmean = np.mean([accl, accr], axis=0)

    accsmooth = savgol_filter(accmean, window_length=accfs, polyorder=3)

    win_samples = accfs * win_min * 60
    acc_win_times = []
    acc_win = []
    task_win = []

    if isinstance(plot_x, np.ndarray):
        for t in plot_x:
            i = np.argmin(abs(acctimes - t))
            acc_win_times.append(acctimes[i])
            acc_win.append(np.mean(accsmooth[i:int(i + win_samples)]))
            task_win.append(tasks[i])

    else:
        # window from start
        for i in np.arange(0, len(acctimes), win_samples):
            acc_win_times.append(acctimes[i])
            acc_win.append(np.mean(accsmooth[i:i + win_samples]))
            task_win.append(tasks[i])
        
    acc_win_times = np.array(acc_win_times)
    acc_win = np.array(acc_win)
    
    if TASK_NUM and return_task:
        task_win_num = []
        for t in task_win:
            if t == 'rest': task_win_num.append(0)
            if t == 'tap': task_win_num.append(1)
            if t == 'free': task_win_num.append(2)
        task_win = task_win_num
    
    task_win = np.atleast_2d(task_win)
    
    
    if return_task:
        return acc_win_times, acc_win, task_win
    
    else:
        return acc_win_times, acc_win


def correct_xticks_50min(xticks, xtlabels):

    if 50 in list(xtlabels):
        i_50 = np.where(xtlabels == 50)[0][0]
        xtlabels = list(xtlabels)
        xtlabels[i_50 - 1] = '40 **'
        xtlabels.pop(i_50)
        xticks = np.delete(xticks, i_50)

    return xticks, xtlabels