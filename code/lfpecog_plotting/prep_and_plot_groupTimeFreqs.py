"""
Plot timefrequencies on group level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs

import utils.utils_fileManagement as utilsFiles
import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq
import lfpecog_analysis.ft_processing_helpers as ftProc


def get_group_timefreq(
    source,
    TARGET: str,
    DATA_VERSION, FT_VERSION,
    MIN_SUBS=3,
    IGNORE_PTS=['011', '104', '106'],
    Z_SCORE_FREQS = False,
    LOG_POWER=False,
):
    """
    Input:
        - TARGET: 'LDOPA' for centered around LDOPA-
            intake, 'LID' for centered around LID onset
    """
    assert TARGET in ['LID', 'LDOPA'], 'should be LID / LDOPA'
    # find available subs
    SUBS = utilsFiles.get_avail_ssd_subs(
        DATA_VERSION=DATA_VERSION,
        FT_VERSION=FT_VERSION,
        IGNORE_PTS=IGNORE_PTS
    )
    # load timefreqs
    TFs = ssd_TimeFreq.get_all_ssd_timeFreqs(
        SUBS=SUBS, FT_VERSION=FT_VERSION,
        DATA_VERSION=DATA_VERSION,
    )
    
    n_sub_incl = 0

    for sub in TFs.keys():
        # timefreq tuple has values; times; freqs
        psx = TFs[sub][source].values
        fx = TFs[sub][source].freqs
        t = TFs[sub][source].times

        scores = ftProc.find_select_nearest_CDRS_for_ephys(
            sub=sub, side='both', INCL_CORE_CDRS=True,
            cdrs_rater='Jeroen',
            ft_times=TFs[sub][source].times/60
        )
        try:
            # define time in seconds of LID start
            lid_start = t[np.where(scores > 0)[0][0]]
        except IndexError:
            print(f'skip sub-{sub}, no dyskinesia')
            continue

        if lid_start < (15 * 60):
            print(f'skip sub-{sub}, dyskinesia '
                f'after {round(lid_start/60)} minutes')
            continue
        
        print(f'INCLUDE sub-{sub}, dyskinesia from {round(lid_start/60)} minutes')

        # include subject and create empty dataframe, with time index
        n_sub_incl += 1
        if TARGET == 'LDOPA': df_times = np.arange(-900, 90*60, 1)
        else: df_times = np.arange(-10*60, 90*60, 1)
        sub_df = pd.DataFrame(index=df_times, columns=fx)

        # shift times of powers to LID onset
        if TARGET == 'LID':
            t -= lid_start

        # z score arr within freq band
        if Z_SCORE_FREQS:
            zpsx = [(p - np.nanmean(p)) / np.nanstd(p) for p in psx]
            psx = np.array(zpsx)
        
        if LOG_POWER:
            psx = np.log(psx.astype(np.float64))

        # fill spectral data into df depend on target
        if TARGET == 'LDOPA':
            # add all negative values as close to zero as possible
            if t[0] < 0:
                if sub == '010':
                    n_neg = np.sum(t < (-14*60))  # only before minus 14 was rest
                    sub_df.loc[-n_neg:-1, fx] = psx[:, t < (-14 * 60)].T
                else:
                    n_neg = np.sum(t < 0)
                    sub_df.loc[-n_neg:-1, fx] = psx[:, t < 0].T
            # add positive values on true time, WITHOUT LID
            pos_sel = np.logical_and(t > 0, t < lid_start)
            pos_sel = np.logical_and(pos_sel, scores == 0)
            pos_t = t[pos_sel]
            sub_df.loc[pos_t, fx] = psx[:, pos_sel].T
        
        elif TARGET == 'LID':
            # add from -10 to later (0 is LID-ONSET, time is already centered)
            lid_sel = t > (-10 * 60)
            lid_t = t[lid_sel]
            sub_df.loc[lid_t, fx] = psx[:, lid_sel].T
        
        if n_sub_incl == 1:
            data_arr_out = sub_df.values
        else:
            data_arr_out = np.dstack([data_arr_out, sub_df.values])
        
    arr_times = sub_df.index.values

    # remove times with only NaNs
    nan_ts = np.isnan(data_arr_out[:, 0, :].astype(float)).all(axis=1)
    data_arr_out = data_arr_out[~nan_ts, :, :]
    arr_times = arr_times[~nan_ts]

    # include minimal 3 subjects per timepoint
    if MIN_SUBS > 1:
        nan_subs = np.isnan(
            data_arr_out[:, 0, :].astype(float)
        ).sum(axis=1)
        present_subs = data_arr_out.shape[-1] - nan_subs
        sub_sel = present_subs >= MIN_SUBS
        # take only included times
        data_arr_out = data_arr_out[sub_sel]
        arr_times = arr_times[sub_sel]
        present_subs = present_subs[sub_sel]
    
        return data_arr_out, arr_times, fx, present_subs
    
    else:

        return data_arr_out, arr_times, fx



def plot_group_timeFreq(data_arr, freqs, times,
                        TARGET: str,
                        FS=18,
                        SHOW_PRES_SUBS=False,
                        present_subs=False,
                        DATA_VERSION=None,
                        FT_VERSION=None,
                        save_figname=False,
                        source=False,):

    

    fig, ax = plt.subplots(1, 1, figsize=(16, 8),
                        #    gridspec_kw={'height_ratios': h_ratio}
    )
    tf_ax = ax
    

    mean_arr = np.nanmean(data_arr, axis=2)
    # if LOG_POWER:
    # tf_mesh = ax.pcolormesh(np.log(mean_arr.astype(np.float64).T),
    tf_mesh = tf_ax.pcolormesh(mean_arr.astype(np.float64).T,
                vmin=0, vmax=-5,
                cmap='viridis')
    cbar = fig.colorbar(tf_mesh, pad=.1)
    cbar.set_label('Log Spectral Power (a.u.)', size=FS)
    cbar.ax.tick_params(size=FS, labelsize=FS)

    # X axis
    if TARGET == 'LDOPA':
        xlab = 'Time after L-Dopa intake (minutes)'
        xlabels = [-5, 0, 5, 10, 15, 20, 25]
        fig_title = 'Group level spectral data around L-Dopa intake'
    else:
        xlab = 'Time after Dyskinesia-onset (minutes)'
        xlabels = [-10, 0, 10, 20, 30, 40]
        fig_title = 'Group level spectral data around dyskinesia-onset'
    if source: fig_title += f' ({source})'

    tf_ax.set_xlabel(xlab, size=FS,)    
    xticks = [np.argmin(abs(t - times / 60)) for t in xlabels]
    tf_ax.set_xticks(xticks)
    tf_ax.set_xticklabels(xlabels)
    # Y axis
    tf_ax.set_ylabel('Frequency (Hz)', size=FS,)
    ylabels = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    yticks = [f in ylabels for f in freqs]
    yticks = np.arange(len(freqs))[yticks]
    tf_ax.set_yticks(yticks)
    tf_ax.set_yticklabels(ylabels)
    tf_ax.tick_params(axis='both', size=FS, labelsize=FS)

    if fig_title: ax.set_title(fig_title, size=FS,)

    if SHOW_PRES_SUBS:
        sub_ax = ax.twinx()
        sub_ax.plot(present_subs, alpha=.3,
                color='orange', lw=2,)
        sub_ax.set_yticks([0, 4, 8, 12, 16])
        sub_ax.set_ylabel('Included subjects (count)',
                    size=FS, color='orangered',)
        sub_ax.tick_params(axis='both', size=FS, labelsize=FS,
                        labelcolor='orangered',
                        color='orangered',)

    plt.tight_layout()

    if save_figname:
        save_dir = join(utilsFiles.get_project_path('figures'),
                         'ft_exploration',
                         f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                         'group_timeFreqs')
        if not exists(save_dir): makedirs(save_dir)

        plt.savefig(join(save_dir, save_figname),
                    dpi=300, facecolor='w',)
        plt.close()
    else:
        plt.show()