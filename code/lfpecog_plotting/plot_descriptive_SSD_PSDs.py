"""
Plot overview PSDs for Paper
"""

# import public functions
from os.path import join
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib as mpl

# import own functions
from lfpecog_analysis.ft_processing_helpers import (
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS
)
from lfpecog_plotting.plotHelpers import get_colors
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_cdrs_specific, get_ecog_side
)
from utils.utils_fileManagement import get_project_path


def plot_PSD_vs_DopaTime(all_timefreqs,
                         sel_subs=None,
                         STN_or_ECOG='STN',
                         LOG_POWER=True,
                         ZSCORE_FREQS=True,
                         SMOOTH_PLOT_FREQS=0,
                         BASELINE_CORRECT=False,
                         BREAK_X_AX=False,
                         plt_ax_to_return=False,
                         fsize=12,):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean PSDs for temporal course after LDopa-intake.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LOG_POWER: plot powers logarithmic
        - ZSCORE_FREQS: plot PSDs as z-Scores per freq-bin
        - SMOOTH_PLOT_FREQS: if > 0, window (n freqs) to
            smoothen PSDs values
        - BASELINE_CORRECT: plot difference of time-windows
            versus pre-Dopa-intake, baeline-correction performed
            on subject-ephys-source specific level
        - BREAK_X_AX: break x-axis between beta and gamma
        - plt_ax_to_return: if plotted as subplot in another plot,
            give the defined axis here
        - fsize: fontsize, defaults to 12
    """
    assert STN_or_ECOG.upper() in ['STN', 'ECOG'], 'STN_or_ECOG should STN / ECOG'
    timings = np.arange(0, 76, 15)  # create timings between 0 and 61/76 with 10/15 min steps
    psds_to_plot = {t: [] for t in timings}

    # first timing 0, is regharded as < 0
    # last timing (70) is regarded as > 60
    if sel_subs: subs = [sub for sub in all_timefreqs.keys() if sub in sel_subs]
    else: subs = all_timefreqs.keys()
    
    for sub in subs:

        for src in all_timefreqs[sub].keys():

            if STN_or_ECOG.upper() == 'STN':
                if 'ecog' in src: continue
            elif STN_or_ECOG.upper() == 'ECOG':
                if not 'ecog' in src: continue

            # get data for this sub and ephys-source
            tf_values = all_timefreqs[sub][src].values.copy()
            if LOG_POWER: tf_values = np.log(tf_values)
            if ZSCORE_FREQS:
                for f in np.arange(tf_values.shape[0]):
                    tf_values[f] = (tf_values[f] - np.mean(tf_values[f])
                                 ) / np.std(tf_values[f])

            tf_times = all_timefreqs[sub][src].times / 60
            tf_freqs = all_timefreqs[sub][src].freqs

            for timing in timings:

                if timing == 0:
                    sel = tf_times < 0
                    # take first 5 minutes if no pre-LDOPA data
                    if sum(sel) == 0:
                        sel = tf_times < 5
                        # print(f'take first 5 minutes as PRE LT for {sub} {src}')
                elif timing == timings[-1]:    
                    sel = tf_times > timing
                else:
                    sel = np.logical_and(tf_times > timing - 10,
                                        tf_times < timing)
                if sum(sel) == 0: continue

                mean_psd = np.mean(tf_values[:, sel], axis=1)

                if BASELINE_CORRECT and timing == 0:
                    BL = mean_psd
                elif BASELINE_CORRECT:
                    mean_psd = (mean_psd - BL) / BL * 100
        
                psds_to_plot[timing].append(list(mean_psd))

    if not plt_ax_to_return:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = plt_ax_to_return
    
    cmap = mpl.cm.get_cmap('winter')  # 'winter' / 'gist_yarg', 'cool
    gradient_colors = np.linspace(0, 1, len(timings))
    gradient_colors = [cmap(g) for g in gradient_colors]
    
    if BASELINE_CORRECT:
        gradient_colors[1:len(timings)] = [cmap(g) for g in np.linspace(0, 1, len(timings)-1)]
    
    for i, timing in enumerate(timings):

        if BASELINE_CORRECT and timing == 0: continue

        psds = np.array(psds_to_plot[timing])
        if not len(psds.shape) == 2: continue
        psd_mean = np.mean(psds, axis=0)
        # blank freqs irrelevant after SSD
        blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        psd_mean[blank_sel] = np.nan
        # smoothen signal for plot
        if SMOOTH_PLOT_FREQS > 0:
            psd_mean = Series(psd_mean).rolling(
                window=SMOOTH_PLOT_FREQS, center=True
            ).mean().values

        
        # adjust label
        if timing == 0: label = 'pre L-DOPA intake'
        elif timing == timings[-1]: label = f'post {timings[-2]} min'
        else: label = f'{timings[i-1]} - {timing} min'
        
        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
            x_axis = xticks

        if not BREAK_X_AX: x_axis = tf_freqs

        # PLOT LINE
        ax.plot(x_axis, psd_mean, label=label,
                color=gradient_colors[i],
                lw=5, alpha=.5,)

    if BREAK_X_AX:
        ax.set_xticks(xticks[::8], size=fsize)
        ax.set_xticklabels(xlabels[::8], fontsize=fsize)
        if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
        elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
        elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
        # ax.fill_betweenx(y=yfill, x1=i_sel, x2=i_sel+nan_pad,
        #                  facecolor='gray', edgecolor='gray', alpha=.3,)
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
              color='gray', lw=1, alpha=.5,)
    
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Power (a.u.)'
    if BASELINE_CORRECT: ylabel = 'Power %-change vs pre - L-DOPA' + ylabel[5:]
    ax.set_ylabel(ylabel, size=fsize,)

    ax.legend(frameon=False, fontsize=fsize, ncol=2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if plt_ax_to_return: return ax
    else: plt.show()


def find_unilateral_LID_sides(sub):
    sub_cdrs = {s: {} for s in ['left', 'right']}  # CDRS scores related to BODYSIDE
    
    for side in ['left', 'right']:
        _, sub_cdrs[side] = get_cdrs_specific(
            sub=sub, rater='Jeroen', side=side,)
        
    # take only moments with unilateral LID on matching side
    left_sel = np.logical_and(sub_cdrs['left'] > 0, sub_cdrs['right'] == 0)
    right_sel = np.logical_and(sub_cdrs['left'] == 0, sub_cdrs['right'] > 0)
    
    if sum(left_sel) == 0 and sum(right_sel) == 0:
        return False, False, False, False
    
    # get nearest CDRS values for psd-timings
    if sum(left_sel) > 0:
        LID_side = 'left'
        noLID_side = 'right'
        LFP_match_side = 'right'
        LFP_nonmatch_side = 'left'
    if sum(right_sel) > 0:
        LID_side = 'right'
        noLID_side = 'left'
        LFP_match_side = 'left'
        LFP_nonmatch_side = 'right'

    return LID_side, noLID_side, LFP_match_side, LFP_nonmatch_side


def plot_STN_PSD_vs_LID(
    all_timefreqs, sel_subs=None,
    LAT_or_SCALE='LAT',
    LOG_POWER=True, ZSCORE_FREQS=True,
    SMOOTH_PLOT_FREQS=0,
    BASELINE_CORRECT=False,
    BREAK_X_AX=False, STD_ERR=True,
    plt_ax_to_return=False,
    fsize=12,
):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean unilateral PSDs versus unilateral CDRS.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LAT_or_SCALE='LAT', should be LAT (laterality),
            or SCALE (PSDs vs CDRS categories)
        - LOG_POWER: plot powers logarithmic
        - ZSCORE_FREQS: plot PSDs as z-Scores per freq-bin
        - SMOOTH_PLOT_FREQS: if > 0, window (n freqs) to
            smoothen PSDs values
        - BASELINE_CORRECT: plot difference of time-windows
            versus pre-Dopa-intake
        - BREAK_X_AX: break x-axis between beta and gamma
        - plt_ax_to_return: if plotted as subplot in another plot,
            give the defined axis here
        - fsize: fontsize, defaults to 12
    """
    std_dict = {'match': [], 'nonmatch': []}
    tf_values, tf_times, psds_to_plot = std_dict.copy(), std_dict.copy(), std_dict.copy()
    if LAT_or_SCALE == 'SCALE':
        for lab in ['match', 'nonmatch', 'bi']: psds_to_plot[lab] = {}
        cdrs_categs = {}
    
    ft_cdrs = {'LID': [], 'noLID': []}
    
    if sel_subs:
        subs = [sub for sub in all_timefreqs.keys()
                if sub in sel_subs]
    else:
        subs = all_timefreqs.keys()

    n_uni_subs = 0
    for sub in subs:
        # check for presence unilateral LID
        (LID_side, noLID_side,
         LFP_match_side, LFP_nonmatch_side) = find_unilateral_LID_sides(sub)
        
        if LAT_or_SCALE == 'LAT':
            if not LID_side: continue  # no UNILATERAL DYSKINESIA PRESENT
        elif LAT_or_SCALE == 'SCALE':
            # only bilat present, sides dont matter
            if not LID_side:
                (LID_side, noLID_side,
                 LFP_match_side, LFP_nonmatch_side) = ('left', 'right', 'right', 'left')
        if LID_side: n_uni_subs += 1  # INCLUDE SUBJECT WITH UNILATERAL LID IN PLOT

        # get PSD values and process them
        tf_values['match'] = all_timefreqs[sub][f'lfp_{LFP_match_side}'].values
        tf_values['nonmatch'] = all_timefreqs[sub][f'lfp_{LFP_nonmatch_side}'].values
        
        if LOG_POWER or ZSCORE_FREQS:
            for lab in tf_values.keys():
                if LOG_POWER: tf_values[lab] = np.log(tf_values[lab])
                if ZSCORE_FREQS:
                    for i_f in np.arange(tf_values[lab].shape[0]):
                        dat = tf_values[lab][i_f, :]
                        tf_values[lab][i_f, :] = (dat - np.mean(dat)) / np.std(dat)
        
        tf_times['match'] = all_timefreqs[sub][f'lfp_{LFP_match_side}'].times / 60
        tf_times['nonmatch'] = all_timefreqs[sub][f'lfp_{LFP_nonmatch_side}'].times / 60
        tf_freqs = all_timefreqs[sub][f'lfp_{LFP_match_side}'].freqs
        
        # find matching windows (UNILAT LID) for both LFP PSDs
        # matching LFP side to LID
        _, ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=LID_side, ft_times=tf_times['match'],
            cdrs_rater='Jeroen',
        ) 
        _, ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=noLID_side, ft_times=tf_times['match'],
            cdrs_rater='Jeroen',
        )
        if LAT_or_SCALE == 'SCALE':
            _, ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side='both', ft_times=tf_times['match'],
                cdrs_rater='Jeroen',
            ) 
        
        # extract BASELINE and BILAT selections from tf_values, before
        # tf_values itself is adjusted due to the unilat selection
        if BASELINE_CORRECT:
            bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
            tf_values['match_BL'] = tf_values['match'][:, bl_sel]

        if LAT_or_SCALE == 'SCALE':
            bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)       
            tf_values['bi_match'] = tf_values['match'][:, bi_lid_sel]
            cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['bi'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False)
            cdrs_categs['bi_match'] = cdrs_cats[bi_lid_sel]

        # select unilat LID itself, changes tf_values
        uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
        tf_values['match'] = tf_values['match'][:, uni_lid_sel]
        if LAT_or_SCALE == 'SCALE':
            cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False)
            cdrs_categs['match'] = cdrs_cats[uni_lid_sel]

        
        # NON-matching LFP side to LID
        _, ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=LID_side, ft_times=tf_times['nonmatch'],
            cdrs_rater='Jeroen',
        ) 
        _, ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=noLID_side, ft_times=tf_times['nonmatch'],
            cdrs_rater='Jeroen',
        )
        if LAT_or_SCALE == 'SCALE':
             _, ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side='both', ft_times=tf_times['nonmatch'],
                cdrs_rater='Jeroen',
            )
             
        # extract BASELINE and BILAT selections from tf_values, before
        # tf_values itself is adjusted due to the unilat selection
        if BASELINE_CORRECT:
            bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
            tf_values['nonmatch_BL'] = tf_values['nonmatch'][:, bl_sel]
        
        if LAT_or_SCALE == 'SCALE':
            bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)       
            tf_values['bi_nonmatch'] = tf_values['nonmatch'][:, bi_lid_sel]
            cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['bi'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False,
                                         return_coding_dict=True)
            cdrs_categs['bi_nonmatch'] = cdrs_cats[bi_lid_sel]
            
        # select unilat LID itself´, changes tf_values
        uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
        tf_values['nonmatch'] = tf_values['nonmatch'][:, uni_lid_sel]
        if LAT_or_SCALE == 'SCALE':
            cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False,)
            cdrs_categs['nonmatch'] = cdrs_cats[uni_lid_sel]

        # calculate MEAN PSD VALUES (match vs non-match) for LATERALITY
        for match_label in ['match', 'nonmatch']:
            # for all settings, calculate baseline if needed
            if BASELINE_CORRECT:
                bl_psd = np.mean(tf_values[f'{match_label}_BL'], axis=1)
            
            # if LATERALITY, take mean psds per side and subject,
            # and add individual mean to grand plotting
            if LAT_or_SCALE == 'LAT':
                mean_psd = np.mean(tf_values[match_label], axis=1)
                if BASELINE_CORRECT:
                    mean_psd = (mean_psd - bl_psd) / bl_psd * 100
                psds_to_plot[match_label].append(list(mean_psd))
            
            # for SCALING, add mean-psds per category, per subject
            elif LAT_or_SCALE == 'SCALE':
                # correct bilat-psds against the baseline from the corresponding
                # hemisphere, add all bilat psds to one list
                for label in ['match', 'nonmatch', 'bi_match', 'bi_nonmatch']:
                    assert tf_values[label].shape[1] == len(cdrs_categs[label]), (
                        f'tf_values and cdrs_categs "{sub, label}" DOES NOT MATCH:'
                        f' {tf_values[label].shape} vs {len(cdrs_categs[label])}'
                    )
                    dict_lab = label.split('_')[0]  # take match/nonmatch/bi
                
                    for cat in np.unique(cdrs_categs[label]):
                        cat_sel = cdrs_categs[label] == cat
                        cat_values = np.array(tf_values[label])[:, cat_sel]
                        mean_cat_values = np.mean(cat_values, axis=1)
                        if BASELINE_CORRECT:
                            mean_cat_values = (mean_cat_values - bl_psd) / bl_psd * 100
                        # add subject-mean per category (add empty list if necessary)
                        if cat not in psds_to_plot[dict_lab].keys():
                            psds_to_plot[dict_lab][cat] = []
                        psds_to_plot[dict_lab][cat].append(list(mean_cat_values))
                        
                        
    ### PLOTTING PART
    if LAT_or_SCALE == 'LAT':
        plot_unilateral_LID(plt_ax_to_return=plt_ax_to_return,
                            datatype='STN',
                            psds_to_plot=psds_to_plot,
                            tf_freqs=tf_freqs,
                            n_uni_subs=n_uni_subs,
                            BASELINE_CORRECT=BASELINE_CORRECT,
                            LOG_POWER=LOG_POWER,
                            SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                            STD_ERR=STD_ERR,
                            BREAK_X_AX=BREAK_X_AX,
                            fsize=fsize,)
    
    
    elif LAT_or_SCALE == 'SCALE':
        plot_scaling_LID(plt_ax_to_return=plt_ax_to_return,
                            psds_to_plot=psds_to_plot,
                            datatype='STN',
                            tf_freqs=tf_freqs,
                            cdrs_cat_coding=coding_dict,
                            BASELINE_CORRECT=BASELINE_CORRECT,
                            LOG_POWER=LOG_POWER,
                            SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                            STD_ERR=STD_ERR,
                            BREAK_X_AX=BREAK_X_AX,
                            fsize=fsize,)       
    

def plot_scaling_LID(
    psds_to_plot, tf_freqs,
    cdrs_cat_coding, datatype,
    plt_ax_to_return=False,
    BASELINE_CORRECT=True, LOG_POWER=False,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True,
    BREAK_X_AX=True, fsize=14,
    SAVE_PLOT=True, SHOW_PLOT=False,
    fig_name='PSD_CDRS_scaling_STN_n11',
):
    assert datatype.upper() in ['STN', 'ECOG'], (
        f'datatype ({datatype}) should be STN or ECOG'
    )

    if plt_ax_to_return == False:
        fig, axes = plt.subplots(1, len(psds_to_plot),
                                 figsize=(len(psds_to_plot) * 6, 6))
    else:
        axes = plt_ax_to_return
    
    colors = [list(get_colors().values())[c] for c in [4, 3, 0, 7]]  # colors are called by cat-value, so 0 is never used if baseline corrected

    for i_ax, side in enumerate(psds_to_plot.keys()):
        if side == 'bi': ax_title = f'{datatype} during bilateral dyskinesia'
        elif side == 'match': ax_title = f'{datatype} during only contralateral dyskinesia'
        elif side == 'nonmatch': ax_title = f'{datatype} during only ipsilateral dyskinesia'

        for i_cat, cat in enumerate(psds_to_plot[side].keys()):
            PSD = {}
            psds = np.array(psds_to_plot[side][cat])
            PSD['mean'] = np.mean(psds, axis=0)
            PSD['sd'] = np.std(psds, axis=0)
            if STD_ERR: PSD['sd'] = PSD['sd'] / np.sqrt(psds.shape[0])
            # blank freqs irrelevant after SSD
            blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
            for k in PSD: PSD[k][blank_sel] = np.nan
            # smoothen signal for plot (both mean and stddev)
            if SMOOTH_PLOT_FREQS > 0:
                for k in PSD:
                    # PSD[k] = Series(PSD[k]).rolling(window=SMOOTH_PLOT_FREQS,
                    #                                 center=True).mean().values
                    h_winlen = int(SMOOTH_PLOT_FREQS / 2)
                    new_arr = np.array([np.nan] * len(PSD[k]))
                    for win_mid in np.arange(h_winlen, len(PSD[k]) - h_winlen):
                        new_arr[win_mid] = np.nanmean(PSD[k][win_mid-h_winlen: win_mid+h_winlen])
                    PSD[k] = new_arr
            # n-subjects to add to legend-label
            n_subs_cat = psds.shape[0]
                    
            # BREAK X AXIS and adjust xticks and labels
            if BREAK_X_AX:
                PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
                x_axis = xticks

            if not BREAK_X_AX: x_axis = tf_freqs

            # PLOT LINE
            axes[i_ax].plot(x_axis, PSD['mean'], lw=5, alpha=.5,
                            label=f'{list(cdrs_cat_coding.keys())[int(cat)]} dyskinesia'
                            f' (n={n_subs_cat})',
                            color=colors[int(cat)], )
            # PLOT VARIANCE SHADING
            axes[i_ax].fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                    y2=PSD['mean'] + PSD['sd'],
                                    alpha=.25, color=colors[int(cat)],)
        
        if BREAK_X_AX:
            axes[i_ax].set_xticks(xticks[::8], size=fsize)
            axes[i_ax].set_xticklabels(xlabels[::8], fontsize=fsize)
            # if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
            # elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
            # elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
            # if BASELINE_CORRECT: yfill = [-30, 40]
            # ax.fill_betweenx(y=yfill, x1=i_sel, x2=i_sel+nan_pad,
            #                  facecolor='gray', edgecolor='gray', alpha=.2,)
        else:
            axes[i_ax].set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
            axes[i_ax].set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

        if not LOG_POWER: axes[i_ax].hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
                                    color='gray', lw=1, alpha=.5,)
        
        axes[i_ax].set_title(ax_title, size=fsize, weight='bold',)
        axes[i_ax].set_xlabel('Frequency (Hz)', size=fsize,)
        ylabel = 'Power (a.u.)'
        if LOG_POWER: ylabel = f'Log. {ylabel}'
        if BASELINE_CORRECT: ylabel = f'{ylabel[:-6]} %-change vs bilat-no-LID (a.u.)'
        axes[i_ax].set_ylabel(ylabel, size=fsize,)
        
        axes[i_ax].legend(frameon=False, fontsize=fsize, loc='upper left')
        
        axes[i_ax].spines['top'].set_visible(False)
        axes[i_ax].spines['right'].set_visible(False)
        axes[i_ax].tick_params(axis='both', size=fsize, labelsize=fsize)
    
    
    if plt_ax_to_return != False:
        return axes

    else:
        # plot or save axes from here
        # equalize axes
        ymin = min([min(ax.get_ylim()) for ax in axes])
        ymax = max([max(ax.get_ylim()) for ax in axes])
        for ax in axes: ax.set_ylim(ymin, ymax)

        for ax in axes: ax.tick_params(axis='both', size=fsize, labelsize=fsize)
        plt.tight_layout()

        if SAVE_PLOT:
            DATA_VERSION = 'v4.0'
            plt.savefig(join(get_project_path('figures'), 'ft_exploration',
                                DATA_VERSION, 'descr_PSDs', fig_name),
                        facecolor='w', dpi=300,)

        if SHOW_PLOT: plt.show()
        else: plt.close()



def plot_unilateral_LID(
    psds_to_plot, tf_freqs, n_uni_subs,
    datatype, plt_ax_to_return=None,
    BASELINE_CORRECT=True, LOG_POWER=False,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True, BREAK_X_AX=True,
    fsize=14,
):
    assert datatype.upper() in ['STN', 'ECOG'], (
        f'datatype ({datatype}) should be STN or ECOG'
    )

    if plt_ax_to_return == None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = plt_ax_to_return      
    
    colors = [list(get_colors().values())[3],
              list(get_colors().values())[7]]
    
    leg_label = {'match': f'{datatype} contralateral to Dyskinesia',
                 'nonmatch': f'{datatype} ipsilateral to Dyskinesia'}
    
    for i, match_label in enumerate(psds_to_plot.keys()):
        PSD = {}
        psds = np.array(psds_to_plot[match_label])
        PSD['mean'] = np.mean(psds, axis=0)
        PSD['sd'] = np.std(psds, axis=0)
        if STD_ERR: PSD['sd'] = PSD['sd'] / np.sqrt(psds.shape[0])
        # blank freqs irrelevant after SSD
        blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        for k in PSD: PSD[k][blank_sel] = np.nan
        # smoothen signal for plot
        if SMOOTH_PLOT_FREQS > 0:
            for k in PSD:
                PSD[k] = Series(PSD[k]).rolling(window=SMOOTH_PLOT_FREQS,
                                                   center=True).mean().values
        # add n-subjects to leg_label
        leg_label[match_label] += f'(n={psds.shape[0]})'
                
        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
            x_axis = xticks

        if not BREAK_X_AX: x_axis = tf_freqs

        # PLOT LINE
        ax.plot(x_axis, PSD['mean'], label=leg_label[match_label],
                color=colors[i], lw=5, alpha=.5,)
        # PLOT VARIANCE SHADING
        if i == 0: ax.fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                    y2=PSD['mean'] + PSD['sd'],
                                    alpha=.4, color=colors[i],)
        if i == 1: ax.fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                    y2=PSD['mean'] + PSD['sd'],
                                    alpha=.4, edgecolor=colors[i],
                                    facecolor='w', hatch='//',)

    if BREAK_X_AX:
        ax.set_xticks(xticks[::8], size=fsize)
        ax.set_xticklabels(xlabels[::8], fontsize=fsize)
        if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
        elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
        elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
        if BASELINE_CORRECT: yfill = [-30, 40]
        # ax.fill_betweenx(y=yfill, x1=i_sel, x2=i_sel+nan_pad,
        #                  facecolor='gray', edgecolor='gray', alpha=.2,)
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    if not LOG_POWER: ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
                                color='gray', lw=1, alpha=.5,)
    
    ax.set_title(f'Subjects with unilateral Dyskinesia (n={n_uni_subs})',
                  size=fsize, weight='bold',)
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Power (a.u.)'
    if LOG_POWER: ylabel = f'Log. {ylabel}'
    if BASELINE_CORRECT: ylabel = f'{ylabel[:-6]} %-change vs bilat-no-LID (a.u.)'
    ax.set_ylabel(ylabel, size=fsize,)
    
    ax.legend(frameon=False, fontsize=fsize, loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)
    
    plt.tight_layout()

    if plt_ax_to_return: return ax
    else: plt.show()


def break_x_axis_psds_ticks(tf_freqs, PSD,
                            x_break = (35, 60), nan_pad = 5):
    
    del_sel = np.logical_and(tf_freqs > x_break[0],
                             tf_freqs < x_break[1])
    del_sel = np.logical_or(del_sel, np.isnan(PSD['mean']))
    PSD['mean'] = np.delete(PSD['mean'], del_sel,)
    PSD['sd'] = np.delete(PSD['sd'], del_sel,)
    plt_freqs = np.delete(tf_freqs.copy(), del_sel,).astype(float)

    i_sel = np.argmin(abs(plt_freqs - x_break[0]))

    PSD['mean'] = np.insert(PSD['mean'], i_sel + 1,
                            values=[np.nan] * nan_pad,)
    PSD['sd'] = np.insert(PSD['sd'], i_sel + 1,
                            values=[np.nan] * nan_pad,)
    plt_freqs = np.insert(plt_freqs, i_sel + 1,
                            values=[np.nan] * nan_pad,)

    xticks = np.arange(len(PSD['mean']))
    xlabels = [''] * len(xticks)
    low_ticks = plt_freqs[plt_freqs < x_break[0]]
    xlabels[:len(low_ticks)] = low_ticks
    high_ticks = plt_freqs[plt_freqs > x_break[1]]
    xlabels[len(xlabels) - len(high_ticks):] = high_ticks

    return PSD, xticks, xlabels


def plot_ECOG_PSD_vs_LID(
    all_timefreqs, sel_subs=None,
    LAT_or_SCALE='LAT',
    LOG_POWER=True, ZSCORE_FREQS=True,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True,
    BASELINE_CORRECT=False,
    BREAK_X_AX=False, plt_ax_to_return=False,
    fsize=12, fig_name=None,
):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean unilateral PSDs versus unilateral CDRS.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LAT_or_SCALE='LAT', should be LAT (laterality),
            or SCALE (PSDs vs CDRS categories)
        - LOG_POWER: plot powers logarithmic
        - ZSCORE_FREQS: plot PSDs as z-Scores per freq-bin
        - SMOOTH_PLOT_FREQS: if > 0, window (n freqs) to
            smoothen PSDs values
        - BASELINE_CORRECT: plot difference of time-windows
            versus pre-Dopa-intake
        - BREAK_X_AX: break x-axis between beta and gamma
        - plt_ax_to_return: if plotted as subplot in another plot,
            give the defined axis here
        - fsize: fontsize, defaults to 12
    """
    # tf_values, tf_times = {'match': [], 'nonmatch': []}, {'match': [], 'nonmatch': []}
    if LAT_or_SCALE == 'LAT':
        psds_to_plot = {'match': [], 'nonmatch': []}
    elif LAT_or_SCALE == 'SCALE':
        cdrs_categs, psds_to_plot = {}, {}
        for lab in ['match', 'nonmatch', 'bi']: psds_to_plot[lab] = {}

    ft_cdrs = {'LID': [], 'noLID': []}

    if sel_subs:
        subs = [sub for sub in all_timefreqs.keys()
                if sub in sel_subs]
    else:
        subs = all_timefreqs.keys()

    n_uni_subs = 0
    for sub in subs:
        if sub.startswith('1'): continue  # SKIP NON-ECOG subjects
        # check for presence unilateral LID
        (LID_side, noLID_side,
         LFP_match_side, LFP_nonmatch_side) = find_unilateral_LID_sides(sub)
        ecog_side = get_ecog_side(sub)

        if LAT_or_SCALE == 'LAT':
            if not LID_side: continue  # no UNILATERAL DYSKINESIA PRESENT
        elif LAT_or_SCALE == 'SCALE':
            # only bilat present, sides dont matter
            if not LID_side:
                (LID_side, noLID_side,
                 LFP_match_side, LFP_nonmatch_side) = ('left', 'right', 'right', 'left')
        if LID_side:
            n_uni_subs += 1  # INCLUDE SUBJECT WITH UNILATERAL LID IN PLOT
            # define LATERALITY OF LID VERSUS ECOG (in case of unilat lid)
            if ecog_side == LFP_match_side: match_label_uni = 'match'
            elif ecog_side == LFP_nonmatch_side: match_label_uni = 'nonmatch'

        # get PSD values and process them
        tf_values = all_timefreqs[sub][f'ecog_{ecog_side}'].values
        
        if LOG_POWER: tf_values = np.log(tf_values)
        if ZSCORE_FREQS:
                for f in np.arange(tf_values.shape[0]):
                    tf_values[f] = (
                        tf_values[f] - np.mean(tf_values[f])
                    ) / np.std(tf_values[f])
        tf_times = all_timefreqs[sub][f'ecog_{ecog_side}'].times / 60
        tf_freqs = all_timefreqs[sub][f'ecog_{ecog_side}'].freqs

        # find matching windows (UNILAT LID) for both ECoG PSDs
        # matching LFP side to LID
        _, ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=LID_side, ft_times=tf_times,
            cdrs_rater='Jeroen',
        ) 
        _, ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=noLID_side, ft_times=tf_times,
            cdrs_rater='Jeroen',
        )
        if LAT_or_SCALE == 'SCALE':
            _, ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side='both', ft_times=tf_times,
                cdrs_rater='Jeroen',
            ) 

        # extract BASELINE and BILAT selections from tf_values, before
        # tf_values itself is adjusted due to the unilat selection
        if BASELINE_CORRECT:
            bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
            tf_values_BL = tf_values[:, bl_sel]
            bl_psd = np.mean(tf_values_BL, axis=1)
        
        if LAT_or_SCALE == 'SCALE':
            bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)       
            tf_values_bi = tf_values[:, bi_lid_sel]
            cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['bi'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False)
            cdrs_categs['bi'] = cdrs_cats[bi_lid_sel]
        
        # select unilat LID itself, changes tf_values
        uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
        tf_values = tf_values[:, uni_lid_sel]
        
        if LAT_or_SCALE == 'SCALE':
            cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                         preLID_minutes=0,
                                         preLID_separate=False,
                                         convert_inBetween_zeros=False,
                                         return_coding_dict=True)
            cdrs_categs[match_label_uni] = cdrs_cats[uni_lid_sel]

        # if LATERALITY, take mean psds per side and subject,
        # and add individual mean to grand plotting
        if LAT_or_SCALE == 'LAT':
            mean_psd = np.mean(tf_values, axis=1)
            if BASELINE_CORRECT:
                mean_psd = (mean_psd - bl_psd) / bl_psd * 100
            psds_to_plot[match_label_uni].append(list(mean_psd))
        
        # for SCALING, add mean-psds per category, per subject
        elif LAT_or_SCALE == 'SCALE':
            # correct bilat-psds against the baseline from the corresponding
            # hemisphere, add all bilat psds to one list
            assert tf_values.shape[1] == len(cdrs_categs[match_label_uni]), (
                    f'NO MATCH unilat tf_values and cdrs_categs "{sub}" '
                    f' {tf_values.shape} vs {len(cdrs_categs[match_label_uni])}'
            )
            assert tf_values_bi.shape[1] == len(cdrs_categs['bi']), (
                    f'NO MATCH bilat tf_values and cdrs_categs "{sub}" '
                    f' {tf_values_bi.shape} vs {len(cdrs_categs["bi"])}'
            )
            for label in [match_label_uni, 'bi']:
                for cat in np.unique(cdrs_categs[label]):
                    cat_sel = cdrs_categs[label] == cat
                    if label != 'bi':
                        cat_values = np.array(tf_values)[:, cat_sel]
                    elif label == 'bi':
                        cat_values = np.array(tf_values_bi)[:, cat_sel]
                    mean_cat_values = np.mean(cat_values, axis=1)
                    if BASELINE_CORRECT:
                        mean_cat_values = (mean_cat_values - bl_psd) / bl_psd * 100
                    # add subject-mean per category (add empty list if necessary)
                    if cat not in psds_to_plot[label].keys():
                        psds_to_plot[label][cat] = []
                    psds_to_plot[label][cat].append(list(mean_cat_values))
            
    ### PLOTTING PART
    if LAT_or_SCALE == 'SCALE':
        plot_scaling_LID(plt_ax_to_return=plt_ax_to_return,
                         datatype='ECoG',
                         psds_to_plot=psds_to_plot,
                         tf_freqs=tf_freqs,
                         cdrs_cat_coding=coding_dict,
                         BASELINE_CORRECT=BASELINE_CORRECT,
                         LOG_POWER=LOG_POWER,
                         SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                         STD_ERR=STD_ERR,
                         BREAK_X_AX=BREAK_X_AX,
                         fsize=fsize,
                         fig_name=fig_name,)
        
        return 'plotted/saved in script'

    elif LAT_or_SCALE == 'LAT':
        print('TODO: PUT LATERALITY PLOTTING ECOG IN PY function')

    if not plt_ax_to_return:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = plt_ax_to_return      
    
    colors = [list(get_colors().values())[3],
              list(get_colors().values())[7]]
    
    for i, match_label in enumerate(psds_to_plot.keys()):
        PSD = {}
        psds = np.array(psds_to_plot[match_label])
        PSD['mean'] = np.mean(psds, axis=0)
        PSD['sd'] = np.std(psds, axis=0)
        if STD_ERR: PSD['sd'] = PSD['sd'] / np.sqrt(psds.shape[0])
        # blank freqs irrelevant after SSD
        blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        for k in PSD: PSD[k][blank_sel] = np.nan
        # smoothen signal for plot
        if SMOOTH_PLOT_FREQS > 0:
            for k in PSD:
                PSD[k] = Series(PSD[k]).rolling(window=SMOOTH_PLOT_FREQS,
                                                   center=True).mean().values

                
        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
            x_axis = xticks

        if not BREAK_X_AX: x_axis = tf_freqs

        # PLOT LINE
        if match_label == 'match':
            label = f'ECoG contralateral to Dyskinesia (n={psds.shape[0]})'
        elif match_label == 'nonmatch':
            label = f'ECoG ipsilateral to Dyskinesia (n={psds.shape[0]})'
        
        ax.plot(x_axis, PSD['mean'], label=label,
                color=colors[i], lw=5, alpha=.5,)
        # PLOT VARIANCE SHADING
        if i == 0: ax.fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                    y2=PSD['mean'] + PSD['sd'],
                                    alpha=.4, color=colors[i],)
        if i == 1: ax.fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                    y2=PSD['mean'] + PSD['sd'],
                                    alpha=.4, edgecolor=colors[i],
                                    facecolor='w', hatch='//',)

    if BREAK_X_AX:
        ax.set_xticks(xticks[::8], size=fsize)
        ax.set_xticklabels(xlabels[::8], fontsize=fsize)
        if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
        elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
        elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
        if BASELINE_CORRECT: yfill = [-20, 100]
        # ax.fill_betweenx(y=yfill, x1=i_sel, x2=i_sel+nan_pad,
        #                  facecolor='gray', edgecolor='gray', alpha=.2,)
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    if not LOG_POWER: ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
                                color='gray', lw=1, alpha=.5,)
    
    ax.set_title(f'ECoG-subjects with unilateral Dyskinesia (n={n_uni_subs})',
                  size=fsize, weight='bold',)
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Power (a.u.)'
    if LOG_POWER: ylabel = f'Log. {ylabel}'
    if BASELINE_CORRECT: ylabel = f'{ylabel[:-6]} %-change vs bilat-no-LID (a.u.)'
    ax.set_ylabel(ylabel, size=fsize,)

    ax.legend(frameon=False, fontsize=fsize, loc='upper left'
            #   ncol=2,
              )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)
    plt.tight_layout()

    if plt_ax_to_return: return ax
    else: plt.show()