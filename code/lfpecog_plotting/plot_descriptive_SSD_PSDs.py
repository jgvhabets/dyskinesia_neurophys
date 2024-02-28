"""
Plot overview PSDs for Paper
"""

# import public functions
from os.path import join, exists
from os import makedirs
import numpy as np
from pandas import Series, isna
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import compress

# import own functions
from lfpecog_analysis.ft_processing_helpers import (
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS, FeatLidClass,
    split_lid_subs
)
from lfpecog_plotting.plotHelpers import get_colors
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_cdrs_specific, get_ecog_side
)
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle)
from lfpecog_plotting.plotHelpers import remove_duplicate_legend
from lfpecog_analysis.psd_lid_stats import (
    process_mean_stats,
    # get_binary_p_perHz
)
from lfpecog_analysis.get_acc_task_derivs import(
    select_tf_on_movement
)

def plot_PSD_vs_DopaTime(all_timefreqs,
                         sel_subs=None,
                         SOURCE='STN',
                         CONN_METRIC=False,
                         LOG_POWER=True,
                         ZSCORE_FREQS=True,
                         SMOOTH_PLOT_FREQS=0,
                         BASELINE_CORRECT=False,
                         BREAK_X_AX=False,
                         plt_ax_to_return=False,
                         fsize=12, ax_title='PSD over Time'):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean PSDs for temporal course after LDopa-intake.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - SOURCE: ECOG, STN, CONN
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
    allowed_sources = ['STN', 'ECOG', 'CONN_IPSI', 'CONN_CONTRA']
    assert SOURCE.upper() in allowed_sources, (
        f'SOURCE should be in {allowed_sources}'
    )
    timings = np.arange(0, 76, 15)  # create timings between 0 and 61/76 with 10/15 min steps
    psds_to_plot = {t: [] for t in timings}

    # first timing 0, is regharded as < 0
    # last timing (70) is regarded as > 60
    if sel_subs: subs = [sub for sub in all_timefreqs.keys() if sub in sel_subs]
    else: subs = all_timefreqs.keys()
    subs_incl = []

    if 'CONN' in SOURCE: YLAB = CONN_METRIC
    else: YLAB = 'Power'

    for sub in subs:

        for src in all_timefreqs[sub].keys():

            if SOURCE.upper() == 'STN' and not 'lfp' in src: continue
            elif SOURCE.upper() == 'ECOG' and not 'ecog' in src: continue
            elif SOURCE.upper() == 'CONN_CONTRA' and src != 'contralateral': continue
            elif SOURCE.upper() == 'CONN_IPSI' and src != 'ipsilateral': continue
            # print(f'continue with {sub}, {src} (in {SOURCE})')

            subs_incl.append(sub)  # add subject to count

            # get data for this sub and ephys-source
            tf_values = all_timefreqs[sub][src].values.copy()
            if tf_values.shape[0] > tf_values.shape[1]: tf_values = tf_values.T

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
                    # take only rest-PreDopa for sub-010
                    if sub == '010': sel = tf_times < -14
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

                mean_psd = np.nanmean(tf_values[:, sel], axis=1)

                if BASELINE_CORRECT and timing == 0:
                    BL = mean_psd
                elif BASELINE_CORRECT:
                    mean_psd = (mean_psd - BL) / BL * 100
        
                psds_to_plot[timing].append(list(mean_psd))

    n_subs_incl = len(np.unique(subs_incl))

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
        # print(f'{SOURCE}, t={timing} ({i}) is n={psds.shape[0]}')
        if not len(psds.shape) == 2: continue
        psd_mean = np.nanmean(psds, axis=0)
        if np.isnan(psd_mean).all():
            print(f'timing {timing} skipped, only NaNs')
            continue
        # blank freqs irrelevant after SSD
        # blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        # psd_mean[blank_sel] = np.nan
        # tf_freqs[blank_sel] = np.nan

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
            psd_mean, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, psd_mean)
            x_axis = xticks

        if not BREAK_X_AX: x_axis = tf_freqs

        # PLOT LINE
        ax.plot(x_axis, psd_mean, label=f'{label} (n={psds.shape[0]})',
                color=gradient_colors[i],
                lw=5, alpha=.5,)

    if BREAK_X_AX:
        ax.set_xticks(xticks[::8],)
        ax.set_xticklabels(xlabels[::8])

        if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
        elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
        elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
        
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
              color='gray', lw=1, alpha=.5,)
    
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = f'{YLAB} (a.u.)'
    if BASELINE_CORRECT: ylabel = f'{YLAB} %-change vs pre - L-DOPA {ylabel[len(YLAB):]}'
    ax.set_ylabel(ylabel, size=fsize,)

    if 'CONN' in SOURCE: leg_loc = 'upper left'
    else: leg_loc = 'lower right'
    ax.legend(frameon=False, fontsize=fsize-3, ncol=1, loc=leg_loc)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)
    
    ax.set_title(f"{ax_title} (n={n_subs_incl} pt's)",
                        size=fsize, weight='bold',)

    if plt_ax_to_return: return ax
    else: plt.show()


def find_unilateral_LID_sides(sub, CDRS_RATER='Patricia'):
    sub_cdrs = {s: {} for s in ['left', 'right']}  # CDRS scores related to BODYSIDE
    
    for side in ['left', 'right']:
        _, sub_cdrs[side] = get_cdrs_specific(
            sub=sub, rater=CDRS_RATER, side=side,)
        
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
    CDRS_RATER='Patricia',
    LAT_or_SCALE='LAT_UNI',
    incl_PRE_LID=False,  # currently coded as category 0.5
    PRELID_MIN=10,  # in minutes
    LOG_POWER=True, ZSCORE_FREQS=True,
    SMOOTH_PLOT_FREQS=0,
    BASELINE_CORRECT=False,
    SELECT_ON_ACC_RMS=False,
    BREAK_X_AX=False, STD_ERR=True,
    plt_ax_to_return=False,
    fsize=12,
    fig_name='PLOT_STN_PSD_vs_DYSK',
    RETURN_FREQ_CORR=False,
    SINGLE_SUB_LINES=False,
    PLOT_ONLY_MATCH=False,
    SHOW_ONLY_GAMMA=False,
    SHOW_SIGN=False,
    p_SAVED_DATE='0000',
    PROCESS_STATS=False,
    DATA_VERSION='v4.0',
    FT_VERSION='v4',
):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean unilateral PSDs versus unilateral CDRS.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LAT_or_SCALE='LAT_UNI', should be LAT_UNI (laterality of
            unilateral LID), 'LAT_BILAT' (laterality during bilateral
            LID), or SCALE (PSDs vs CDRS categories),
            LAT_ALL_SCALE: merge uni- and bilateral LID, show scaling
        - LOG_POWER: plot powers logarithmic
        - ZSCORE_FREQS: plot PSDs as z-Scores per freq-bin
        - SMOOTH_PLOT_FREQS: if > 0, window (n freqs) to
            smoothen PSDs values
        - BASELINE_CORRECT: plot difference of time-windows
            versus pre-Dopa-intake
        - SELECT_ON_ACC_RMS: defaults to False, no movement selection
            applied; if 'INCL_MOV' or 'EXCL_MOV', movement threshold
            is 0 within z-scored averaged RMS of both ACCs.
        - BREAK_X_AX: break x-axis between beta and gamma
        - plt_ax_to_return: if plotted as subplot in another plot,
            give the defined axis here
        - fsize: fontsize, defaults to 12
        - fig_name
        - CALC_FREQ_CORR: collect full spectrum data and full-CDRS
            scores to collect Correlations per Freq-bin
    """
    print(f'start plotting {fig_name}')
    std_dict = {'match': [], 'nonmatch': []}
    tf_values, tf_times, psds_to_plot = std_dict.copy(), std_dict.copy(), std_dict.copy()
    if LAT_or_SCALE == 'SCALE':
        for lab in ['match', 'nonmatch', 'bi']: psds_to_plot[lab] = {}
        cdrs_categs = {}
    elif LAT_or_SCALE =='LAT_BILAT':
        psds_to_plot, cdrs_categs, cdrs_full = {}, {}, {}  # start with empty dict, dont use match/nonmatch
        for lab in ['bi_match', 'bi_nonmatch']: psds_to_plot[lab] = {}
    elif LAT_or_SCALE =='LAT_ALL_SCALE':
        psds_to_plot, cdrs_categs, cdrs_full = {}, {}, {}  # start with empty dict, dont use match/nonmatch
        for lab in ['match', 'nonmatch', 'bi',
                    'bi_match', 'bi_nonmatch']: psds_to_plot[lab] = {}

    if LAT_or_SCALE == 'LAT_ALL_SCALE':
        n_reps = 2
        PLOT_ONLY_MATCH=True
    else: n_reps = 1

    ft_cdrs = {'LID': [], 'noLID': []}
    if RETURN_FREQ_CORR: FREQ_CORRs = {}

    subs, none_lid_subs = split_lid_subs(DATA_VERSION=DATA_VERSION,
                                             FT_VERSION=FT_VERSION)

    n_uni_subs = 0
    subs_incl = {k: {} for k in psds_to_plot.keys()}  # keep track of indiv subs per cat
    
    # if movement selection on RMS is defined, load pickled class for all subjects
    if SELECT_ON_ACC_RMS:
        featLabPath = join(get_project_path('data'),
                           'prediction_data',
                           'featLabelClasses',
                           f'featLabels_ft{FT_VERSION}_Cdrs_StnOnly.P')
        feat10sClass = load_class_pickle(featLabPath,
                                         convert_float_np64=True)
        print(f'10s-feature class loaded for RMS-ACC selection ({featLabPath})')

    # gets per hemisphere the subject ID and the mean-psd-array (no LID and LID)
    # CDRS contains the scores corresponding to LID
    mean_stats = {'LID': [], 'noLID': [], 'CDRS': []}
    
    for sub in subs:

        stats_lid_pow = {'match': [], 'nonmatch': []}
        stats_lid_cdrs = {'match': [], 'nonmatch': []}
        # enable combining of unilateral and bilateral dyskinesia for 'LAT_ALL_SCALE'
        for i_rep in np.arange(n_reps):
            
            # if 'LAT_ALL_SCALE' is defined, first run uni, than bilat
            if n_reps == 2 and i_rep == 0: LAT_or_SCALE = 'SCALE'
            elif n_reps == 2 and i_rep == 1: LAT_or_SCALE = 'LAT_BILAT'

            # check for presence unilateral LID
            (LID_side, noLID_side, LFP_match_side,
             LFP_nonmatch_side) = find_unilateral_LID_sides(sub, CDRS_RATER=CDRS_RATER)
            
            if LAT_or_SCALE == 'LAT_UNI':
                if not LID_side: continue  # skip subject if no UNILATERAL DYSKINESIA PRESENT
                elif LID_side: n_uni_subs += 1  # INCLUDE SUBJECT WITH UNILATERAL LID IN PLOT

            else:
                # sides dont matter for LAT_BILAT or SCALE (both sides are included)
                if not LID_side:
                    (LID_side, noLID_side,
                    LFP_match_side, LFP_nonmatch_side) = ('left', 'right',
                                                        'right', 'left')
            
            # get STN PSD values and process them
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

            
            ### SELECT OUT ON ACC RMS IF DEFINED HERE ###
            if SELECT_ON_ACC_RMS:
                for match_key in ['match', 'nonmatch']:
                    (tf_values[match_key],
                     tf_times[match_key]) = select_tf_on_movement(
                        feat10sClass=feat10sClass, sub=sub,
                        tf_values_arr=tf_values[match_key],
                        tf_times_arr=tf_times[match_key],
                        SELECT_ON_ACC_RMS=SELECT_ON_ACC_RMS,
                        # RMS_Z_THRESH=0,  # defaults to -0.5
                    )
            

            # find matching windows (UNILAT LID) for both LFP PSDs
            # matching LFP side to LID
            ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=LID_side, ft_times=tf_times['match'],
                cdrs_rater=CDRS_RATER,
            ) 
            ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=noLID_side, ft_times=tf_times['match'],
                cdrs_rater=CDRS_RATER,
            )
            if LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                    sub=sub, side='both', ft_times=tf_times['match'],
                    cdrs_rater=CDRS_RATER,
                )
                if incl_PRE_LID:
                    start_lid_i = np.where(ft_cdrs['bi'] > 0)[0][0]
                    start_lid_t = tf_times['match'][start_lid_i]
                    start_prelid_t = start_lid_t - PRELID_MIN
                    prelid_sel = np.logical_and(tf_times['match'] > start_prelid_t,
                                                tf_times['match'] < start_lid_t)
                    # select ft values for pre LID
                    tf_values['match_preLID'] = tf_values['match'][:, prelid_sel]
                    cdrs_categs['match_preLID'] = [0.5] * len(tf_values['match_preLID'])


            
            # extract BASELINE and BILAT selections from tf_values, before
            # tf_values itself is adjusted due to the unilat selection
            if BASELINE_CORRECT:
                bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
                # # excl pre LID from BASELINE
                if incl_PRE_LID and start_prelid_t > 5:
                    bl_sel = bl_sel.astype(int) - prelid_sel.astype(int)
                    bl_sel[bl_sel == -1] = 0
                    bl_sel = bl_sel.astype(bool)

                # select baseline
                tf_values['match_BL'] = tf_values['match'][:, bl_sel]

                if i_rep == 0:  # add tuple 
                    mean_stats['noLID'].append((f'{sub}_match', tf_values['match_BL']))

            if LAT_or_SCALE == 'SCALE':
                # select lateral spectral values for both sides (now match)
                bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)       
                # add match-lfp-side with MATCHING body side
                tf_values['bi_match'] = tf_values['match'][:, bi_lid_sel]
                cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['bi'],  # was bi, should be match-LID
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False)
                cdrs_categs['bi_match'] = cdrs_cats[bi_lid_sel]

                # add match STN during bilateral LID for STATS
                stats_lid_pow['match'].append(tf_values[f'bi_match']) 
                stats_lid_cdrs['match'].append(ft_cdrs['LID'][bi_lid_sel])  # add cdrs values for body side corr to powers


            elif LAT_or_SCALE == 'LAT_BILAT':
                # select lateral spectral values for both sides (now match)
                bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)
                if sum(bi_lid_sel) == 0:
                    print(f'skip sub-{sub} during {LAT_or_SCALE}, no BILATERAL LID')
                    continue     
                # add match-lfp-side with MATCHING body side
                tf_values['bi_match1'] = tf_values['match'][:, bi_lid_sel]
                cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],  # was bi, should be match-LID
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False)
                cdrs_categs['bi_match1'] = cdrs_cats[bi_lid_sel]
                
                if RETURN_FREQ_CORR: cdrs_full['bi_match1'] = ft_cdrs['LID'][bi_lid_sel]
                
                # add match-lfp-side with NONMATCHING body side (is new)
                tf_values['bi_nonmatch1'] = tf_values['match'][:, bi_lid_sel]
                cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['noLID'],  # was bi, should be match-LID
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False)
                cdrs_categs['bi_nonmatch1'] = cdrs_cats[bi_lid_sel]

                if RETURN_FREQ_CORR: cdrs_full['bi_nonmatch1'] = ft_cdrs['noLID'][bi_lid_sel]

            # select unilat LID itself, changes tf_values
            if LAT_or_SCALE in ['LAT_UNI', 'SCALE']:
                uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
                tf_values['match'] = tf_values['match'][:, uni_lid_sel]
                # get matching cdrs scores
                if LAT_or_SCALE == 'SCALE':
                    cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                                preLID_minutes=0,
                                                preLID_separate=False,
                                                convert_inBetween_zeros=False)
                    cdrs_categs['match'] = cdrs_cats[uni_lid_sel]                        
                
                # add matching STN during unilateral LID for STATS
                stats_lid_pow['match'].append(tf_values['match'])
                stats_lid_cdrs['match'].append(ft_cdrs['LID'][uni_lid_sel])  # add cdrs values for body side corr to powers
            
            
            # NON-matching LFP side to LID
            ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=LID_side, ft_times=tf_times['nonmatch'],
                cdrs_rater=CDRS_RATER,
            ) 
            ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=noLID_side, ft_times=tf_times['nonmatch'],
                cdrs_rater=CDRS_RATER,
            )
            if LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                    sub=sub, side='both', ft_times=tf_times['nonmatch'],
                    cdrs_rater=CDRS_RATER,
                )
                if incl_PRE_LID:
                    start_lid_i = np.where(ft_cdrs['bi'] > 0)[0][0]
                    start_lid_t = tf_times['nonmatch'][start_lid_i]
                    start_prelid_t = start_lid_t - PRELID_MIN
                    prelid_sel = np.logical_and(tf_times['nonmatch'] > start_prelid_t,
                                                tf_times['nonmatch'] < start_lid_t)
                    # select preLID timefreq values
                    tf_values['nonmatch_preLID'] = tf_values['nonmatch'][:, prelid_sel]
                    cdrs_categs['nonmatch_preLID'] = [0.5] * len(tf_values['nonmatch_preLID'])
                
            # extract BASELINE and BILAT selections from tf_values, before
            # tf_values itself is adjusted due to the unilat selection
            if BASELINE_CORRECT:
                bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
                # excl pre LID from BASELINE
                if incl_PRE_LID and start_prelid_t > 5:
                    bl_sel = bl_sel.astype(int) - prelid_sel.astype(int)
                    bl_sel[bl_sel == -1] = 0
                    bl_sel = bl_sel.astype(bool)
                # select baseline
                tf_values['nonmatch_BL'] = tf_values['nonmatch'][:, bl_sel]

                if i_rep == 0:  # add tuple 
                    mean_stats['noLID'].append((f'{sub}_nonmatch',
                                                tf_values['nonmatch_BL']))

            
            if LAT_or_SCALE == 'SCALE':
                # select lateral spectral values for both sides (now nonmatch)
                bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0,
                                            ft_cdrs['noLID'] > 0)       
                # take for nonmatch-epys-hemisphere, the non-matching LID_cdrs side 
                tf_values['bi_nonmatch'] = tf_values['nonmatch'][:, bi_lid_sel]
                cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['bi'],
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False,
                                            return_coding_dict=True)
                cdrs_categs['bi_nonmatch'] = cdrs_cats[bi_lid_sel]

                # add nonmatch STN during bilateral LID for STATS
                stats_lid_pow['nonmatch'].append(tf_values[f'bi_nonmatch'])                        
                stats_lid_cdrs['nonmatch'].append(ft_cdrs['noLID'][bi_lid_sel])  # add cdrs values for body side corr to powers

                
            elif LAT_or_SCALE == 'LAT_BILAT':
                # select lateral spectral values for both sides (now nonmatch)
                bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0,
                                            ft_cdrs['noLID'] > 0)
                # print(f'Subject-{sub} bilateral-LID: n={sum(bi_lid_sel)}')    
                # take for nonmatch-epys-hemisphere, the non-matching LID_cdrs side 
                tf_values['bi_nonmatch2'] = tf_values['nonmatch'][:, bi_lid_sel]
                cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['LID'],  # was bi
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False,
                                            return_coding_dict=True)
                cdrs_categs['bi_nonmatch2'] = cdrs_cats[bi_lid_sel]

                if RETURN_FREQ_CORR: cdrs_full['bi_nonmatch2'] = ft_cdrs['LID'][bi_lid_sel]

                # take for nonmatch-epys-hemisphere, the MATCHING noLID_cdrs side 
                tf_values['bi_match2'] = tf_values['nonmatch'][:, bi_lid_sel]
                cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['noLID'],  # was bi
                                            preLID_minutes=0,
                                            preLID_separate=False,
                                            convert_inBetween_zeros=False,
                                            return_coding_dict=True)
                cdrs_categs['bi_match2'] = cdrs_cats[bi_lid_sel]

                if RETURN_FREQ_CORR: cdrs_full['bi_match2'] = ft_cdrs['noLID'][bi_lid_sel]
                
            # select nonmatching EPHYS to unilat LID
            if LAT_or_SCALE in ['LAT_UNI', 'SCALE']:
                uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
                tf_values['nonmatch'] = tf_values['nonmatch'][:, uni_lid_sel]
                if LAT_or_SCALE == 'SCALE':
                    cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                                preLID_minutes=0,
                                                preLID_separate=False,
                                                convert_inBetween_zeros=False,)
                    cdrs_categs['nonmatch'] = cdrs_cats[uni_lid_sel]

            # create sub dict for FREQxCORR
            if RETURN_FREQ_CORR: FREQ_CORRs[sub] = {}

            # calculate MEAN PSD VALUES (match vs non-match) for LATERALITY
            for match_label in ['match', 'nonmatch']:  # loops over two STNs
                # for all settings, calculate baseline if needed
                if BASELINE_CORRECT:
                    bl_psd = np.mean(tf_values[f'{match_label}_BL'], axis=1)
                
                # if LATERALITY, take mean psds per side and subject,
                # and add individual mean to grand plotting
                if LAT_or_SCALE == 'LAT_UNI':
                    mean_psd = np.mean(tf_values[match_label], axis=1)
                    if BASELINE_CORRECT:
                        mean_psd = (mean_psd - bl_psd) / bl_psd * 100
                    psds_to_plot[match_label].append(list(mean_psd))
                
                # for SCALING, add mean-psds per category, per subject
                elif LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                    # correct bilat-psds against the baseline from the corresponding
                    # hemisphere, add all bilat psds to one list
                    for label in [match_label, f'bi_{match_label}']:
                        
                        if LAT_or_SCALE == 'LAT_BILAT' and not label.startswith('bi'):
                            # dont include unilateral values in LAT_BILAT
                            continue

                        dict_lab = label.split('_')[0]  # take match/nonmatch/bi
                    
                        if LAT_or_SCALE == 'SCALE':
                            # add preLID category
                            if incl_PRE_LID:
                                prelid_psd = np.nanmean(tf_values[f'{match_label}_preLID'],
                                                     axis=1)
                                if BASELINE_CORRECT:
                                    prelid_psd = (prelid_psd - bl_psd) / bl_psd * 100
                                if 0.5 not in psds_to_plot[dict_lab].keys():
                                    psds_to_plot[dict_lab][0.5] = []
                                if not all(isna(prelid_psd)):
                                    psds_to_plot[dict_lab][0.5].append(list(prelid_psd))
                                    # keep track of individual subs included
                                    try: subs_incl[dict_lab][0.5].append(sub)
                                    except KeyError: subs_incl[dict_lab][0.5] = [sub,]
                            # add subject-mean per category (add empty list if necessary)
                            for cat in np.unique(cdrs_categs[label]):
                                cat_sel = cdrs_categs[label] == cat
                                cat_values = np.array(tf_values[label])[:, cat_sel]
                                mean_cat_values = np.mean(cat_values, axis=1)
                                if BASELINE_CORRECT:
                                    mean_cat_values = (mean_cat_values - bl_psd) / bl_psd * 100
                                # save psds, bilateral is merged into 'bi'
                                if cat not in psds_to_plot[dict_lab].keys():
                                    psds_to_plot[dict_lab][cat] = []
                                psds_to_plot[dict_lab][cat].append(list(mean_cat_values))

                                # keep track of individual subs included
                                try: subs_incl[dict_lab][cat].append(sub)
                                except KeyError: subs_incl[dict_lab][cat] = [sub,]
                            
                        elif LAT_or_SCALE == 'LAT_BILAT':
                            # save only bilateral psds, but with laterality
                            # put bi_match1 and bi_match2 in bi_match
                            for i_lab, lab in enumerate([f'{label}1', f'{label}2']):

                                # create sub dict for FREQxCORR
                                if RETURN_FREQ_CORR:
                                    if i_lab == 0:
                                        # start array with baseline values
                                        bl_values = np.array(tf_values[f'{match_label}_BL'])
                                        bl_values = (bl_values.T - bl_psd) / bl_psd * 100
                                        freqCorr_arr = bl_values.T
                                        freqCorr_scores = [0] * freqCorr_arr.shape[1]
                                    temp_values = np.array(tf_values[lab]).T
                                    temp_values = (temp_values - bl_psd) / bl_psd * 100
                                    freqCorr_arr = np.concatenate([freqCorr_arr, temp_values.T], axis=1)
                                    freqCorr_scores.extend(list(cdrs_full[lab]))
                                    
                                for cat in np.unique(cdrs_categs[lab]):
                                    # results only in bi_match vs bi_nonmatch
                                    cat_sel = cdrs_categs[lab] == cat  # use both match1 and match2
                                    cat_values = np.array(tf_values[lab])[:, cat_sel]
                                    mean_cat_values = np.mean(cat_values, axis=1)
                                    if BASELINE_CORRECT:
                                        mean_cat_values = (mean_cat_values - bl_psd) / bl_psd * 100
                                    if cat not in psds_to_plot[label].keys():
                                        psds_to_plot[label][cat] = []
                                    psds_to_plot[label][cat].append(list(mean_cat_values))

                                    # keep track of individual subs included
                                    try: subs_incl[label][cat].append(sub)
                                    except KeyError: subs_incl[label][cat] = [sub,]
                            

                            if RETURN_FREQ_CORR:
                                FREQ_CORRs[sub][label] = (freqCorr_arr, freqCorr_scores, tf_freqs)

        # process sub stats at end of subject iteration
        if PROCESS_STATS:
            for side in ['match', 'nonmatch']:
                temp_pows = stats_lid_pow[side][0]
                temp_cdrs = stats_lid_cdrs[side][0]
                for n in np.arange(1, len(stats_lid_pow[side])):
                    temp_pows = np.concatenate([temp_pows,
                                                stats_lid_pow[side][n]],
                                                axis=1)
                    temp_cdrs = np.concatenate([temp_cdrs,
                                                stats_lid_cdrs[side][n]])
                mean_stats['LID'].append((f'{sub}_{side}', temp_pows))
                mean_stats['CDRS'].append((f'{sub}_{side}', temp_cdrs))
            mean_stats['freqs'] = tf_freqs

    # merge all contra and ipsilateral psds together for LAT_ALL_SCALE
    if PLOT_ONLY_MATCH:
        combi_psds = {'all_match': {}}
        all_cats = list(np.unique(list(psds_to_plot['match'].keys())))
        all_cats.extend(np.unique(list(psds_to_plot['nonmatch'].keys())))
        all_cats.extend(np.unique(list(psds_to_plot['bi'].keys())))
        all_cats = list(np.unique(all_cats))
        # create list for every category
        for cat in all_cats: combi_psds['all_match'][cat] = []
        for side in psds_to_plot.keys():        
            # fill cat-list with all different sides
            for cat in psds_to_plot[side].keys():
                combi_psds['all_match'][cat].extend(psds_to_plot[side][cat])
        psds_to_plot = combi_psds

    if RETURN_FREQ_CORR:
        print('not plotting, only returning freq corr values')
        return FREQ_CORRs

    # extract number of incl indiv subs per cat
    n_subs_incl = {}
    if n_reps == 1:
        for side in subs_incl.keys():
            n_subs_incl[side] = {}
            for cat in subs_incl[side].keys():
                n_subs_incl[side][cat] = len(np.unique(subs_incl[side][cat]))
    if n_reps == 2 or PLOT_ONLY_MATCH:
        n_subs_incl['all_match'] = {c: [] for c in all_cats}
        for side in subs_incl.keys():
            for cat in subs_incl[side].keys():
                n_subs_incl['all_match'][cat].extend(subs_incl[side][cat])
        for cat in n_subs_incl['all_match']:
            n_subs_incl['all_match'][cat] = len(
                np.unique(n_subs_incl['all_match'][cat])
            )
            
    if PROCESS_STATS:
        process_mean_stats(datatype='STN', mean_stats=mean_stats,
                           DATA_VERSION=DATA_VERSION,
                           FT_VERSION=FT_VERSION,
                           save_stats=True,)
        get_binary_p_perHz(datatype='STN', save_date=p_SAVED_DATE,
                           DATA_VERSION=DATA_VERSION,
                           FT_VERSION=FT_VERSION,)

    if incl_PRE_LID:
        coding_keys = list(coding_dict.keys())
        coding_values = list(coding_dict.values())
        coding_dict ={coding_keys[0]: coding_values[0],
                      'pre-LID': 0.5}
        for k, v in zip(coding_keys[1:], coding_values[1:]):
            coding_dict[k] = v

    ### PLOTTING PART
    if LAT_or_SCALE == 'LAT_UNI':
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
                            fsize=fsize,
                            single_sub_lines=SINGLE_SUB_LINES,)
    
    
    elif LAT_or_SCALE == 'SCALE':
        plot_scaling_LID(plt_ax_to_return=plt_ax_to_return,
                         psds_to_plot=psds_to_plot,
                         datatype='STN',
                         tf_freqs=tf_freqs,
                         n_subs_incl=n_subs_incl,
                         cdrs_cat_coding=coding_dict,
                         BASELINE_CORRECT=BASELINE_CORRECT,
                         LOG_POWER=LOG_POWER,
                         SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                         STD_ERR=STD_ERR,
                         BREAK_X_AX=BREAK_X_AX,
                         fsize=fsize,
                         fig_name=fig_name,
                         single_sub_lines=SINGLE_SUB_LINES,
                         show_only_gamma=SHOW_ONLY_GAMMA,
                         SHOW_SIGN=SHOW_SIGN,
                         p_SAVED_DATE=p_SAVED_DATE,
                         PLOT_ONLY_MATCH=PLOT_ONLY_MATCH,
                         DATA_VERSION=DATA_VERSION,
                         FT_VERSION=FT_VERSION,)
    
    elif LAT_or_SCALE == 'LAT_BILAT':
        plot_scaling_LID(plt_ax_to_return=plt_ax_to_return,
                         psds_to_plot=psds_to_plot,
                         datatype='STN',
                         tf_freqs=tf_freqs,
                         n_subs_incl=n_subs_incl,
                         cdrs_cat_coding=coding_dict,
                         BASELINE_CORRECT=BASELINE_CORRECT,
                         LOG_POWER=LOG_POWER,
                         SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                         STD_ERR=STD_ERR,
                         BREAK_X_AX=BREAK_X_AX,
                         fsize=fsize,
                         fig_name=fig_name,
                         single_sub_lines=SINGLE_SUB_LINES,
                         show_only_gamma=SHOW_ONLY_GAMMA,
                         DATA_VERSION=DATA_VERSION,
                         FT_VERSION=FT_VERSION,)       
    
    # if PROCESS_STATS: return mean_stats

import json

def plot_scaling_LID(
    psds_to_plot, tf_freqs,
    cdrs_cat_coding, datatype,
    cdrs_origin=None,
    plt_ax_to_return=False,
    BASELINE_CORRECT=True, LOG_POWER=False,
    SELECT_MOVEMENT=False,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True,
    BREAK_X_AX=True, fsize=14,
    SAVE_PLOT=True, SHOW_PLOT=False,
    fig_name='PSD_CDRS_scaling_STN_nXX',
    single_sub_lines=False,
    show_only_gamma=False,
    n_subs_incl=False,
    SHOW_SIGN=False,
    PLOT_ONLY_MATCH=False,
    p_SAVED_DATE='0000',
    DATA_VERSION = 'v4.0', FT_VERSION='v4',
):
    """
    Input:
        - psds_to_plot: contains first side-keys, then cat-keys
    """
    allowed_dtypes = ['STN', 'ECOG', 'MIC', 'TRGC']
    assert datatype.upper() in allowed_dtypes, (
        f'datatype ({datatype}) should be in {allowed_dtypes}'
    )

    if SHOW_SIGN:
        print('load significancies')
        store_path = join(get_project_path('results'), 'stats',
                          f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                          f'{datatype.upper()}_LMM_noLID_vs_LID')

        with open(join(store_path, f'{datatype.upper()}_LMM_results_'
                                   f'pvalues_{p_SAVED_DATE}.json'),
                  'r') as json_file:
            p_dict = json.load(json_file)

    if plt_ax_to_return == False:
        fig, axes = plt.subplots(1, len(psds_to_plot),
                                 figsize=(len(psds_to_plot) * 6, 6))
    else:
        axes = plt_ax_to_return
    
    # colors = [list(get_colors('PaulTol).values())[c] for c in [4, 3, 0, 7]]  # colors are called by cat-value, so 0 is never used if baseline corrected
    colors = get_colors('Jacoba')
    # if not 'pre-LID' in cdrs_cat_coding.keys(): colors = colors[:]   # skip gray
    
    for i_ax, side in enumerate(psds_to_plot.keys()):
        # prevent error with none subscriptable ax vs axes
        if len(psds_to_plot) == 1: loop_ax = axes
        else: loop_ax = axes[i_ax]

        if side == 'bi': ax_title = f'{datatype} during bilateral dyskinesia'
        elif side == 'match': ax_title = f'{datatype} during only contralateral dyskinesia'
        elif side == 'nonmatch': ax_title = f'{datatype} during only ipsilateral dyskinesia'
        elif side == 'bi_match': ax_title = f'Contralateral {datatype} during bilateral dyskinesia'
        elif side == 'bi_nonmatch': ax_title = f'Ipsilateral {datatype} during bilateral dyskinesia'
        elif side == 'all_match': ax_title = f'{datatype} versus all contralateral dyskinesia'
        elif side == 'all_nonmatch': ax_title = f'{datatype} versus all ipsilateral dyskinesia'
        elif side == 'ipsilateral': ax_title = f'{datatype.upper()} (ipsilateral STN) versus {cdrs_origin} dyskinesia'
        elif side == 'contralateral': ax_title = f'{datatype.upper()} (contralateral STN) versus {cdrs_origin} dyskinesia'
        
        if PLOT_ONLY_MATCH: ax_title = f'{datatype} versus all contralateral dyskinesia'
        if SELECT_MOVEMENT: ax_title += f'\n({SELECT_MOVEMENT})'



        for i_cat, cat in enumerate(psds_to_plot[side].keys()):

            cat_sel = [v == cat for v in cdrs_cat_coding.values()]
            cat_name = list(cdrs_cat_coding.keys())[np.where(cat_sel)[0][0]]
            try: n_subs = n_subs_incl[side][cat]
            except: n_subs = len(psds_to_plot[side][cat])

            PSD = {}
            psds = np.array(psds_to_plot[side][cat])
            PSD['mean'] = np.nanmean(psds, axis=0)
            # if nanmean only contains NaN, skip this category
            if np.isnan(PSD['mean']).all():
                print(f'skip CAT {cat} (n={n_subs}) bcs of only-NaNs')
                continue
            PSD['sd'] = np.nanstd(psds, axis=0)
            if STD_ERR: PSD['sd'] = PSD['sd'] / np.sqrt(psds.shape[0])
            
            # blank freqs irrelevant after SSD
            blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
            for k in PSD: PSD[k][blank_sel] = np.nan

            # ge significancies
            if SHOW_SIGN:
                p_freqs = np.array(p_dict['freqs'].copy())
                ps = np.array(p_dict['p_values'].copy())
            
            # smoothen signal for plot (both mean and stddev) without cutting length
            if SMOOTH_PLOT_FREQS > 0:
                for k in PSD:
                    idx = np.arange(len(PSD[k]))
                    s_win = SMOOTH_PLOT_FREQS // 2
                    PSD[k] = [np.nanmean(
                        PSD[k][np.logical_and(idx > (i_v - s_win), idx < (i_v + s_win))]
                    ) for i_v in idx]
            # n-subjects to add to legend-label
            n_subs_cat = psds.shape[0]
                    
            # BREAK X AXIS and adjust xticks and labels
            if BREAK_X_AX:
                assert sum(np.isnan(PSD['mean'])) != len(PSD['mean']), (
                    'only NaNs in PSD["mean"]'
                )
                PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
                if SHOW_SIGN: ps, _, _ = break_x_axis_psds_ticks(p_freqs, ps)
                x_axis = xticks
                if single_sub_lines:
                    cut_psds = np.zeros((psds.shape[0], len(x_axis)))
                    for row in np.arange(psds.shape[0]):
                        cut_psds[row, :], _, _ = break_x_axis_psds_ticks(
                            tf_freqs.copy(), psds[row])
                    psds = cut_psds
                
                if show_only_gamma:
                    f_sel = []
                    for f in xlabels:
                        if not isinstance(f, str) and f >= 60 and f  <= 90: f_sel.append(True)
                        else: f_sel.append(False)
                    xlabels = list(compress(xlabels, f_sel))
            
            if not BREAK_X_AX:
                x_axis = tf_freqs
                if show_only_gamma:
                    f_sel = np.logical_and(tf_freqs >= 60, tf_freqs <= 90)

            if show_only_gamma:
                x_axis = np.array(x_axis)[f_sel]
                xticks = np.array(xticks)[f_sel]
                for k in PSD.keys(): PSD[k] = PSD[k][f_sel]
                if single_sub_lines: psds = psds[:, f_sel]
                if SHOW_SIGN: ps = ps[f_sel]

            # PLOT MEAN PSD LINE, AND STDDEV SHADING
            if single_sub_lines and n_subs_cat <= 5:
                loop_ax.plot(x_axis, psds.T, lw=3, alpha=.5,
                                label=f'{cat_name} dyskinesia'
                                f' (n={n_subs})',  # n_subs instead of n_subs_cat
                                color=colors[int(cat)], )
            
            elif not single_sub_lines:
                # plot full line
                loop_ax.plot(x_axis, PSD['mean'], lw=5, alpha=.5,
                                label=f'{cat_name} dyskinesia'
                                f' (n={n_subs})',
                                color=colors[int(cat)], )
                # PLOT VARIANCE SHADING
                loop_ax.fill_between(x=x_axis, y1=PSD['mean'] - PSD['sd'],
                                     y2=PSD['mean'] + PSD['sd'],
                                     alpha=.3,
                                     color=colors[int(cat)], )

                # one significance shade for all severities
                if SHOW_SIGN:
                    print(f'SHOW SIGN part in plotting, ps: {ps}')
                    print(f'length ps = {len(ps)}, 68 freqs?' ({[(p, type(p)) for p in ps]}))
                    sig_mask = np.array(ps) < (.05 / 68)  # 68 freqs compared
                    loop_ax.fill_between(x=x_axis, y1=-50, y2=100,
                                         alpha=.05, color=colors[-1],
                                         where=sig_mask,)
        
        # SET AXTICKS
        if BREAK_X_AX:
            loop_ax.set_xticks(xticks[::8],)
            loop_ax.set_xticklabels(xlabels[::8], fontsize=fsize)
            
        else:
            loop_ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
            loop_ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

        if not LOG_POWER: loop_ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
                                    color='gray', lw=1, alpha=.5,)
        # set LABELS, LEGEND, AXES
        loop_ax.set_title(ax_title, size=fsize, weight='bold',)
        loop_ax.set_xlabel('Frequency (Hz)', size=fsize,)
        if datatype.upper() == 'MIC': ylabel = 'Max. Imag. Coh. (a.u.)'
        elif datatype.upper() == 'TRGC': ylabel = 'time-res. Granger Caus. (a.u.)'
        else: ylabel = 'Power (a.u.)'
        if LOG_POWER: ylabel = f'Log. {ylabel}'
        if BASELINE_CORRECT:
            ylabel = ylabel.replace('(a.u.)', '%-change vs bilat-no-LID (a.u.)')
        loop_ax.set_ylabel(ylabel, size=fsize,)
        # plot legend without duplicates
        leg_info = loop_ax.get_legend_handles_labels()
        hands, labs = remove_duplicate_legend(leg_info)
        loop_ax.legend(hands, labs, frameon=False,
                fontsize=fsize, loc='upper left')
        
        loop_ax.spines['top'].set_visible(False)
        loop_ax.spines['right'].set_visible(False)
        loop_ax.tick_params(axis='both', size=fsize, labelsize=fsize)

    
    if plt_ax_to_return != False:
        return axes

    else:
        # plot or save axes from here
        # equalize axes
        if len(psds_to_plot) > 1:
            ymin = min([min(ax.get_ylim()) for ax in axes])
            ymax = max([max(ax.get_ylim()) for ax in axes])
            for ax in axes: ax.set_ylim(ymin, ymax)

            for ax in axes: ax.tick_params(axis='both', size=fsize, labelsize=fsize)
        else:
            axes.tick_params(axis='both', size=fsize, labelsize=fsize)
        plt.tight_layout()

        if SAVE_PLOT:
            if datatype.upper() in ['STN', 'ECOG']: folder = 'descr_PSDs'
            elif datatype.upper() in ['MIC', 'TRGC']: folder = 'connectivity'
            path = join(get_project_path('figures'),
                        'ft_exploration',
                        f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                        folder)
            if not exists(path): makedirs(path)
            plt.savefig(join(path, fig_name),
                        facecolor='w', dpi=300,)

        if SHOW_PLOT: plt.show()
        else: plt.close()



def plot_unilateral_LID(
    psds_to_plot, tf_freqs, n_uni_subs,
    datatype, plt_ax_to_return=None,
    BASELINE_CORRECT=True, LOG_POWER=False,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True, BREAK_X_AX=True,
    fsize=14, single_sub_lines=False,
    show_only_gamma=False,
):
    assert datatype.upper() in ['STN', 'ECOG'], (
        f'datatype ({datatype}) should be STN or ECOG'
    )

    if plt_ax_to_return == None or plt_ax_to_return == False:
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
        
        # blank freqs irrelevant after ssd
        blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        for k in PSD: PSD[k][blank_sel] = np.nan
        

        # smoothen signal for plot
        if SMOOTH_PLOT_FREQS > 0:
            for k in PSD:
                idx = np.arange(len(PSD[k]))
                s_win = SMOOTH_PLOT_FREQS // 2
                PSD[k] = [np.nanmean(PSD[k][np.logical_and(idx > (i_v - s_win),
                                                           idx < (i_v + s_win))])
                                            for i_v in idx]
                
        # add n-subjects to leg_label
        leg_label[match_label] += f'(n={psds.shape[0]})'
                
        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            PSD, xticks, xlabels = break_x_axis_psds_ticks(tf_freqs, PSD)
            x_axis = xticks
            
            if single_sub_lines:
                cut_psds = np.zeros((psds.shape[0], len(x_axis)))
                for row in np.arange(psds.shape[0]):
                    cut_psds[row, :], _, _ = break_x_axis_psds_ticks(tf_freqs, psds[row])
                psds = cut_psds

        else: x_axis = tf_freqs

        # PLOT LINE
        if not single_sub_lines:
            ax.plot(x_axis, PSD['mean'], label=leg_label[match_label],
                    color=colors[i], lw=5, alpha=.5,)
        elif single_sub_lines:
            # for y in psds:
            ax.plot(x_axis, psds.T, label=leg_label[match_label],
                    color=colors[i], lw=2, alpha=.7,)
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
        
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    if not LOG_POWER: ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
                                color='gray', lw=1, alpha=.5,)
    
    ax.set_title(f'{datatype.upper()}-subjects with unilateral Dyskinesia (n={n_uni_subs})',
                  size=fsize, weight='bold',)
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Power (a.u.)'
    if LOG_POWER: ylabel = f'Log. {ylabel}'
    if BASELINE_CORRECT: ylabel = f'{ylabel[:-6]} %-change vs bilat-no-LID (a.u.)'
    ax.set_ylabel(ylabel, size=fsize,)
    
    # plot legend without duplicates
    leg_info = ax.get_legend_handles_labels()
    hands, labs = remove_duplicate_legend(leg_info)
    ax.legend(hands, labs, frameon=False,
              fontsize=fsize, loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)
    
    plt.tight_layout()

    if plt_ax_to_return: return ax
    else: plt.show()


def break_x_axis_psds_ticks(tf_freqs, PSD, PSD_sd=False,
                            x_break = (35, 60), nan_pad = 5):
    """
    remove frequencies to not show on X-axis, replace with
    NaN-values (n = nan_pad)

    Arguments:
        - tf_freqs: frequencies
        - PSD: psd values, can be dict, arr or list (should contain
            float and NO INTEGERS to be able to add NaNs)
        - PSD_sd: similar to PSD, then std dev instead of mean i.e.

    Returns:
        - PSD: nan-padded dor freqs not to show
        - (IF INSERTED: corresponding std dev values)
        - xticks: x-array for plotting
        - corr labels for ticks
    """
    if isinstance(PSD[0], int): PSD = PSD.astype(float)
    
    del_sel = np.logical_and(tf_freqs > x_break[0],
                             tf_freqs < x_break[1])

    if isinstance(PSD, list): PSD = np.array(PSD)

    if isinstance(PSD, dict):
        del_sel = np.logical_or(del_sel, np.isnan(PSD['mean']))
        
        assert not np.isnan(del_sel).all(), 'all PSDs removed, probably only NaNs in PSD["mean"]'
        
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

    elif isinstance(PSD, np.ndarray):    
        del_sel = np.logical_or(del_sel, np.isnan(PSD))

        PSD = np.delete(PSD, del_sel,)
        
        if isinstance(PSD_sd, np.ndarray):
            PSD_sd = np.delete(PSD_sd, del_sel,)

        plt_freqs = np.delete(tf_freqs.copy(), del_sel,).astype(float)
        i_sel = np.argmin(abs(plt_freqs - x_break[0]))
        PSD = np.insert(PSD, i_sel + 1, values=[np.nan] * nan_pad,)
        
        if isinstance(PSD_sd, np.ndarray):
            PSD_sd = np.insert(PSD_sd, i_sel + 1, values=[np.nan] * nan_pad,)
        
        plt_freqs = np.insert(plt_freqs, i_sel + 1,
                                values=[np.nan] * nan_pad,)

        xticks = np.arange(len(PSD))
    
    else:
        raise ValueError(f'PSD should be array or list, type: {type(PSD)}')
        
    xlabels = [''] * len(xticks)
    low_ticks = plt_freqs[plt_freqs <= x_break[0]]
    low_ticks = [round(t) for t in low_ticks]
    xlabels[:len(low_ticks)] = low_ticks
    high_ticks = plt_freqs[plt_freqs >= x_break[1]]
    high_ticks = [round(t) for t in high_ticks]
    xlabels[len(xlabels) - len(high_ticks):] = high_ticks

    # correct missing NaNs (necessary during sign True/False arras)
    if not np.isnan(PSD).any():
        for i, lab in enumerate(xlabels):
            if lab == '' and PSD[i] == True: PSD[i] = False

    if isinstance(PSD, dict): return PSD, xticks, xlabels
    elif isinstance(PSD, np.ndarray):
        if isinstance(PSD_sd, np.ndarray): return PSD, PSD_sd, xticks, xlabels
        else: return PSD, xticks, xlabels



def plot_ECOG_PSD_vs_LID(
    all_timefreqs, sel_subs=None,
    CDRS_RATER='Patricia',
    LAT_or_SCALE='LAT_UNI',
    LOG_POWER=True, ZSCORE_FREQS=False,
    SELECT_ON_ACC_RMS=False,
    SMOOTH_PLOT_FREQS=0, STD_ERR=True,
    BASELINE_CORRECT=False,
    BREAK_X_AX=False, plt_ax_to_return=False,
    fsize=12,
    fig_name='PLOT_ECoG_PSD_vs_DYSK',
    single_sub_lines=False,
    PLOT_ONLY_MATCH=False,
    SHOW_ONLY_GAMMA=False,
    SHOW_SIGN=False,
    PROCESS_STATS=False,
    p_SAVED_DATE='0000',
    DATA_VERSION='v4.0',
    FT_VERSION='v4',
):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean unilateral PSDs versus unilateral CDRS.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LAT_or_SCALE='LAT_UNI', should be LAT_UNI (laterality of
            unilateral LID), 'LAT_BILAT' (laterality during bilateral
            LID), or SCALE (PSDs vs CDRS categories),
            or LAT_ALL_SCALE for all uni- and bilat LID splitted
            on contralat and ipsilat;
        - SELECT_ON_ACC_RMS: defaults False: no selection,
            INCL_MOVE or EXCL_MOVE
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
    # prepare all dictionaries and lists to store info in 
    if LAT_or_SCALE == 'LAT_UNI':
        psds_to_plot = {'match': [], 'nonmatch': []}
    elif LAT_or_SCALE == 'SCALE':
        psds_to_plot = {}
        for lab in ['match', 'nonmatch', 'bi']: psds_to_plot[lab] = {}
    elif LAT_or_SCALE =='LAT_BILAT':
        psds_to_plot = {}
        for lab in ['match', 'nonmatch','bi_match', 'bi_nonmatch']:
            psds_to_plot[lab] = {}
    elif LAT_or_SCALE =='LAT_ALL_SCALE':
        psds_to_plot, cdrs_categs, cdrs_full = {}, {}, {}  # start with empty dict, dont use match/nonmatch
        for lab in ['match', 'nonmatch', 'bi',
                    'bi_match', 'bi_nonmatch']: psds_to_plot[lab] = {}

    if LAT_or_SCALE == 'LAT_ALL_SCALE': n_reps = 2
    else: n_reps = 1

    # select out patients if defined
    if sel_subs:
        subs = [sub for sub in all_timefreqs.keys()
                if sub in sel_subs]
    else:
        subs = all_timefreqs.keys()

    n_uni_subs = 0
    subs_incl = {k: {} for k in psds_to_plot.keys()}  # keep track of indiv subs per cat

    # if movement selection on RMS is defined, load pickled class for all subjects
    # if SELECT_ON_ACC_RMS:
    featLabPath = join(get_project_path('data'),
                        'prediction_data',
                        'featLabelClasses',
                        f'featLabels_ft{FT_VERSION}_Cdrs_StnOnly.P')
    feat10sClass = load_class_pickle(featLabPath,
                                        convert_float_np64=True)
    print(f'10s-feature class loaded for RMS-ACC selection ({featLabPath})')

    # gets per hemisphere the subject ID and the mean-psd-array (no LID and LID)
    # CDRS contains the scores corresponding to LID
    mean_stats = {'LID': [], 'noLID': [], 'CDRS': []}

    for sub in subs:
        if sub.startswith('1'): continue  # SKIP NON-ECOG subjects
        # store temporary powers and LID scores for stats
        stats_lid_pow = {'match': [], 'nonmatch': []}
        stats_lid_cdrs = {'match': [], 'nonmatch': []}

        for i_rep in np.arange(n_reps):

            # if 'LAT_ALL_SCALE' is defined, first run uni, than bilat
            if n_reps == 2 and i_rep == 0: LAT_or_SCALE = 'SCALE'
            elif n_reps == 2 and i_rep == 1: LAT_or_SCALE = 'LAT_BILAT'

            # create empty dicts for current subject
            tf_values_dict, ft_cdrs, cdrs_categs = {}, {}, {}
            
            # check for presence unilateral LID
            (LID_side, noLID_side, LFP_match_side,
             LFP_nonmatch_side) = find_unilateral_LID_sides(sub, CDRS_RATER=CDRS_RATER)
            ecog_side = get_ecog_side(sub)

            if LAT_or_SCALE == 'LAT_UNI':
                if not LID_side:
                    print(f'skip sub-{sub}, no uni-lateral dyskinesia')
                    continue  # no UNILATERAL DYSKINESIA PRESENT
                else: n_uni_subs += 1  # INCLUDE SUBJECT WITH UNILATERAL LID IN PLOT
                # define LATERALITY OF LID VERSUS ECOG (in case of unilat lid)
                if ecog_side == LFP_match_side: match_label_uni = 'match'
                elif ecog_side == LFP_nonmatch_side: match_label_uni = 'nonmatch'
    
            elif LAT_or_SCALE in ['LAT_BILAT', 'SCALE']:
                add_uni = False
                if LID_side:
                    n_uni_subs += 1
                    add_uni = True
                    # define LATERALITY OF LID VERSUS ECOG (in case of unilat lid)
                    if ecog_side == LFP_match_side: match_label_uni = 'match'
                    elif ecog_side == LFP_nonmatch_side: match_label_uni = 'nonmatch'
                # add labels for bilateral cdrs
                LFP_match_side, noLID_side = ecog_side, ecog_side
                if ecog_side == 'left':
                    LID_side, LFP_nonmatch_side = 'right', 'right'
                elif ecog_side == 'right':
                    LID_side, LFP_nonmatch_side = 'left', 'left'            
            
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


            ### CREATE BASELINE WIHTOUT MOVEMENT ###
            if BASELINE_CORRECT:
                bl_sel = tf_times < 10  # baseline not after ten minutes
                bl_times = tf_times[bl_sel]
                bl_values = tf_values[:, bl_sel]
                bl_values, bl_times = select_tf_on_movement(
                    feat10sClass=feat10sClass, sub=sub,
                    tf_values_arr=bl_values, tf_times_arr=bl_times,
                    SELECT_ON_ACC_RMS='EXCL_MOVE',
                )
                bl_cdrs = find_select_nearest_CDRS_for_ephys(
                    sub=sub, cdrs_rater=CDRS_RATER,
                    side='both', ft_times=bl_times,
                )
                bl_sel = bl_cdrs == 0
                bl_times = bl_times[bl_sel]
                bl_values = bl_values[:, bl_sel]
                bl_psd = np.mean(bl_values, axis=1)
                

            ### SELECT OUT ON ACC RMS IF DEFINED HERE ###
            if SELECT_ON_ACC_RMS:
                tf_values, tf_times = select_tf_on_movement(
                    feat10sClass=feat10sClass, sub=sub,
                    tf_values_arr=tf_values, tf_times_arr=tf_times,
                    SELECT_ON_ACC_RMS=SELECT_ON_ACC_RMS,
                    # RMS_Z_THRESH=0,  # defaults to -0.5
                )

            # find matching windows (UNILAT LID) for both ECoG PSDs
            # matching LFP side to LID
            ft_cdrs['LID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=LID_side, ft_times=tf_times,
                cdrs_rater=CDRS_RATER,
            ) 
            ft_cdrs['noLID'] = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=noLID_side, ft_times=tf_times,
                cdrs_rater=CDRS_RATER,
            )
            if LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                ft_cdrs['bi'] = find_select_nearest_CDRS_for_ephys(
                    sub=sub, side='both', ft_times=tf_times,
                    cdrs_rater=CDRS_RATER,
                ) 

            # # extract BASELINE and BILAT selections from tf_values, before
            # # tf_values itself is adjusted due to the unilat selection
            # if BASELINE_CORRECT:
            #     bl_sel = np.logical_and(ft_cdrs['LID'] == 0, ft_cdrs['noLID'] == 0)
            #     tf_values_BL = tf_values[:, bl_sel]
            #     bl_psd = np.mean(tf_values_BL, axis=1)

            #     if i_rep == 0:  # add tuple 
            #         mean_stats['noLID'].append((f'{sub}_match', tf_values[:, bl_sel]))

            # select moments with BILATERAL present dyskinesia 
            if LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                bi_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] > 0)
                # add all bilateral cdrs
                if LAT_or_SCALE == 'SCALE':
                    tf_values_dict['bi'] = tf_values[:, bi_lid_sel]
                    cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['bi'],
                                                 preLID_minutes=0,
                                                 preLID_separate=False,
                                                 convert_inBetween_zeros=False)
                    cdrs_categs['bi'] = cdrs_cats[bi_lid_sel]
                    # add match STN during bilateral LID for STATS
                    stats_lid_pow['match'].append(tf_values[:, bi_lid_sel]) 
                    stats_lid_cdrs['match'].append(ft_cdrs['LID'][bi_lid_sel])  # add cdrs values for body side corr to powers

                # add lateral-cdrs scores DURING BILATERAL cdrs
                # uni-lat-ECOG-tf-values are added twice, once labeled match
                # with contra-lat-cdrs, once labeled non-match with ipsi-lat-cdrs
                elif LAT_or_SCALE == 'LAT_BILAT':
                    tf_values_dict['bi_match'] = tf_values[:, bi_lid_sel]
                    cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                                 preLID_minutes=0,
                                                 preLID_separate=False,
                                                 convert_inBetween_zeros=False)
                    cdrs_categs['bi_match'] = cdrs_cats[bi_lid_sel]
                    
                    tf_values_dict['bi_nonmatch'] = tf_values[:, bi_lid_sel]
                    cdrs_cats = categorical_CDRS(y_full_scale=ft_cdrs['noLID'],
                                                 preLID_minutes=0,
                                                 preLID_separate=False,
                                                 convert_inBetween_zeros=False)
                    cdrs_categs['bi_nonmatch'] = cdrs_cats[bi_lid_sel]

            
            # select unilat LID itself
            if LAT_or_SCALE == 'LAT_UNI':
                uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0,
                                             ft_cdrs['noLID'] == 0)
                tf_values_dict[match_label_uni] = tf_values[:, uni_lid_sel]
            
            elif add_uni:  # change from else            
                if match_label_uni == 'match':
                    uni_lid_sel = np.logical_and(ft_cdrs['LID'] > 0, ft_cdrs['noLID'] == 0)       
                elif match_label_uni == 'nonmatch':
                    uni_lid_sel = np.logical_and(ft_cdrs['noLID'] > 0, ft_cdrs['LID'] == 0)       
                            
                if add_uni:
                    tf_values_dict[match_label_uni] = tf_values[:, uni_lid_sel]
                    # add unilateral LID for stats
                    stats_lid_pow['match'].append(tf_values[:, uni_lid_sel])
                    stats_lid_cdrs['match'].append(ft_cdrs['LID'][uni_lid_sel])  # add cdrs values for body side corr to powers
               

                    if match_label_uni == 'match':
                        cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['LID'],
                                                    preLID_minutes=0,
                                                    preLID_separate=False,
                                                    convert_inBetween_zeros=False,
                                                    return_coding_dict=True)
                    elif match_label_uni == 'nonmatch':
                        cdrs_cats, coding_dict = categorical_CDRS(y_full_scale=ft_cdrs['noLID'],
                                                    preLID_minutes=0,
                                                    preLID_separate=False,
                                                    convert_inBetween_zeros=False,
                                                    return_coding_dict=True)
                    cdrs_categs[match_label_uni] = cdrs_cats[uni_lid_sel]

            # if LATERALITY, take mean psds per side and subject,
            # and add individual mean to grand plotting
            if LAT_or_SCALE == 'LAT_UNI':
                mean_psd = np.mean(tf_values_dict[match_label_uni], axis=1)
                if BASELINE_CORRECT:
                    mean_psd = (mean_psd - bl_psd) / bl_psd * 100
                psds_to_plot[match_label_uni].append(list(mean_psd))
            
            # for SCALING, add mean-psds per category, per subject
            elif LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
                # correct bilat-psds against the baseline from the corresponding
                # hemisphere, add all bilat psds to one list
                for label in tf_values_dict.keys():

                    assert tf_values_dict[label].shape[1] == len(cdrs_categs[label]), (
                            f'NO MATCH {LAT_or_SCALE} tf_values and cdrs_categs "{sub}" '
                            f' {tf_values_dict[label].shape} vs {len(cdrs_categs[label])}'
                    )
                    # add PSDs to correct category
                    for cat in np.unique(cdrs_categs[label]):
                        cat_sel = cdrs_categs[label] == cat
                        cat_values = np.array(tf_values_dict[label])[:, cat_sel]
                        mean_cat_values = np.mean(cat_values, axis=1)
                        if BASELINE_CORRECT:
                            mean_cat_values = (mean_cat_values - bl_psd) / bl_psd * 100
                        # add subject-mean per category (add empty list if necessary)
                        if cat not in psds_to_plot[label].keys():
                            psds_to_plot[label][cat] = []
                        psds_to_plot[label][cat].append(list(mean_cat_values))

                        # keep track of individual subs included
                        try: subs_incl[label][cat].append(sub)
                        except KeyError: subs_incl[label][cat] = [sub,]
        
        # add STATS at end of subject-iteration (only MATCH)
        if PROCESS_STATS:
            temp_pows = stats_lid_pow['match'][0]
            temp_cdrs = stats_lid_cdrs['match'][0]
            for n in np.arange(1, len(stats_lid_pow['match'])):
                temp_pows = np.concatenate([temp_pows,
                                            stats_lid_pow['match'][n]],
                                            axis=1)
                temp_cdrs = np.concatenate([temp_cdrs,
                                            stats_lid_cdrs['match'][n]])
            mean_stats['LID'].append((f'{sub}_match', temp_pows))
            mean_stats['CDRS'].append((f'{sub}_match', temp_cdrs))


    # merge all contra and ipsilateral psds together for LAT_ALL_SCALE
    if n_reps == 2:
        combi_psds = {'all_match': {}, 'all_nonmatch': {}}
        for side in ['match', 'nonmatch']:
            all_cats = list(np.unique(list(psds_to_plot[side].keys()) +
                                list(psds_to_plot[f'bi_{side}'].keys())))
            for cat in all_cats:
                combi_psds[f'all_{side}'][cat] = []
            for cat in psds_to_plot[side].keys():
                combi_psds[f'all_{side}'][cat].extend(psds_to_plot[side][cat])
            for cat in psds_to_plot[f'bi_{side}'].keys():
                combi_psds[f'all_{side}'][cat].extend(psds_to_plot[f'bi_{side}'][cat])

        psds_to_plot = combi_psds

    elif PLOT_ONLY_MATCH:
        combi_psds = {'all_match': {}}
        all_cats = list(np.unique(list(psds_to_plot['match'].keys())))
        all_cats.extend(np.unique(list(psds_to_plot['bi'].keys())))
        all_cats = list(np.unique(all_cats))
        # create list for every category
        for cat in all_cats: combi_psds['all_match'][cat] = []
        for side in psds_to_plot.keys():        
            # fill cat-list with all different sides
            if 'nonmatch' in side: continue
            for cat in psds_to_plot[side].keys():
                combi_psds['all_match'][cat].extend(psds_to_plot[side][cat])
        psds_to_plot = combi_psds

    # if CALC_FREQ_CORR:
    #     print('not plotting, only returning freq corr values')
    #     return FREQ_CORRs

    # extract number of incl indiv subs per cat
    n_subs_incl = {}
    if n_reps == 1 and not PLOT_ONLY_MATCH:
        for side in subs_incl.keys():
            n_subs_incl[side] = {}
            for cat in subs_incl[side].keys():
                n_subs_incl[side][cat] = len(np.unique(subs_incl[side][cat]))
    elif n_reps == 2:
        for side in ['match', 'nonmatch']:
            if PLOT_ONLY_MATCH and side == 'nonmatch': continue
            n_subs_incl[f'all_{side}'] = {}
            for cat in psds_to_plot[f'all_{side}'].keys():
                try:
                    temp_subs = subs_incl[side][cat] + subs_incl[f'bi_{side}'][cat]
                    n_subs_incl[f'all_{side}'][cat] = len(np.unique(temp_subs))
                except:
                    try: n_subs_incl[f'all_{side}'][cat] = len(np.unique(subs_incl[side][cat]))
                    except: n_subs_incl[f'all_{side}'][cat] = len(np.unique(subs_incl[f'bi_{side}'][cat]))
    elif PLOT_ONLY_MATCH:
        n_subs_incl['all_match'] = {}
        for cat in psds_to_plot['all_match'].keys():
            try:
                temp_subs = subs_incl['match'][cat] + subs_incl['bi'][cat]
                n_subs_incl['all_match'][cat] = len(np.unique(temp_subs))
            except:
                try: n_subs_incl['all_match'][cat] = len(np.unique(subs_incl['match'][cat]))
                except: n_subs_incl['all_match'][cat] = len(np.unique(subs_incl['bi'][cat]))



    if PROCESS_STATS:
        mean_stats['freqs'] = tf_freqs
        process_mean_stats(datatype='ECOG', mean_stats=mean_stats,
                           DATA_VERSION=DATA_VERSION,
                           FT_VERSION=FT_VERSION,
                           save_stats=True,)
        get_binary_p_perHz(datatype='ECOG', save_date=p_SAVED_DATE,
                           DATA_VERSION=DATA_VERSION,
                           FT_VERSION=FT_VERSION,)


    ### PLOTTING PART
    if LAT_or_SCALE in ['SCALE', 'LAT_BILAT']:
        
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
                         fig_name=fig_name,
                         single_sub_lines=single_sub_lines,
                         PLOT_ONLY_MATCH=PLOT_ONLY_MATCH,
                         show_only_gamma=SHOW_ONLY_GAMMA,
                         n_subs_incl=n_subs_incl,
                         SHOW_SIGN=SHOW_SIGN,
                         p_SAVED_DATE=p_SAVED_DATE,
                         DATA_VERSION=DATA_VERSION,
                         FT_VERSION=FT_VERSION,
)
        

    elif LAT_or_SCALE == 'LAT_UNI':

        plot_unilateral_LID(plt_ax_to_return=plt_ax_to_return,
                            datatype='ECoG',
                            psds_to_plot=psds_to_plot,
                            tf_freqs=tf_freqs,
                            n_uni_subs=n_uni_subs,
                            BASELINE_CORRECT=BASELINE_CORRECT,
                            LOG_POWER=LOG_POWER,
                            SMOOTH_PLOT_FREQS=SMOOTH_PLOT_FREQS,
                            STD_ERR=STD_ERR,
                            BREAK_X_AX=BREAK_X_AX,
                            fsize=fsize,
                            single_sub_lines=single_sub_lines,
                            show_only_gamma=SHOW_ONLY_GAMMA,)

