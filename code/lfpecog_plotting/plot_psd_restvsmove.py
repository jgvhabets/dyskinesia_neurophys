"""
Plots PSDs that are millisecond precise
on movement selection
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, compress
from pandas import read_csv, DataFrame

from utils.utils_fileManagement import get_project_path
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side
)
from lfpecog_plotting.plot_descriptive_SSD_PSDs import break_x_axis_psds_ticks
from lfpecog_analysis.specific_ephys_selection import get_hemisphere_movement_location
from lfpecog_analysis.prep_stats_movLidspecPsd import (
    get_REST_stat_grouped_data
)
from lfpecog_analysis.psd_lid_stats import calc_lmem_freqCoeffs, run_mixEff_wGroups


cond_colors = {
    'nolid': 'green',
    'nolidbelow30': 'limegreen',
    'nolidover30': 'darkgreen',
    'alllid': 'blue', 'mildlid': 'orange',
    'moderatelid': 'red', 'severelid': 'purple',
}


def prep_and_plot_restvsmove(
    PSD_DICT, BASELINE, SOURCE,
    FEATURE: str = 'PSD',
    MOVESIDES_SPLITTED: bool = False,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    INCL_STATS: bool = True,
    STAT_PER_LID_CAT: bool = True,
    ALPHA: float = 0.01,
    REST_u30_BASELINE: bool = True,
    STATS_VERSION: str = '2Hz',
    STAT_LID_COMPARE: str = 'categs',
    ADD_TO_FIG_NAME = False,
):
    # check feature and source input
    if FEATURE.upper() == 'PSD':
        assert SOURCE in ['lfp', 'ecog'], f'incorrect source ({SOURCE}) for {FEATURE}'
    elif FEATURE.upper() in ['ICOH', 'SQCOH']:
        assert SOURCE in ['STNECOG', 'STNs'], f'incorrect source ({SOURCE}) for {FEATURE}'
    else:
        raise ValueError('FEATURE should be PSD / ICOH / SQCOH')
    
    # prep figure
    n_rows, n_cols = 1, 2
    kw_params = {'sharey': 'row'}
    if MOVESIDES_SPLITTED == True:
        n_cols += 1
    
    elif MOVESIDES_SPLITTED == '4panel':
        n_cols, n_rows = 2, 2
        kw_params['sharex'] = 'col'

    
    if FEATURE == 'PSD': YLIM = (-60, 175)
    else: YLIM = (-50, 100)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols,
                                                      4*n_rows),
                             **kw_params)
    
    if MOVESIDES_SPLITTED == '4panel':
        axes = axes.flatten()
    
    for i_ax, ax in enumerate(axes):
        # define correct prep function
        if i_ax == 0: PLOT_MOVE = 'REST'
        
        elif MOVESIDES_SPLITTED == True and i_ax == 1: PLOT_MOVE = 'CONTRA'

        elif MOVESIDES_SPLITTED == True and i_ax == 2: PLOT_MOVE = 'IPSI'

        elif MOVESIDES_SPLITTED == '4panel':
            if i_ax == 1: PLOT_MOVE = 'ALLMOVE'
            if i_ax == 2: PLOT_MOVE = 'CONTRA'
            if i_ax == 3: PLOT_MOVE = 'IPSI'

        else: PLOT_MOVE = 'ALLMOVE'
   
        # execute prep function
        print(f'\n\n###### PREP {PLOT_MOVE}, {SOURCE}')
        psd_arrs, psd_freqs, psd_subs = prep_RestVsMove_psds(
            SRC=SOURCE, FEATURE=FEATURE,
            PLOT_MOVE=PLOT_MOVE,
            PSD_DICT=PSD_DICT,
            BASELINE=BASELINE,
            SPLIT_CONTRA_IPSI=False,
            RETURN_IDS=True,
        )

        if INCL_STATS:
            if PLOT_MOVE == 'REST' and REST_u30_BASELINE:
                STAT_BL_epochs = list(psd_arrs.values())[0]
                STAT_BL_subs = list(psd_subs.values())[0]

            stat_df = get_restMove_stats(
                SOURCE=SOURCE, FEATURE=FEATURE,
                MOVE_TYPE=PLOT_MOVE,
                STAT_BL_epochs=STAT_BL_epochs,
                STAT_BL_subs=STAT_BL_subs,
                epoch_values=psd_arrs,
                epoch_ids=psd_subs,
                epoch_freqs=psd_freqs,
                STATS_VERSION=STATS_VERSION,
                STAT_LID_COMPARE=STAT_LID_COMPARE,
                ALPHA=ALPHA,
            )
        else:
            stat_df = False

        # plot AX
        print(f'\n....PLOT {PLOT_MOVE}')
        plot_moveLidSpec_PSDs(
            psd_arrs, psd_freqs=psd_freqs, psd_subs=psd_subs,
            SOURCE=SOURCE, FEATURE=FEATURE,
            AX=ax, PLOT_MOVE_TYPE=PLOT_MOVE,
            stat_df=stat_df,
            INCL_STATS=INCL_STATS,
            STAT_PER_LID_CAT=STAT_PER_LID_CAT,
            YLIM=YLIM,
            )
        if MOVESIDES_SPLITTED == '4panel' and i_ax in [1, 3]:
            y_ax = ax.axes.get_yaxis()
            ylab = y_ax.get_label()
            ylab.set_visible(False)
    
    plt.tight_layout()

    if SAVE_PLOT:
        FIG_NAME = f'{SOURCE}_{FEATURE}_restVsMove'
        if MOVESIDES_SPLITTED == True: FIG_NAME += 'Split'
        elif MOVESIDES_SPLITTED == '4panel': FIG_NAME += '4Panel'

        if INCL_STATS:
            FIG_NAME += f'_stats{STAT_LID_COMPARE}'
            if len(STATS_VERSION) > 1: FIG_NAME += STATS_VERSION
        if ADD_TO_FIG_NAME:
            FIG_NAME = ADD_TO_FIG_NAME + FIG_NAME
        FIG_PATH = os.path.join(get_project_path('figures'),
                                'final_Q1_2024',
                                'moveSpecPsds')
        
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)
        print(f'saved plot {FIG_NAME} in {FIG_PATH}!')

    if SHOW_PLOT: plt.show()
    else: plt.close()


def plot_moveLidSpec_PSDs(
    psd_arrs, psd_freqs, psd_subs,
    PLOT_MOVE_TYPE, SOURCE, stat_df,
    AX=None, FEATURE: str = 'PSD',
    STATS_VERSION: str = '4Hz',
    PLOT_ALL_SUBS: bool = False,
    INCL_STATS: bool = False,
    STAT_PER_LID_CAT: bool = True,
    YLIM: tuple = (-75, 200),
):
    """
    Plot either rest, or movement on single axis
    """
    print(f'######### START PLOTTING {PLOT_MOVE_TYPE}')

    XTICKS = [4, 12, 20, 28, 34, 60, 70, 80, 89]  # 89 will be labeled 90
    ls = 'solid'
    FS = 16  # fontsize
    
    # LOOP OVER LID-CATEGORIES/ GROUPS
    for lid in psd_arrs.keys():
        # get and average correct PSDs (subject means)
        uniq_subs_cat = np.unique(psd_subs[lid])
        n_subs = len(uniq_subs_cat)
        if n_subs == 0: continue
        
        sub_means = []
        for s in uniq_subs_cat:
            sub_sel = np.array(psd_subs[lid]) == s
            sub_epochs = list(compress(psd_arrs[lid], sub_sel))
            sub_epochs = np.array([row for arr in sub_epochs for row in arr])
            sub_means.append(np.mean(sub_epochs, axis=0))

        sub_means = np.array(sub_means)
        
        # PLOT GROUP MEAN, mean of sub-means
        if not PLOT_ALL_SUBS:
            m = np.mean(sub_means, axis=0)
            sd = np.std(sub_means, axis=0)
            sem = np.std(sub_means, axis=0) / np.sqrt(n_subs)
            
            # break and nan-pad x-axis
            m, sem, x_plot, xlabs = break_x_axis_psds_ticks(
                tf_freqs=psd_freqs,
                PSD=m, PSD_sd=sem,
                x_break = (35, 60), nan_pad = 5
            )
            lab = lid.replace('lid', ' LID')  #  add spaces for readability
            lab = lab.replace('below30', ' < 30min')
            lab = lab.replace('over30', ' > 30min')
            # plot mean line of LID severity
            AX.plot(x_plot, m,
                    color=cond_colors[lid], alpha=.8, ls=ls,
                    label=f"{lab} (n={n_subs})",)
            # plot variance shades (LID severity)
            AX.fill_between(x_plot, y1=m-sem, y2=m+sem, # PSDs[cond].freqs, 
                            color=cond_colors[lid], alpha=.3,)

        elif PLOT_ALL_SUBS:
            for i_m, m in enumerate(sub_means):
                # break and nan-pad x-axis
                m, x_plot, xlabs = break_x_axis_psds_ticks(
                    tf_freqs=psd_freqs, PSD=m,
                    x_break = (35, 60), nan_pad = 5
                )
                lab = lid.replace('lid', ' LID')  #  add spaces for readability
                lab = lab.replace('below30', ' < 30min')
                lab = lab.replace('over30', ' > 30min')
                # plot mean line of LID severity
                if i_m == 0:
                    AX.plot(x_plot, m, color=cond_colors[lid], lw=.5,
                            alpha=.8, label=f"{lab} (n={n_subs})",)
                else:
                    AX.plot(x_plot, m, color=cond_colors[lid], alpha=.8, lw=.5,)

    # plot significancies shades (once per AX)
    if INCL_STATS and STAT_PER_LID_CAT:
        if FEATURE == 'PSD': y_legend_gap = 60
        else: y_legend_gap = 35
        plot_stats_categs(stat_df=stat_df, ax=AX,
                          PLOT_MOVE_TYPE=PLOT_MOVE_TYPE,
                          Y_BASE=YLIM[1] - y_legend_gap, lw=5,
                          STATS_VERSION=STATS_VERSION,)

    # add title (once per AX)
    src_title = SOURCE.replace('lfp', 'stn')
    if PLOT_MOVE_TYPE == 'REST':
        ax_title = (f'{src_title.upper()} changes: Rest')
    else:
        ax_title = (f'{src_title.upper()} changes: Movement')
    ax_title = ax_title.replace('STNS', 'Inter-Subthalamic')
    ax_title = ax_title.replace('STNECOG', 'Cortico-Subthalamic')
    if PLOT_MOVE_TYPE in ['IPSI', 'CONTRA']:
        ax_title = ax_title.replace(
            'Movement',
            f'{PLOT_MOVE_TYPE.capitalize()}lateral Movement'
        )
    AX.set_title(ax_title, weight='bold', size=FS,)


    xtick_sel = np.where([x in XTICKS for x in xlabs])[0]
    xticks = x_plot[xtick_sel]
    xlabs = np.array(xlabs)[xtick_sel]
    xlabs[-1] = 90  # mark last tick with 90 (instead of 89)

    AX.axhline(0, xmin=0, xmax=1, color='gray', alpha=.3,)
    AX.set_xticks(xticks)
    AX.set_xticklabels(xlabs, fontsize=FS,)
    AX.set_ylim(YLIM)
    AX.legend(fontsize=FS - 6, frameon=False, ncol=2,
              bbox_to_anchor=(.5, .99), loc='upper center')
    AX.tick_params(size=FS, labelsize=FS,)
    for s in ['right', 'top']: AX.spines[s].set_visible(False)
    # AX.text(x=xticks[len(xticks) // 2] + 2, y=-73, s='//', size=16, color='k')

    # set axis labels
    # rf"$\bf{SOURCE_SEL.upper()}$" + " " +
    AX.set_xlabel('Frequency (Hz)', size=FS, weight='bold',)
    if FEATURE == 'PSD':
        ylab = rf"$\bfSpectral$" + " " + rf"$\bfPower$"  # + "\n(% vs Med-OFF)"
    elif FEATURE == 'ICOH':
        ylab = rf"$\bfabs.$" + " " + rf"$\bfimag.$" + " " + rf"$\bfCoherence$"
    elif FEATURE == 'SQCOH':
        ylab = rf"$\bfsquared$" + " " + rf"$\bfCoherence$"    
    ylab += "\n(% change vs Med-OFF)"
    AX.set_ylabel(ylab, size=FS, )




def prep_RestVsMove_psds(SRC, PLOT_MOVE, PSD_DICT, BASELINE,
                         FEATURE: str = 'PSD',
                         RETURN_IDS: bool = False,
                         RETURN_STATE_ARRAYS: bool = False,
                         SPLIT_CONTRA_IPSI: bool = False,):
    """

    Arguments:
        - SRC: lfp or ecog
        - PLOT_MOVE: TAP or INVONUT
        - PSD_DICT: dict with conditions as keys and containing
            get_selectedEphys() classes per condition (from
            psd_analysis_classes.py)
        - BASELINE: get_selectedEphys() class
        - MERGE_REST_STNS: to keep arguments parallel with rest

    Returns:
        - psd_arrs: dicts of lists with selected and processed
            psd-arrays, organized in LFP/ECoG (axrows), and
            CONTRA/IPSI-lateral movements (axcols)
    """
    print(f'start prep MOVEMENT spec psdsPLOTMOVE: {PLOT_MOVE}')
    assert PLOT_MOVE in ['REST', 'CONTRA', 'IPSI', 'ALLMOVE']

    if PLOT_MOVE == 'ALLMOVE':
        ATTR_CODE = ['DYSKMOVE', 'TAP']
        MOVE_SPLIT = 'ALL'
    elif PLOT_MOVE == 'CONTRA':
        ATTR_CODE = ['DYSKMOVE', 'TAP']
        MOVE_SPLIT = 'CONTRA'
    elif PLOT_MOVE == 'IPSI':
        ATTR_CODE = ['DYSKMOVE', 'TAP']
        MOVE_SPLIT = 'IPSI'
    elif PLOT_MOVE == 'REST':
        ATTR_CODE = ['REST',]
        MOVE_SPLIT = 'ALL'

    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['nolidbelow30', 'nolidover30', 'nolid',
                  'mildlid', 'moderatelid', 'severelid']
    
    # create dicts to store psds
    psd_arrs = {l: [] for l in lid_states}

    # store IDs parallel to psds
    if RETURN_IDS: sub_arrs = {l: [] for l in lid_states}
    
    # categorize, average, baseline-corr all conditions
    for MOVE, cond in product(ATTR_CODE, PSD_DICT.keys()):
        print(f'\n\t- {MOVE}, {cond} ({SRC})')
        if not MOVE.lower() in cond.lower(): continue
        # loop and add subjects
        for attr in vars(PSD_DICT[cond]).keys():
            # skip irelevant attr
            if not MOVE.lower() in attr.lower() or SRC not in attr: continue
            print(f'...selected attr for {PLOT_MOVE}: {attr}')
            
            # define SUB
            if 'lfp' in SRC: sub = attr.split('_')[2]  # exclude lfp side PSD
            elif SRC == 'ecog': sub = attr.split('_')[1]  # exclude lfp side PSD
            else: sub = attr.split('_')[1]  # extract sub from Coherences
            if not sub.startswith('0') and not sub.startswith('1'): continue

            # define EPHYS-side
            if 'lfp' in attr: EPHYS_SIDE = attr.split('_')[1]
            elif 'ecog' in attr: EPHYS_SIDE = PSD_DICT[cond].ecog_sides[sub]

            # for contra/ ipsi, split movements (not for REST and ALLMOVE)
            if MOVE_SPLIT != 'ALL' and FEATURE == 'PSD':
                if any(['tapleft' in attr, 'leftonly' in attr]):
                    MOVE_SIDE = 'left'
                elif any(['tapright' in attr, 'rightonly' in attr]):
                    MOVE_SIDE = 'right'
                else:
                    f'...no UNI side movement, skip {attr} for {PLOT_MOVE}'
                    continue
                if MOVE_SIDE == EPHYS_SIDE: MOVE_LAT = 'IPSI'
                else: MOVE_LAT = 'CONTRA'
                if MOVE_LAT != MOVE_SPLIT:
                    # print(f'...skip {attr}, movement is {MOVE_LAT}, instead of {MOVE_SPLIT}')
                    continue
                # print(f'...INCLUDE {attr} for {MOVE_SPLIT}')

            # get psd arr and correct for baseline
            temp_psd = getattr(PSD_DICT[cond], attr)
            print(f'...SHAPE temp psd {temp_psd.shape}')
            
            try:
                if SRC == 'ecog': bl_attr = f'ecog_{sub}_baseline'
                elif SRC == 'lfp': bl_attr = f'lfp_{EPHYS_SIDE}_{sub}_baseline'
                else: bl_attr = f'{SRC}_{sub}_baseline'

                bl = getattr(BASELINE, bl_attr)
                if len(bl.shape) == 2: bl = bl.mean(axis=0)
            except:
                print(f'### WARNING no baseline {SRC, EPHYS_SIDE} sub {sub}')
                continue

            temp_psd = ((temp_psd - bl) / bl) * 100
            if RETURN_STATE_ARRAYS: print(f'...arr shape {attr}: {temp_psd.shape}')

            # include LID-state:
            temp_lid = attr.split('_')[-1]

            psd_arrs[temp_lid].append(temp_psd)
            # add subject ID to list
            if RETURN_IDS: sub_arrs[temp_lid].append(sub)
            
                        
    psd_freqs = PSD_DICT[cond].freqs

    for l in lid_states:
        if len(psd_arrs[l]) == 0:
            print(f'remove psd/sub-arrs {l}')
            del(psd_arrs[l])
            del(sub_arrs[l])

    if not RETURN_IDS: return psd_arrs, psd_freqs

    elif RETURN_IDS: return psd_arrs, psd_freqs, sub_arrs


def get_restMove_stats(
    SOURCE, MOVE_TYPE, FEATURE,
    epoch_values, epoch_ids, epoch_freqs,
    STAT_BL_epochs,
    STAT_BL_subs,
    STATS_VERSION='2Hz',
    STAT_LID_COMPARE='categs',
    ALPHA=.01,
    REST_u30_BASELINE: bool = True,
):
    # prevent circular import
    from lfpecog_analysis.prep_stats_movLidspecPsd import get_stat_folder

    stat_dir = get_stat_folder(STAT_LID_COMPARE=STAT_LID_COMPARE,
                               STAT_DATA_EXT_PATH=True,
                               STATS_VERSION=STATS_VERSION,
                               ALPHA=ALPHA,)
    
    # define naming of stats file
    dfname = f'restVsMove_1secWins_{MOVE_TYPE.upper()}_{SOURCE.lower()}.csv'  # mov is source
    if FEATURE.lower() != 'psd':
        dfname = dfname.replace('.csv', f'_{FEATURE.lower()}.csv')

    if REST_u30_BASELINE:
        dfname = 'u30_' + dfname

    # try to load    
    if os.path.exists(os.path.join(stat_dir, dfname)):
        print(f'...stats df loading: {dfname}')
        stat_df = read_csv(os.path.join(stat_dir, dfname),
                           header=0, index_col=0)
        print(f'LOADED DF keys: {stat_df.keys()}')
        return stat_df
    
    else:
        print(f'STAT DF not existing ({dfname}), CREATE...')
    

    print('...load rest psds as baseline for movement')

    if MOVE_TYPE == 'REST' and len(epoch_values) == 5:
        lidlabels = list(epoch_values.keys())[1:]
        epoch_values = list(epoch_values.values())[1:]
        epoch_ids = list(epoch_ids.values())[1:]
    
    else:
        lidlabels = list(epoch_values.keys())
        epoch_values = list(epoch_values.values())
        epoch_ids = list(epoch_ids.values())

    print(f'merge epoch values for LMM, lidlabels: {lidlabels}')

    # define baseline values (under30, no LID, rest)
    BL_values = np.array([row for l in STAT_BL_epochs for row in l])
    BL_subs = []
    for og_values, og_sub in zip(STAT_BL_epochs, STAT_BL_subs):
        BL_subs.extend([og_sub] * len(og_values))
    BL_subs = np.array(BL_subs)
    BL_labels = np.array([0] * len(BL_values))
    assert len(BL_labels) == len(BL_subs), (
        f'baseline subs ({len(BL_subs)}) and labels ({len(BL_labels)}) mismatch'
    )

    stat_df = DataFrame(index=epoch_freqs)

    stat_values, stat_labels, stat_ids = {}, {}, {}

    for i, values in enumerate(epoch_values):
        # empty list to store lid-category
        stat_coefs, stat_pvals = [], []
        cat_subs = []  # add sub id for every epoch row in category data
        for s, l in zip(epoch_ids[i], values): cat_subs.extend([s] * len(l))
        values = np.array([row for l in values for row in l])
        stat_values[i] = np.concatenate([values, BL_values], axis=0)
        stat_labels[i] = np.concatenate([[1] * len(values), BL_labels])  # binary comparison per category
        stat_ids[i] = np.concatenate([cat_subs, BL_subs])

        print(f'STAT DATA {lidlabels[i]}: {stat_values[i].shape}, {stat_labels[i].shape}, {stat_ids[i].shape}')
    
        # CALCULATE LMM COEFFS and SIGN
        # get STATS (coeffs, sign-bools) based on grouped data
        if STATS_VERSION == '2Hz': f_hop = 2
        elif STATS_VERSION == '4Hz': f_hop = 4
        
        start_fs = np.concatenate(
            [np.arange(4, 35, f_hop), np.arange(60, 90, f_hop)]
        )
        MULTI_COMP_N = 0
        for f in start_fs:
            MULTI_COMP_N += 1
            f_sel = np.logical_and(epoch_freqs >= f,
                                   epoch_freqs < (f + f_hop))
            f_values = np.mean(stat_values[i][:, f_sel], axis=1)
            (coeff, pval) = run_mixEff_wGroups(
                dep_var=f_values,
                indep_var=stat_labels[i],
                groups=stat_ids[i],
                TO_ZSCORE=False,
            )
            # add to lists
            stat_coefs.extend([coeff] * sum(f_sel))  # add for n involved freqs
            stat_pvals.extend([pval] * sum(f_sel))
            print(f'...added {f} - {f + f_hop} Hz (n={sum(f_sel)})')
        assert len(stat_pvals) == len(epoch_freqs), (
            f'n pvalues mismatch with n freqs ({len(stat_pvals), len(epoch_freqs)})'
        )
        print(f'SUCCESSFUL STAT ADDING, N multi-comp: {MULTI_COMP_N}')
        
        # add all freqs to df
        stat_df[f'{lidlabels[i]}_u30_coef'] = stat_coefs
        stat_df[f'{lidlabels[i]}_u30_pval'] = stat_pvals
    
    stat_df.to_csv(os.path.join(stat_dir, dfname),)
    
    return stat_df


def plot_stats_categs(
    stat_df, ax, PLOT_MOVE_TYPE,
    lw=8, ALPHA=.01, basegroup: str = 'u30',
    STATS_VERSION: str = '4Hz', Y_BASE: int = 150,
):
    if STATS_VERSION == '4Hz': N_MULTI_COMP = 16
    elif STATS_VERSION == '2Hz': N_MULTI_COMP = 32
    ALPHA /= N_MULTI_COMP
    print(f'multi-comparison corrected ALPHA = {ALPHA}')
    # get groups
    lid_cats = np.unique([k.split('_')[0] for k in stat_df.keys()])
    lid_cats_sort = [c for c in cond_colors.keys() if c in lid_cats]

    for i_cat, cat in enumerate(lid_cats_sort):  # loop over CATEGs to show
        # extract pvalues (and coeffs)
        sign_bools = stat_df[f'{cat}_{basegroup}_pval'].values < ALPHA
        sign_bools = sign_bools.astype(float)
        
        # break and nan-pad x-axis
        sign_bools, sign_x, _ = break_x_axis_psds_ticks(
            tf_freqs=stat_df.index.values,
            PSD=sign_bools,
            x_break = (35, 60), nan_pad = 5
        )
        # first plot baseline for color background
        ax.fill_between(
            sign_x, y1=Y_BASE - ((i_cat + 1) * lw),
            y2=Y_BASE - (i_cat * lw),
            color=cond_colors[cat], alpha=.1,
        )
        # plot significancies
        ax.fill_between(
            sign_x, where=sign_bools == 1,
            y1=Y_BASE - ((i_cat + 1) * lw),
            y2=Y_BASE - (i_cat * lw),
            color=cond_colors[cat], alpha=.6,
        )