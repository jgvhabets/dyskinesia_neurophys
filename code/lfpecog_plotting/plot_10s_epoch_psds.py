"""
Plots PSDs over 10-second epochs, no
high-res movement selection
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from lfpecog_plotting.plot_psd_restvsmove import (
    plot_moveLidSpec_PSDs, get_restMove_stats
)
from utils.utils_fileManagement import (
    get_project_path, get_avail_ssd_subs
)
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side, get_cdrs_specific
)
from lfpecog_analysis.get_SSD_timefreqs import (
    get_all_ssd_timeFreqs
)
from lfpecog_analysis.psd_analysis_classes import (
    get_selectedEphys
)



cond_colors = {
    'nolid': 'green',
    'nolidbelow30': 'limegreen',
    'nolidover30': 'darkgreen',
    'alllid': 'blue', 'mildlid': 'orange',
    'moderatelid': 'red', 'severelid': 'purple',
}


def plot_overall_PSD_COH(
    psd_arr_dict, sub_arr_dict, ps_freqs_dict,
    BASE_METHOD: str = 'Z',     
    SAVE_PLOT = True,
    FIG_DATE = '0311',
    SHOW_PLOT = False,
    INCL_STATS = False,
    PEAKSHIFT_GAMMA = True,
    SMOOTH_WIN: int = 0,
    ALPHA = .01,
):
    """
    Aeguments:
        - psd_arrs, sub_arrs, ps_freqs: all dict containing
            lfp and ecog from def prep_10sPSDs_lidCategs()
    """
    DATA_VERSION='v4.0'
    FT_VERSION='v6'
    IGNORE_PTS=['011', '104', '106']

    SUBS = get_avail_ssd_subs(DATA_VERSION=DATA_VERSION,
                              FT_VERSION=FT_VERSION,
                              IGNORE_PTS=IGNORE_PTS)

    n_rows, n_cols = 2, 2
    kw_params = {'sharey': 'row'}

    if BASE_METHOD == 'percChange': YLIM = (-50, 100)
    elif BASE_METHOD == 'Z': YLIM = (-.5, 1.5)
    elif BASE_METHOD == 'no': YLIM = (0, .1)


    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(6*n_cols, 4*n_rows),
                            **kw_params)
    axes = axes.flatten()

    for i_ax, (ft, src, ax) in enumerate(zip(
        ['POWER', 'POWER', 'sqCOH', 'sqCOH'],
        ['lfp', 'ecog', 'STNs', 'STNECOG'],
        axes
    )):
        if ft == 'POWER':
            # use priorly created arrays with  10-sec PREP has to lead to psd_arrs, psd_freqs, psd_subs
            psd_arr = psd_arr_dict[src]
            sub_arr = sub_arr_dict[src]
            ps_freqs = ps_freqs_dict[src]
            
        
        elif ft == 'sqCOH':
            # get timefreq values and baseline for coherences
            COH_TFs = get_all_ssd_timeFreqs(
                SUBS=SUBS, COH_SOURCE=src,
                FEATURE=ft,
                FT_VERSION=FT_VERSION,
                DATA_VERSION=DATA_VERSION,
                GET_CONNECTIVITY=False,  # 'trgc'
            )
            coh_BLs = get_selectedEphys(
                FEATURE=f'COH_{src}',
                COH_TYPE=ft,
                STATE_SEL='baseline',
                MIN_SEL_LENGTH=10,
                LOAD_PICKLE=True,
                USE_EXT_HD=True,
                PREVENT_NEW_CREATION=False,
                RETURN_PSD_1sec=True,
                verbose=False,
            )
            # convert into arrays for plot
            (psd_arr,
             ps_freqs,
             sub_arr) = prep_10sPSDs_lidCategs(
                TFs=COH_TFs, SOURCE=src,
                FEATURE=ft,
                BASELINE=coh_BLs,
                IPSI_CONTRA_UNILID=False,
                BASE_METHOD=BASE_METHOD,
            )
        
        # ### 10-sec PREP has to lead to psd_arrs, psd_freqs, psd_subs
        for arr in [psd_arr, sub_arr]:
            if 'nolid' in list(arr.keys()):
                del(arr['nolid'])  # leave only 30min no lids in

        if INCL_STATS:
            # take all nolid as baseline
            STAT_BL_epochs = list(psd_arr.values())[:2]
            STAT_BL_epochs = np.array([row for l in STAT_BL_epochs for row in l])
            STAT_BL_subs = list(sub_arr.values())[:2]
            STAT_BL_subs = np.array([s for l in STAT_BL_subs for s in l])
            # load or calculate stats
            stat_df = get_restMove_stats(
                SOURCE=src, FEATURE=ft,
                MOVE_TYPE='10sec',
                STAT_BL_epochs=STAT_BL_epochs,
                STAT_BL_subs=STAT_BL_subs,
                epoch_values=psd_arr,
                epoch_ids=sub_arr,
                epoch_freqs=ps_freqs,
                STATS_VERSION='4Hz',
                STAT_LID_COMPARE='categs',
                ALPHA=ALPHA,
                REST_u30_BASELINE=False,
            )
        


        plot_moveLidSpec_PSDs(
            psd_arrs=psd_arr.copy(),
            psd_freqs=ps_freqs.copy(),
            psd_subs=sub_arr.copy(),
            FEATURE=ft,
            SOURCE=src,
            PLOT_MOVE_TYPE='overall',
            AX=ax,
            INCL_STATS=INCL_STATS,
            stat_df=stat_df,
            YLIM=YLIM,
            PEAK_SHIFT_GAMMA=PEAKSHIFT_GAMMA,
            SMOOTH_WIN=SMOOTH_WIN,
            # MIN_SUBS_FOR_MEAN=4,
        )

        if i_ax in [1, 3]:
            y_ax = axes[1].axes.get_yaxis()
            ylab = y_ax.get_label()
            ylab.set_visible(False)
        
    plt.tight_layout()

    if SAVE_PLOT:
        FIG_NAME = f'{FIG_DATE}_10secPSDsCOHs'
        if BASE_METHOD == 'Z': FIG_NAME += '_offZ'
        if BASE_METHOD == 'percChange': FIG_NAME += '_percChange'
        if PEAKSHIFT_GAMMA: FIG_NAME += '_gammaShift'
        if SMOOTH_WIN > 0: FIG_NAME += f'_sm{SMOOTH_WIN}'
        if INCL_STATS:
            FIG_NAME += '_inclStats'
        FIG_PATH = os.path.join(get_project_path('figures'),
                                'final_Q1_2024',
                                'overall_PSDs')
        
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)
        print(f'saved plot {FIG_NAME} in {FIG_PATH}!')

    if SHOW_PLOT: plt.show()
    else: plt.close()


def plot_unilatLID_PSD_10s(
    TFs, BLs,
    SAVE_PLOT = True,
    FIG_DATE = '000',
    BASE_METHOD: str = 'percChange',
    LID_BINARY: bool = False,
    SHOW_PLOT = False,
    INCL_STATS = False,
    SMOOTH_WIN: int = 0,
    PEAKSHIFT_GAMMA: bool = True,
):
    """
    Plot ipsi and contra lateral changes for unilateral
    LID, based on 10s epochs

    Arguments:
        - TFs: timefreq values from
            get_SSD_timefreqs.get_all_ssd_timeFreqs()
        - BLs: baseline values from
            psd_analysis_classes.get_selectedEphys()
    
    """
    subs = TFs.keys()
    unilid_subs = get_unilat_lid_timesscores(subs)

    n_rows, n_cols = 1, 2
    kw_params = {'sharey': 'row'}

    if BASE_METHOD == 'percChange' and not LID_BINARY: YLIM = (-60, 175)
    elif BASE_METHOD == 'percChange' and LID_BINARY: YLIM = (-35, 75)
    elif BASE_METHOD == 'Z': YLIM = (-.5, 1.5)

    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(6*n_cols, 4*n_rows),
                            **kw_params)

    for src, ax in zip(['lfp', 'ecog'], axes):

        ### 10-sec PREP has to lead to psd_arrs, psd_freqs, psd_subs
        (psd_arrs,
         ps_freqs,
         sub_arrs) = prep_10sPSDs_lidCategs(
            TFs=TFs, SOURCE=src,
            BASELINE=BLs,
            IPSI_CONTRA_UNILID=True,
            unilid_subs=unilid_subs,
            BASE_METHOD=BASE_METHOD,
        )

        if INCL_STATS or LID_BINARY:
            # split data in two groups for stats: ipsi vs contra
            ipsi_arr = {k: v for k,v in psd_arrs.items() if 'ipsi' in k}
            ipsi_arr = {
                'alllid': np.concatenate([arr for arr in ipsi_arr.values()
                                          if len(arr.shape) == 2], axis=0)}
            contra_arr = {k: v for k,v in psd_arrs.items() if 'contra' in k}
            contra_arr = {
                'alllid': np.concatenate([arr for arr in contra_arr.values()
                                          if len(arr.shape) == 2], axis=0)}
            ipsi_subs = {k: v for k,v in sub_arrs.items() if 'ipsi' in k}
            ipsi_subs = {'alllid': np.concatenate([arr for arr in ipsi_subs.values()
                                                   if len(arr) > 0], axis=0)}
            contra_subs = {k: v for k,v in sub_arrs.items() if 'contra' in k}
            contra_subs = {'alllid': np.concatenate([arr for arr in contra_subs.values()
                                                    if len(arr) > 0], axis=0)}
            
            if LID_BINARY:
                psd_arrs = {'ipsi_alllid': ipsi_arr['alllid'],
                            'contra_alllid': contra_arr['alllid']}
                sub_arrs = {'ipsi_alllid': ipsi_subs['alllid'],
                            'contra_alllid': contra_subs['alllid']}
            
            if INCL_STATS:
                # load or calculate stats
                stat_df = get_restMove_stats(
                    SOURCE=src, FEATURE='psd',
                    MOVE_TYPE='10secUnilat',
                    STAT_BL_epochs=ipsi_arr,
                    STAT_BL_subs=ipsi_subs,
                    epoch_values=contra_arr,
                    epoch_ids=contra_subs,
                    epoch_freqs=ps_freqs,
                    STATS_VERSION='4Hz',
                    STAT_LID_COMPARE='categs',
                    ALPHA=.01,
                    REST_u30_BASELINE=False,
                )
            else:
                stat_df = False

        else:
            stat_df = False

        plot_moveLidSpec_PSDs(
            psd_arrs=psd_arrs,
            psd_freqs=ps_freqs,
            psd_subs=sub_arrs,
            SOURCE=src,
            PLOT_MOVE_TYPE='unilatLID',
            LID_BINARY=LID_BINARY,
            AX=ax,
            YLIM=YLIM,
            INCL_STATS=INCL_STATS,
            stat_df=stat_df,
            PEAK_SHIFT_GAMMA=PEAKSHIFT_GAMMA,
            SMOOTH_WIN=SMOOTH_WIN,
        )

    y_ax = axes[1].axes.get_yaxis()
    ylab = y_ax.get_label()
    ylab.set_visible(False)
        
    plt.tight_layout()

    if SAVE_PLOT:
        FIG_NAME = f'{FIG_DATE}_Lateral_unilatLID_stnecog'
        if LID_BINARY: FIG_NAME += 'BINARY'
        if BASE_METHOD == 'Z': FIG_NAME += '_offZ'
        if BASE_METHOD == 'percChange': FIG_NAME += '_percChange'
        if PEAKSHIFT_GAMMA: FIG_NAME += '_gammaShift'
        if SMOOTH_WIN > 0: FIG_NAME += f'_sm{SMOOTH_WIN}'
        if INCL_STATS: FIG_NAME += '_inclStats'
        FIG_PATH = os.path.join(get_project_path('figures'),
                                'final_Q1_2024',
                                'unilatLID_lateralization')
        
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)
        print(f'saved plot {FIG_NAME} in {FIG_PATH}!')

    if SHOW_PLOT: plt.show()
    else: plt.close()


def get_unilat_lid_timesscores(subs):

    unilid_subs = {}

    for sub in subs:
        scores_l = get_cdrs_specific(
            sub=sub, INCL_CORE_CDRS=False, side='left'
        )[1]
        scores_r = get_cdrs_specific(
            sub=sub, INCL_CORE_CDRS=False, side='right'
        )[1]
        l_lid = np.logical_and(scores_l > 0, scores_r == 0)
        r_lid = np.logical_and(scores_r > 0, scores_l == 0)
    
        if any(l_lid):
            lidside = 'left'
            otherscores = scores_r
        elif any(r_lid):
            lidside = 'right'
            otherscores = scores_l
        else: continue

        ts, lids = get_cdrs_specific(
            sub=sub, INCL_CORE_CDRS=True, side=lidside
        )
        unilid_subs[f'{sub}_{lidside}'] = (ts, lids, otherscores)

    return unilid_subs


def prep_10sPSDs_lidCategs(
    SOURCE, TFs, BASELINE,
    FEATURE: str = 'POWER',
    IPSI_CONTRA_UNILID: bool = False,
    unilid_subs=None,
    BASE_METHOD: str = 'Z'
):
    """
    
    Parameters:
        - SOURCE: either lfp or ecog
        - TFs: dicts with combined SSDd PSDs
        - BASELINE: class with baseline PSDs from
            psd_analysis_classes.get_selectedEphys()
    """
    if FEATURE == 'POWER':
        sources = ['lfp_left', 'lfp_right', 'ecog']
    else:
        sources = ['STNs', 'STNECOG']
        # print(f'START, {FEATURE}, {sources}')
    lid_states = ['nolidbelow30', 'nolidover30', 'nolid',
                  'mildlid', 'moderatelid', 'severelid']
    lid_categ_ranges = {'nolid': [0, 0],
                        'mildlid': [1, 3],
                        'moderatelid': [4, 7],
                        'severelid': [8, 20]}

    psd_arrs = {l: [] for l in lid_states}
    sub_arrs = {l: [] for l in lid_states}

    if IPSI_CONTRA_UNILID:
        psd_arrs = {f"{s}_{l}": [] for s, l in product(
            ['contra', 'ipsi'], lid_states[-3:]
        )}
        sub_arrs = {f"{s}_{l}": [] for s, l in product(
            ['contra', 'ipsi'], lid_states[-3:]
        )}

    # loop over selected uni-LID subjects and sources
    if not IPSI_CONTRA_UNILID: loop_subs = TFs.keys()
    else: loop_subs = unilid_subs

    for sub_code, src in product(loop_subs, sources):
        if SOURCE not in src: continue

        if IPSI_CONTRA_UNILID: sub, lidside = sub_code.split('_')
        else: sub = sub_code
        if sub.startswith('1') and 'ecog' in src.lower(): continue

        if src == 'ecog':  # IPSI_CONTRA_UNILID and 
            ephysside = get_ecog_side(sub)
            src = f'ecog_{ephysside}'

        # get all ephys value
        psx = TFs[sub][src].values.T
        ps_times = TFs[sub][src].times / 60
        ps_freqs = TFs[sub][src].freqs
        assert psx.shape == (len(ps_times), len(ps_freqs)), (
            f'incorrect psx data loaded {sub}, {src} -> '
            f'psx: {psx.shape} t: {len(ps_times)}, f: {len(ps_freqs)}'
        )
        # select relevant freqs
        sel1 = np.logical_and(ps_freqs >= 4, ps_freqs < 35)
        sel2 = np.logical_and(ps_freqs >= 60, ps_freqs < 90)
        f_sel = np.logical_or(sel1, sel2)
        psx = psx[:, f_sel]
        ps_freqs = ps_freqs[f_sel]


        # split values for unilat-LID
        if not IPSI_CONTRA_UNILID:
            lid_times, lid_scores = get_cdrs_specific(
                sub=sub, INCL_CORE_CDRS=True, side='both'
            )
            # print('lid scores', lid_scores.values)
            # print('lid times', lid_times.values)
            # print('pstimes', ps_times)
            # match ephys values with uniLID epochs
            near_time_idx = [np.argmin(abs(lid_times - t))
                             for t in ps_times]
            ps_scores = lid_scores[near_time_idx]

            
        elif IPSI_CONTRA_UNILID:
            # define orientation of hemisphere to unilat-LID
            if 'ecog' not in src:  # ecog ephysside defined above
                ephysside = src.split('_')[1]
            if lidside == ephysside: SIDE = 'ipsi'
            else: SIDE = 'contra'

            lid_times, cdrs, othercdrs = unilid_subs[sub_code]
            unilid_sel = np.logical_and(cdrs > 0, othercdrs == 0)
            uni_times = lid_times[unilid_sel].values
            uni_cdrs = cdrs[unilid_sel].values
            # match ephys values with uniLID epochs
            near_time_idx = [np.argmin(abs(lid_times - t)) for t in ps_times]
            near_times = lid_times[near_time_idx]  # get matching lid times for ephys
            near_cdrs = cdrs[near_time_idx]
            # select value-epochs for uni-LID times
            unilid_sel = np.isin(near_times, uni_times)
            ps_times = ps_times[unilid_sel]
            ps_scores = near_cdrs[unilid_sel]
            psx = psx[unilid_sel, :]
        
        # baseline correct
        try:
            # power baselines from movement selection
            if FEATURE == 'POWER':
                if 'ecog_' in src:
                    bl_attr = f'ecog_{sub}_baseline'
                elif 'lfp_' in src:
                    bl_attr = f'{src}_{sub}_baseline'
                # coherence baseline from 10-sec epochs, smaller movement-selected data not comparable
                # else: bl_attr = f'{src}_{sub}_baseline'
                bl = getattr(BASELINE, bl_attr)
                if len(bl.shape) == 2: bl_m = np.mean(bl, axis=0)
                if len(bl.shape) == 1: bl_m = bl
            
            elif 'COH' in FEATURE:
                # take baseline over 10-sec epochs, first 5 minutes, no Dyskinesia
                bl = psx[np.logical_and(ps_times < 5, ps_scores.values == 0)]
                bl_m = np.mean(bl, axis=0)

            
        except:
            print(f'### WARNING no baseline {src} sub {sub}')
            continue

        if BASE_METHOD == 'percChange':
            # print(f'BASELINING % CHANGE {sub}')
            # plt.plot(ps_freqs, bl_m, lw=3, alpha=.5, label='BASE')
            # plt.plot(ps_freqs, np.mean(psx, axis=0), lw=3, alpha=.5, label=f'psx (n={len(psx)})')
            # plt.plot(ps_freqs, bl_m, label=f'BASE 2 (n={len(bl)})')
            # plt.legend()
            # plt.show()
            psx = ((psx - bl_m) / bl_m) * 100

        elif BASE_METHOD == 'Z':
            assert len(bl.shape) == 2, 'Z-score not possible, only mean BASELINE provided'
            bl_sd = np.std(bl, axis=0)
            psx = ((psx - bl_m) / bl_sd)
        
        else:
            print(f'\n### NO BASELINING DONE IN GENERAL: {BASE_METHOD}')


        # select values into LID categories
        for lidkey, cdrs_range in lid_categ_ranges.items():
            # print(f'SELECT OUT {lidkey} (){cdrs_range}')
            cat_sel = np.logical_and(ps_scores >= cdrs_range[0],
                                     ps_scores <= cdrs_range[1])
            # print(sum(cat_sel))
            if sum(cat_sel) == 0: continue
            cat_psx = psx[cat_sel, :]
            cat_times = ps_times[cat_sel]
            cat_subs = [sub] * sum(cat_sel)
            # add to store dicts
            if lidkey == 'nolid':
                for extrakey in ['below30', 'over30']:
                    store_key = lidkey + extrakey
                    if extrakey == 'below30': t_sel = cat_times < 30
                    else: t_sel = cat_times >= 30
                    psd_arrs[store_key].append(cat_psx[t_sel])
                    sub_arrs[store_key].append(np.array(cat_subs)[t_sel])

            else:
                store_key = lidkey
                if IPSI_CONTRA_UNILID: store_key = f'{SIDE}_{lidkey}'
                psd_arrs[store_key].append(cat_psx)
                sub_arrs[store_key].append(cat_subs)

    psd_arrs, sub_arrs = unpack_dict_lists(psd_arrs, sub_arrs)

    return psd_arrs, ps_freqs, sub_arrs


def unpack_dict_lists(psd_arrs, sub_arrs):
    """
    unpack every lid-category dict-content
    containing of lists of lists/arrays
    """
    for cat_key in psd_arrs:
        psd_arrs[cat_key] = np.array(
            [row for arr in psd_arrs[cat_key] for row in arr]
        )
        sub_arrs[cat_key] = np.array(
            [s for l in sub_arrs[cat_key] for s in l]
        )
        assert len(sub_arrs[cat_key]) == psd_arrs[cat_key].shape[0], (
            f'MISMAtCH in {cat_key}: n-subs {len(sub_arrs[cat_key])} with'
            f' n-rows: {psd_arrs[cat_key].shape}'
        )

    return psd_arrs, sub_arrs