"""
Plot Connectivity metrics against Dyskinesia
scores (CDRS)

run on WIN as:
xxx\dyskinesia_neurophys\code> python -m lfpecog_plotting.plot_connectivity_vs_LID
"""

import numpy as np
from itertools import product
import os
import matplotlib.pyplot as plt

from lfpecog_analysis.ft_processing_helpers import (
    split_lid_subs,
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS,
    FeatLidClass  # to import acc containing class for movement selection
)
from utils.utils_fileManagement import (
    load_ft_ext_cfg,
    get_project_path,
    load_class_pickle
)
from lfpecog_analysis.get_SSD_timefreqs import get_all_ssd_timeFreqs
from lfpecog_analysis.get_acc_task_derivs import (
    select_tf_on_movement_10s
)
from lfpecog_plotting.plot_descriptive_SSD_PSDs import (
    plot_scaling_LID
)


def get_conn_values_to_plot(
    TF_dict,
    BASELINE_CORRECT: bool = True,
    EXCL_ONLY_IPSI_LID: bool = True,
    CDRS_SIDE: str = 'bilat',
    INCL_CORE_CDRS: bool = False,
    CATEG_CDRS: bool = True,
    CDRS_MOD_THR = {'unilat': 2.5, 'bilat': 4.5},
    CDRS_SEV_THR = {'unilat': 4.5, 'bilat': 7.5},
    incl_conn_sides = ['ipsilateral'],  #, 'contralteral']
    SELECT_MOVEMENT: bool = False,
    class_w_acc = None,
    verbose: bool = True,
):
    assert CDRS_SIDE in ['unilat', 'bilat'], 'incorrect CDRS_SIDE'

    plot_values = {s: {} for s in incl_conn_sides}
    if CATEG_CDRS:
        for cat, s in product([1, 2, 3], incl_conn_sides):
            plot_values[s][cat] = []
        if not BASELINE_CORRECT:
            for s in incl_conn_sides: plot_values[s][0] = []  #  add no LID only w/o baseline correction


    for sub, side in product(TFs.keys(), incl_conn_sides):
        # print(f'\n\tSTART sub-{sub}, {side}')
        # get values for subject and side
        values = TFs[sub][side].values
        times = TFs[sub][side].times
        freqs = TFs[sub][side].freqs
        # remove empty rows
        nan_sel = np.array([all(np.isnan(values[i, :]))
                            for i in np.arange(len(times))])
        values = values[~nan_sel]
        times = times[~nan_sel]

        # get corr CDRS scores
        if CDRS_SIDE == 'unilat': cdrs_code = 'contralat ecog'
        elif CDRS_SIDE == 'bilat': cdrs_code = 'both' # include CORE for both sides full 'contralat ecog'
        cdrs = find_select_nearest_CDRS_for_ephys(
            sub=sub, side=cdrs_code,  
            ft_times=np.array(times)/60,
            INCL_CORE_CDRS=INCL_CORE_CDRS,
        )
        
        # remove unilateral LID ipsilat to ecog
        if EXCL_ONLY_IPSI_LID:
            excl_lids = find_select_nearest_CDRS_for_ephys(
                sub=sub, side=cdrs_code,  #'ipsilat ecog',  # include CORE for both sides
                # cdrs_rater='Jeroen',
                ft_times=np.array(times)/60,
                INCL_CORE_CDRS=INCL_CORE_CDRS,
            )
            excl_lids = np.logical_and(excl_lids > 0, cdrs==0)
            # print(f'for sub {sub} excl: {sum(excl_lids) / len(excl_lids)}')
            values = values[~excl_lids, :]
            times = times[~excl_lids]
            cdrs = cdrs[~excl_lids]
        
        ### SELECT OUT ON ACC RMS IF DEFINED HERE ###
        if SELECT_MOVEMENT:
            # feats10Class (FeatLidClass) defaults to RMS-mean (L+R)
            if verbose: print(f'before move-selection ({SELECT_MOVEMENT}): {values.shape}')
            values, times, move_sel = select_tf_on_movement_10s(
                feat10sClass=class_w_acc,
                sub=sub,
                tf_values_arr=values,
                tf_times_arr=times,
                SELECT_ON_ACC_RMS=SELECT_MOVEMENT,
                RETURN_MOVE_SEL_BOOL=True,
                # RMS_Z_THRESH=0,  # defaults to -0.5
            )
            cdrs = cdrs[move_sel]
            if verbose: print(f'after move-selection ({SELECT_MOVEMENT}): {values.shape}')

        # calculate baseline on NO LID
        if BASELINE_CORRECT:
            BL = np.nanmean(values[cdrs==0, :], axis=0)
            if np.isnan(BL).all():
                first5min = times > (60*5)
                BL = np.nanmean(values[first5min, :], axis=0)
                print(f'corrected baseline for sub-{sub} to 5-MINUTES')

        # convert cdrs scores into categorical values
        if verbose: print(f'({sub}) unique-cdrs: {np.unique(cdrs)}')
        if CATEG_CDRS:
            cdrs = categorical_CDRS(
                y_full_scale=cdrs,
                preLID_separate=False,
                cutoff_mildModerate=CDRS_MOD_THR[CDRS_SIDE],
                cutoff_moderateSevere=CDRS_SEV_THR[CDRS_SIDE],
            )
            if verbose: print(f'({sub}) unique-cdrs-cats: {np.unique(cdrs)}')

            for cat in np.unique(cdrs):
                if verbose: print(f'({sub}) adding cat-{cat}: {sum(cdrs == cat)}')
                if cat == 0 and BASELINE_CORRECT: continue
                cat_mean = np.nanmean(values[cdrs == cat], axis=0)
                # add individual (corrected) cat-mean
                if BASELINE_CORRECT: cat_mean = (cat_mean - BL) / BL * 100
                plot_values[side][cat].append(cat_mean)
                
        else:
            plot_values[side][f'{sub}_{side}'] = np.array([values, cdrs])
        
        if verbose: print()

    return plot_values


if __name__ == '__main__':
    # DEFINE SETTINGS AND VARIABLES FOR DATA PREP AND PLOTTING
    FT_VERSION = 'v6'
    CDRS_ORIGIN = 'unilat'
    DATA_TYPE = 'trgc'  # Connect metric mic / trgc
    INCL_CONN_SIDES = ['ipsilateral',]  # 'contralateral']
    SELECT_MOVEMENT = 'INCL_MOVE'  # should be False, INCL_MOVE, EXCL_MOVE
    BASELINE_CORRECT = True
    DATE = '1212'
    VERBOSE = False

    # PM: check baseline without movement (plot_descr_SSD_PSDs, lines ~1375)

    # settings autom. extracted based on defined variables    
    SETTINGS = load_ft_ext_cfg(FT_VERSION=FT_VERSION)
    DATA_VERSION = SETTINGS['DATA_VERSION']
    # take only subjects with ECoG & LFP data
    SUBJECTS = [sub for sub in SETTINGS['TOTAL_SUBS']
                if sub.startswith("0")]
    if DATA_TYPE == 'trgc': BASELINE_CORRECT = False


    # get values
    TFs = get_all_ssd_timeFreqs(
        SUBS=SUBJECTS, FT_VERSION=FT_VERSION,
        DATA_VERSION=DATA_VERSION,
        GET_CONNECTIVITY=DATA_TYPE,
        verbose=VERBOSE,
    )
    freqs = TFs[SUBJECTS[0]][INCL_CONN_SIDES[0]].freqs
    print('got tf values')

    # get class containing ACC data if movement selection
    if isinstance(SELECT_MOVEMENT, str):
        featLabPath = os.path.join(
            get_project_path('data'),
            'prediction_data', 'featLabelClasses',
            f'featLabels_ft{FT_VERSION}_Cdrs_StnOnly.P'
        )
        ftClass10sec = load_class_pickle(featLabPath,
                                         convert_float_np64=True)
        print('extracted acc data class')
    else: ftClass10sec=None

    # sort and average values into categories 
    plot_values = get_conn_values_to_plot(
        TFs,
        BASELINE_CORRECT=BASELINE_CORRECT,
        CDRS_SIDE=CDRS_ORIGIN,
        incl_conn_sides=INCL_CONN_SIDES,
        SELECT_MOVEMENT=SELECT_MOVEMENT,
        class_w_acc=ftClass10sec,
        verbose=VERBOSE,
    )
    print('got plot values')

    if len(INCL_CONN_SIDES) == 2: mic_type = 'both'
    else: mic_type = INCL_CONN_SIDES[0].split('lat')[0]
    FIG_NAME=f'{DATE}_{DATA_TYPE.upper()}{mic_type}_{CDRS_ORIGIN}LID'
    if SELECT_MOVEMENT: FIG_NAME += f' {SELECT_MOVEMENT}'

    plot_scaling_LID(
        psds_to_plot=plot_values,
        tf_freqs=freqs,
        cdrs_origin=CDRS_ORIGIN,
        cdrs_cat_coding={'no': 0, 'mild': 1,
                         'moderate':2 , 'severe': 3},
        datatype=DATA_TYPE,
        BASELINE_CORRECT=BASELINE_CORRECT,
        fig_name=FIG_NAME,
        FT_VERSION=FT_VERSION,
        DATA_VERSION=DATA_VERSION,
    )
    print(f'plotted {FIG_NAME}')