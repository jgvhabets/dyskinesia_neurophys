"""Process cortex-STN multivariate connectivity results"""

import json
import os
from itertools import product
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle)
from lfpecog_analysis.ft_processing_helpers import (
    split_lid_subs,
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS,
    FeatLidClass  # to import acc containing class for movement selection
)
from lfpecog_analysis.get_acc_task_derivs import (
    select_tf_on_movement_10s
)
from lfpecog_analysis._connectivity_helpers import (
    load_coordinates,
    process_results,
    plot_results_timefreqs,
    plot_results_patterns,
)


def get_conn_values_to_plot(
    TF_dict,
    BASELINE_CORRECT: bool = True,
    BASELINE_EXCL_MOVE: bool = True,
    EXCL_ONLY_IPSI_LID: bool = True,
    CDRS_SIDE: str = 'bilat',
    INCL_CORE_CDRS: bool = False,
    CATEG_CDRS: bool = True,
    CDRS_MOD_THR = {'unilat': 2.5, 'bilat': 4.5},
    CDRS_SEV_THR = {'unilat': 4.5, 'bilat': 7.5},
    incl_conn_sides = ['ipsilateral'],  #, 'contralteral']
    SELECT_MOVEMENT: bool = False,
    RETURN_MEAN_per_CAT: bool = True,  # only not for stats
    verbose: bool = True,
):
    assert CDRS_SIDE in ['unilat', 'bilat'], 'incorrect CDRS_SIDE'

    # get class with acc data if selection is defined (not False)
    if np.logical_or(isinstance(SELECT_MOVEMENT, str),
                     BASELINE_CORRECT and BASELINE_EXCL_MOVE):
        ### get class containing ACC data if movement selection
        featLabPath = os.path.join(get_project_path('data'),
                                    'prediction_data',
                                    'featLabelClasses',
                                    f'featLabels_ftv6_Cdrs_StnOnly.P')
        class_w_acc = load_class_pickle(featLabPath,
                                        convert_float_np64=True)
        print(f'extracted acc data class for {SELECT_MOVEMENT}')

    # create dict to store generated values
    plot_values = {s: {} for s in incl_conn_sides}
    if CATEG_CDRS:
        for cat, s in product([1, 2, 3], incl_conn_sides):
            plot_values[s][cat] = []
        if not BASELINE_CORRECT:
            for s in incl_conn_sides: plot_values[s][0] = []  #  add no LID only w/o baseline correction
    # parallel dict to track sub ids
    value_subs = deepcopy(plot_values)

    for sub, side in product(TF_dict.keys(), incl_conn_sides):
        # print(f'\n\tSTART sub-{sub}, {side}')
        # get values for subject and side
        values = TF_dict[sub][side].values
        times = TF_dict[sub][side].times
        freqs = TF_dict[sub][side].freqs
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
        
        ### get BASELINE without movement
        if BASELINE_CORRECT and BASELINE_EXCL_MOVE:
            BL_values, _ = select_tf_on_movement_10s(
                feat10sClass=class_w_acc, sub=sub,
                tf_values_arr=values[cdrs==0, :],
                tf_times_arr=times[cdrs==0],
                SELECT_ON_ACC_RMS='EXCL_MOVE',
            )
            BL_values = np.nanmean(BL_values, axis=0)
            if np.isnan(BL_values).all():
                first5min = times > (60*5)
                BL_values, _ = select_tf_on_movement_10s(
                    feat10sClass=class_w_acc, sub=sub,
                    tf_values_arr=values[first5min, :],
                    tf_times_arr=times[first5min],
                    SELECT_ON_ACC_RMS='EXCL_MOVE',
                )
                BL_values = np.nanmean(BL_values, axis=0)
                print(f'corrected baseline for sub-{sub} to 5-MINUTES')

        ### SELECT OUT ON ACC RMS IF DEFINED HERE
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
                plot_figure=False,
            )
            cdrs = cdrs[move_sel]
            if verbose: print(f'after move-selection ({SELECT_MOVEMENT}): {values.shape}')

        # calculate baseline on current data (without considering movement)
        if BASELINE_CORRECT and not BASELINE_EXCL_MOVE:
            BL_values = np.nanmean(values[cdrs==0, :], axis=0)
            if np.isnan(BL_values).all():
                first5min = times > (60*5)
                BL_values = np.nanmean(values[first5min, :], axis=0)
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
                cat_values = values[cdrs == cat]
                # return none mean values used for statistics
                if RETURN_MEAN_per_CAT:
                    cat_values = np.nanmean(cat_values, axis=0)
                    # add individual (corrected) cat-mean
                    if BASELINE_CORRECT: cat_values = (cat_values - BL_values) / BL_values * 100
                plot_values[side][cat].append(cat_values)
                # add originating sub id
                value_subs[side][cat].append(sub)
                
        else:
            plot_values[side][f'{sub}_{side}'] = np.array([values, cdrs])
        
        if verbose: print()

    return plot_values, freqs, value_subs


def get_conn_values_sub_side(
    sub, stn_side, conn_method,
    results_df=False, CONN_FT_PATH=None,
    return_times_freqs=False,
):
    
    if isinstance(results_df, bool) and not results_df:
        (results_df,
         window_times,
         freqs) = process_results(
            method=conn_method, subjects=sub,
            results_path=CONN_FT_PATH
        )
        return_times_freqs = True

    # select data on sub and side
    conn_data = [
        results_df.iloc[i][conn_method]
        for i in np.arange(results_df.shape[0])
        if (results_df.iloc[i]['subject'] == sub and
            results_df.iloc[i]['seed_target_lateralisation'] == stn_side)
    ][0]

    if return_times_freqs: return conn_data, window_times, freqs
    else: return conn_data



if __name__ == '__main__':

    # define project path
    PROJECT_PATH = (
        "C:\\Users\\tsbin\\OneDrive - Charité - Universitätsmedizin Berlin\\"
        "PROJECT ECOG-LFP Coherence\\Jeroen_Dyskinesia"
    )
    RESULTS_PATH = os.path.join(
        PROJECT_PATH,
        "results",
        "features",
        "connectivity",
        "windows_10s_0.5overlap",
    )

    # get available subjects
    INFO_PATH = os.path.join(PROJECT_PATH, "data", "meta_info")
    with open(
        os.path.join(INFO_PATH, "ftExtr_spectral_v6.json"), encoding="utf8"
    ) as file:
        subjects = json.load(file)["TOTAL_SUBS"]
    # take only subjects with ECoG & LFP data
    SUBJECTS = [sub for sub in subjects if sub.startswith("0")]

    METHOD = "mic"

    results, window_times, freqs = process_results(
        method=METHOD, subjects=SUBJECTS, results_path=RESULTS_PATH
    )

    fig, axis = plot_results_timefreqs(
        results=results,
        method=METHOD,
        times=window_times,
        freqs=freqs,
        eligible_entries={"seed_target_lateralisation": "ipsilateral"},
        show=False,
    )

    if METHOD == "mic":
        coordinates = load_coordinates(
            os.path.join(INFO_PATH, "ECoG_LFP_coords.csv")
        )
        coordinates["x"] = np.abs(coordinates["x"])
        fig, axis = plot_results_patterns(
            results=results,
            coordinates=coordinates,
            method=METHOD,
            times=window_times,
            freqs=freqs,
            time_range=None,  # (0, 800),
            freq_range=(12, 35),
            eligible_entries={"seed_target_lateralisation": "ipsilateral"},
            show=False,
        )

    plt.show(block=True)

    print("jeff")
