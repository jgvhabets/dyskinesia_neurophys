"""
Selected specific data based on TASK,
MOVEMENT PRESENCE, and DYSKINESIA SEVERITY
"""
# import functions and packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# import own functions
# import lfpecog_analysis.load_SSD_features as load_ssd_fts
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle,
                                        correct_acc_class,
                                        load_ft_ext_cfg,
                                        get_avail_ssd_subs)
from lfpecog_analysis.ft_processing_helpers import (
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS
)


def select_3d_ephys_moveTaskLid(
    ephys_3d: np.ndarray, ephys_time_2d: np.ndarray,
    move_masks: dict,
    task_mask: np.ndarray, lid_mask: np.ndarray,
    SEL: str, SEL_bodyside: str = 'both',
    NAN_ACCEPT_WIN: int = 2,
    verbose: bool = False,
):
    """
    processes data for single subject, for one
    ephys-source (lfp_left, lfp_right, ecog_...).
    To keep 3d-2d structure, first mask with NaNs
    instead of deleting data (and destroying)
    2d-3d-arrays.

    Arguments:
        - ephys_3d: ndarray [n-windows,
            n-samples(10sec), n-bands]
        - move_masks (dict): different movement
            selections, all 2d arrays [n-wins, n-samples]
        - task_mask: 2d array [n-wins, n-samples], coded
            0: rest, 1: tap, 2: free
        - lid_mask: 2d array [n-wins, n-samples]
        - NAN_ACCEPT_WIN: NaN-seconds accepted, NaNs can
            have caused temporal inaccuracy during windowing
    """
    # check selection
    allowed_sels = ['REST_NOLID', 'REST',
                    'VOLUNTARY', 'INVOLUNTARY']
    assert SEL.upper() in allowed_sels, f'incorrect SEL: {SEL}'
    SEL = SEL.upper()
    allowed_sides = ['left', 'right', 'both']
    assert SEL_bodyside.lower() in allowed_sides, f'incorrect side'
    SEL_bodyside = SEL_bodyside.lower()

    if verbose: print(f'avail move_masks: {move_masks.keys()}')

    # create overall 2d-MASK to finally select desired data
    MASK = np.zeros_like(task_mask)  # take 2d mask, 1 is keep, 0 is drop

    # take one band to calculate, NaNs per window
    window_nans = np.sum(np.isnan(dat[:, :, 0]), axis=-1)
    # windows to exclude bcs of NaNs (accuracy with movement labels)
    nan_win_sel = window_nans > NAN_ACCEPT_WIN

    if verbose: print(
        f'NaN-window-selection: {sum(nan_win_sel)} out of'
        f' {len(nan_win_sel)} ({round(sum(nan_win_sel) / len(nan_win_sel) * 100)}'
        f'%) will be removed'
    )

    # Selection-1: add task selection to mask
    if 'REST' in SEL or SEL == 'INVOLUNTARY': TASK = [0]  # only select rest
    elif SEL == 'VOLUNTARY': TASK = [1]  # only select tap
    else: TASK == [0, 1, 2]  # select all

    for T in TASK: MASK[task_mask == T] = 1
    if verbose: print(f'total KEEPS in MASK after TASK : {np.sum(MASK) / 7 / 2048} (sec, p/band)')

    # Selection-2: add 2d-mask for Movement selection
    if 'REST' in SEL:
        mov_mask = move_masks['no_move']  # select move-mask
    
    # add taps per defined side to mask
    elif SEL == 'VOLUNTARY':
        mov_mask = np.zeros_like(MASK)
        for mask_side in ['left', 'right']:
            if SEL_bodyside in ['both', mask_side]:
                mov_mask[move_masks[f'{mask_side}_tap'] == 1] = 1
        
    # add most ensured dyskinesia moments
    # above only rest-task selected, now add all movements
    elif SEL == 'INVOLUNTARY':
        mov_mask = np.zeros_like(MASK)
        for mask_side in ['left', 'right']:
            if SEL_bodyside in ['both', mask_side]:
                mov_mask[move_masks[f'{mask_side}_allmove'] == 1] = 1
        
    MASK = np.logical_and(MASK, mov_mask)  # combine with existing mask-selection
    if verbose: print(f'total KEEPS in MASK after MOVE: {np.sum(MASK) / 7 / 2048} (sec, p/band)')

    # Selection-3: add 2d-mask for LID
    if 'NOLID' in SEL:
        lid_mask = lid_mask == 0
    elif SEL == 'INVOLUNTARY':
        lid_mask = lid_mask > 0
        check = np.logical_and(MASK, ~lid_mask)  # checks REST-MOVEMENT-NoDYSK
        if verbose: print(f'VOLUNTARY check (REST, MOVE, noLID): {np.sum(check) / 7 / 2048} (sec, p/band)')
    else:
        if verbose: print('all (no) LID states included')
        lid_mask = np.ones_like(lid_mask)
    
    MASK = np.logical_and(MASK, lid_mask)  # combine with existing mask-selection
    if verbose: print(f'total KEEPS in MASK after LID: {np.sum(MASK) / 7 / 2048} (sec, p/band)')

    # Selection-4: apply mask on all ephys AND times
    ephys_3d[MASK == 0, :] = np.nan   # masks all excluded samples with nan
    ephys_time_2d[MASK == 0] = np.nan

    if verbose: print(f'total NaNs in 3d after MASK: '
                      f'{np.sum(np.isnan(ephys_3d)) / 7 / ephys_3d.shape[0] / 2048}'
                      ' sec per 10-sec window (mean)')
    
    # masks windows to exclude based on NaNs with full-row-nans
    ephys_3d[nan_win_sel, :, :] = np.nan
    ephys_time_2d[nan_win_sel, :] = np.nan
    if verbose: print(f'total NaNs in 3d after NaN-window-masking: '
                        f'{np.sum(np.isnan(ephys_3d)) / 7 / ephys_3d.shape[0] / 2048}'
                        ' sec per 10-sec window (mean)')
    
    # Selection-5: select for NaNs in ephys, parallel in times
    if verbose: print(f'EPHYS shape orig: {ephys_3d.shape}')
    SEL_ephys_3d = ephys_3d[~np.isnan(ephys_time_2d), :]  # with excluding NaNs loses one dimension
    SEL_time_2d = ephys_time_2d[~np.isnan(ephys_time_2d)]  # with excluding NaNs loses one dimension
    if verbose: print(f'SELECTED EPHYS shape: {SEL_ephys_3d.shape}')
    if verbose: print(f'SELECTED TIME shape: {SEL_time_2d.shape}')
    ### ASSUMES SAME NAN PATTERN IN ALL BANDS

    # Selection-5: ravel with unique times left after masking and nan-removal
    uniq_times, uniq_idx = np.unique(SEL_time_2d, return_index=True)
    if verbose: print(f'{len(SEL_time_2d)} time-samples, '
                      f'{len(uniq_times)} UNIQUE times')
    uniq_ephys_2d = SEL_ephys_3d[uniq_idx, :]
    
    return uniq_ephys_2d, uniq_times


    # check how many data present per 10-sec window!
    
    
    # try out to plot REST-rest vs TAP-tap vs REST-all_move+CDRS
    # include baseline correction