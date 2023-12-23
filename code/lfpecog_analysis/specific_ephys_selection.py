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
from lfpecog_analysis.psd_analysis_classes import PSD_vs_Move_sub


def select_3d_ephys_moveTaskLid(
    SEL: str, SEL_bodyside: str = 'both',
    psdMoveClass = None, ephys_source: str = 'no class',
    DYSK_SEL = 'all', DYSK_UNILAT_SIDE = False,
    ephys_3d: np.ndarray = None,
    ephys_time_2d: np.ndarray = None,
    move_masks: dict = None,
    task_mask: np.ndarray = None,
    lid_mask: np.ndarray = None,
    DYSK_CUTOFFS = {'mild': (1, 3), 'moderate': (4, 7),
                    'severe': (8, 25)},
    EXCL_ECOG_IPSILAT: bool = False,
    BASELINE_MINUTES: int = 5,
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
        - psdMoveClass: can be given containing most other
            variables
        - ephys_3d: ndarray [n-windows, n-samples(10sec), n-bands]
        - ephys_time_2d: 2d timestamp array corr to every ephys-band
        - move_masks (dict): different movement
            selections, all 2d arrays [n-wins, n-samples], contain
            any_move, no_move, left_tap, left_allmove, right...
        - task_mask: 2d array [n-wins, n-samples], coded
            0: rest, 1: tap, 2: free
        - lid_mask: 2d array [n-wins, n-samples]
        - DYSK_SEL: dyskinesia selection, 'no': only without lid,
            'mild': 1-3 CDRS, 'moderate': 4-7 CDRS, 'severe': 8+,
            'lid': >= 1 CDRS, 'all': no selection (with and without incl)
        - DYSK_UNILAT_SIDE: defaults to False (no unilat dyskinesia
            selected), if left or right: CDRS scores for only unilat
            side are used, without axial scores
        - DYSK_CUTOFFS: default cutoffs for LID categories
            'mild': (1, 3), 'moderate': (4, 7), 'severe': (8, 25)
        - BASELINE_MIN: baseline consists first 5 minutes after LDopa
        - NAN_ACCEPT_WIN: NaN-seconds accepted, NaNs can
            have caused temporal inaccuracy during windowing
    
    Returns: uniq_ephys_2d, uniq_times, lid_out, task_out
        - selected and unique ephys array (2d: [n-samples, n-bands]
        - timestamps (array, in sec, correspoding to ephys-samples)
        - CDRS scores (array, in sec, correspoding to ephys-samples)
        - task codes (in sec, correspoding to ephys-samples)
            0: rest, 1: tap, 2: free
    """
    # automated input check and correction
    allowed_sels = ['BASELINE', 'REST',
                    'VOLUNTARY', 'INVOLUNTARY',]
    assert SEL.upper() in allowed_sels, f'incorrect SEL: {SEL}'
    SEL = SEL.upper()
    
    allowed_sides = ['left', 'right', 'both']
    assert SEL_bodyside.lower() in allowed_sides, f'incorrect side'
    SEL_bodyside = SEL_bodyside.lower()
    
    allowed_dysk_sels = ['no', 'mild', 'moderate', 'severe', 'all', 'lid']
    assert DYSK_SEL.lower() in allowed_dysk_sels, f'incorrect dysk_sel'
    DYSK_SEL = DYSK_SEL.lower()

    assert not np.logical_and(SEL == 'INVOLUNTARY', DYSK_SEL in ['no', 'all']), (
        'if INVOLUNTARY is defined, NO DYSK (or all) cannot be included'
    )

    # get all variables out of class
    if type(psdMoveClass) == PSD_vs_Move_sub:
        assert ephys_source in psdMoveClass.ephys_sources, (
            'incorrect ephyssource for psdMoveClass usage'
        )
        ephys_3d = getattr(psdMoveClass, f'{ephys_source}_3d').copy()
        ephys_time_2d = psdMoveClass.ephys_time_arr.copy()
        move_masks=psdMoveClass.move_masks
        task_mask=psdMoveClass.task_mask.copy()
        if DYSK_UNILAT_SIDE in ['left', 'right']:
            print(f'use UNILAT LID_MASK {DYSK_UNILAT_SIDE}')
            if DYSK_UNILAT_SIDE == 'left': NOLID_SIDE = 'right'
            elif DYSK_UNILAT_SIDE == 'right': NOLID_SIDE = 'left'
            lid_mask = np.logical_and(
                getattr(psdMoveClass, f'lid_mask_{DYSK_UNILAT_SIDE}') > 0,
                getattr(psdMoveClass, f'lid_mask_{NOLID_SIDE}') == 0
            )
        else:
            lid_mask = psdMoveClass.lid_mask
        # optionnaly remove ipsilat-unilat-dyskinesia comp to ECoG
        if EXCL_ECOG_IPSILAT and any(['ecog' in s for s in
                                      psdMoveClass.ephys_sources]):
            if 'ecog_left' in psdMoveClass.ephys_sources:
                EXCL_ECOG_IPSILAT = np.logical_and(psdMoveClass.lid_mask_left > 0,
                                                   psdMoveClass.lid_mask_right == 0)
            elif 'ecog_right' in psdMoveClass.ephys_sources:
                EXCL_ECOG_IPSILAT = np.logical_and(psdMoveClass.lid_mask_left == 0,
                                                   psdMoveClass.lid_mask_right > 0)
                
            
    
    if DYSK_UNILAT_SIDE in ['left', 'right']:
        DYSK_SEL = 'lid'  # do not categorize unilat Dyskinesia
        print(f'UNILAT DYSKINESIA ({DYSK_UNILAT_SIDE}) taken in raw CDRS scores')

    # create overall 2d-MASK to finally select desired data
    MASK = np.zeros_like(task_mask)  # take 2d mask, 1 is keep, 0 is drop

    # take one band to calculate, NaNs per window
    window_nans = np.sum(np.isnan(ephys_3d[:, :, 0]), axis=-1)
    # windows to exclude bcs of NaNs (accuracy with movement labels)
    nan_win_sel = window_nans > NAN_ACCEPT_WIN

    if verbose: print(
        f'NaN-window-selection: {sum(nan_win_sel)} out of'
        f' {len(nan_win_sel)} ({round(sum(nan_win_sel) / len(nan_win_sel) * 100)}'
        f'%) will be removed'
    )


    ### SELECTION-1: add TASK SELECTION to mask
    task_out = task_mask.copy()
    if 'REST' in SEL or SEL in ['INVOLUNTARY', 'BASELINE']:
        TASK = [0]  # only select rest
    elif SEL == 'VOLUNTARY':
        TASK = [1]  # only select tap
    else:
        TASK == [0, 1, 2]  # select all

    for T in TASK: MASK[task_mask == T] = 1
    if verbose: print(f'total KEEPS in MASK after TASK : {np.sum(MASK) / 7 / 2048} (sec, p/band)')

    # select only first X minutes for baseline
    if SEL == 'BASELINE':
        time_mask = ephys_time_2d < (BASELINE_MINUTES * 60)
        MASK = np.logical_and(MASK, time_mask)  # combine with existing mask-selection
        if verbose: print(f'total KEEPS in MASK after baseline time : {np.sum(MASK) / 7 / 2048} (sec, p/band)')


    ### SELECTION-2: add 2d-mask for MOVEMENT SELECTION
    if 'REST' in SEL or SEL == 'BASELINE':
        mov_mask = move_masks['no_move']  # select move-mask
        if verbose: print('\t...all movement excluded in rest or baseline')
    
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


    ### SELECTION-3: add 2d-mask for LID (DYSKINESIA PRESENCE or SEVERITY)
    lid_out = lid_mask.copy()
    if DYSK_SEL == 'no' or SEL == 'BASELINE':
        lid_mask = lid_mask == 0
    
    elif DYSK_SEL == 'lid':
        lid_mask = lid_mask > 0
        # check = np.logical_and(MASK, ~lid_mask)  # checks REST-MOVEMENT-NoDYSK
        # if verbose: print(f'VOLUNTARY check (REST, MOVE, noLID): {np.sum(check) / 7 / 2048} (sec, p/band)')
    
    elif DYSK_SEL in ['mild', 'moderate', 'severe']:
        lid_mask = np.logical_and(lid_mask >= DYSK_CUTOFFS[DYSK_SEL][0],
                                  lid_mask <= DYSK_CUTOFFS[DYSK_SEL][1],)

    else:
        if verbose: print('...all LID states (with + without) included')
        lid_mask = np.ones_like(lid_mask)
    
    MASK = np.logical_and(MASK, lid_mask)  # combine with existing mask-selection
    if verbose: print(f'total KEEPS in MASK after LID: {np.sum(MASK) / 7 / 2048} (sec, p/band)')

    # optionally exclude ipsilat-Dysk IPSI-lat to ECoG
    if isinstance(EXCL_ECOG_IPSILAT, np.ndarray):  # if defined above 
        MASK = np.logical_and(MASK, ~EXCL_ECOG_IPSILAT)  # unilat ecog array is positive for excl
        if verbose: print(f'total KEEPS in MASK after uni-lat-ipsi-lat-ecog '
                          f'exclusion: {np.sum(MASK) / 7 / 2048} (sec, p/band)')


    ### SELECTION-4: APPLY MASKs on all ephys AND times
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


    ### SELECTION-5: select for EPHYS-NaNs, parallel in times
    if verbose: print(f'EPHYS shape orig: {ephys_3d.shape}')
    SEL_ephys_3d = ephys_3d[~np.isnan(ephys_time_2d), :]  # with excluding NaNs loses one dimension
    SEL_time_2d = ephys_time_2d[~np.isnan(ephys_time_2d)]  # with excluding NaNs loses one dimension
    lid_out = lid_out[~np.isnan(ephys_time_2d)]  # with excluding NaNs loses one dimension
    task_out = task_out[~np.isnan(ephys_time_2d)]  # with excluding NaNs loses one dimension
    if verbose: print(f'SELECTED EPHYS shape: {SEL_ephys_3d.shape}')
    if verbose: print(f'SELECTED TIME shape: {SEL_time_2d.shape}')
    ### ASSUMES SAME NAN PATTERN IN ALL BANDS


    ### SELECTION-6: ravel with UNIQUE TIMES left after masking and nan-removal
    uniq_times, uniq_idx = np.unique(SEL_time_2d, return_index=True)
    uniq_ephys_2d = SEL_ephys_3d[uniq_idx, :]
    if verbose: print(f'{len(SEL_time_2d)} time-samples, '
                      f'{len(uniq_times)} UNIQUE times, '
                      f'{uniq_ephys_2d.shape} ({len(uniq_times) / 2048} sec)')
    lid_out = lid_out[uniq_idx]
    task_out = task_out[uniq_idx]
    

    return uniq_ephys_2d, uniq_times, lid_out, task_out


def plot_check_ephys_selection(
    sel_times, sel_ephys, sel_cdrs, sel_task
):
    # plot all three returned data types

    fig, axes = plt.subplots(3, 1, sharex='col')

    axes[0].scatter(sel_times/60, sel_cdrs, s=5,)
    axes[0].set_ylabel('Dyskinesia\n(CDRS sum)',
                    weight='bold',)

    axes[1].scatter(sel_times/60, sel_task, s=5,)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['rest', 'tap', 'free'],
                            weight='bold',)

    axes[2].plot(sel_times/60, sel_ephys,)
    axes[2].set_ylabel('SSD signal\n(a.u.)', weight='bold',)
    # axes[2].legend(frameon=False, loc='upper center',
    #             bbox_to_anchor=(.5, -.5),
    #             ncol=4,)

    for ax in axes:
        ax.tick_params(size=12, labelsize=12, axis='both',)
        ax.set_xlabel ('Time (after L-Dopa)', weight='bold',)

    plt.show()