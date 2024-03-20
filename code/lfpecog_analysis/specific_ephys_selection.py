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
from scipy.signal import welch

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
from lfpecog_features.feats_spectral_features import (
    calc_coherence
)
from lfpecog_features.feats_spectral_helpers import (
    get_indiv_band_peaks
)
# from lfpecog_analysis.psd_analysis_classes import PSD_vs_Move_sub


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
    EPHYS_FS: int = 2048,
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
                    'VOLUNTARY', 'INVOLUNTARY',
                    'FREENOMOVE', 'FREEMOVE']
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


    ephys_3d = getattr(psdMoveClass, f'{ephys_source}_3d').copy()
    ephys_time_2d = psdMoveClass.ephys_time_arr.copy()
    move_masks=psdMoveClass.move_masks
    task_mask=psdMoveClass.task_mask.copy()

    # if unilateral dyskinesia analysis is required
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
    elif 'FREE' in SEL:
        TASK = [2]  # only select free
    else:
        TASK == [0, 1, 2]  # select all

    for T in TASK: MASK[task_mask == T] = 1
    if verbose: print(f'total KEEPS in MASK after TASK : {np.sum(MASK) / 7 / EPHYS_FS} (sec, p/band)')

    # select only first X minutes for baseline
    if SEL == 'BASELINE':
        time_mask = ephys_time_2d < (BASELINE_MINUTES * 60)
        MASK = np.logical_and(MASK, time_mask)  # combine with existing mask-selection
        if verbose: print(f'total KEEPS in MASK after baseline time : {np.sum(MASK) / 7 / EPHYS_FS} (sec, p/band)')


    ### SELECTION-2: add 2d-mask for MOVEMENT SELECTION
    if 'REST' in SEL or SEL in ['BASELINE', 'FREENOMOVE']:
        mov_mask = move_masks['no_move']  # select move-mask
        if verbose: print('\t...all movement excluded in rest or baseline or freenomove')
    
    # add taps per defined side to mask
    elif SEL == 'VOLUNTARY':
        mov_mask = np.zeros_like(MASK)
        for mask_side in ['left', 'right']:
            if SEL_bodyside in ['both', mask_side]:
                mov_mask[move_masks[f'{mask_side}_tap'] == 1] = 1
        
    # add most ensured dyskinesia moments
    # above only rest-task selected, now add all movements
    elif SEL in ['INVOLUNTARY', 'FREEMOVE']:  # FREEMOVE comes as left/right move
        mov_mask = np.zeros_like(MASK)
        for mask_side in ['left', 'right']:
            if SEL_bodyside in ['both', mask_side]:
                mov_mask[move_masks[f'{mask_side}_allmove'] == 1] = 1
        
    MASK = np.logical_and(MASK, mov_mask)  # combine with existing mask-selection
    if verbose: print(f'total KEEPS in MASK after MOVE: {np.sum(MASK) / 7 / EPHYS_FS} (sec, p/band)')


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
    if verbose: print(f'total KEEPS in MASK after LID: {np.sum(MASK) / 7 / EPHYS_FS} (sec, p/band)')

    # optionally exclude ipsilat-Dysk IPSI-lat to ECoG
    if isinstance(EXCL_ECOG_IPSILAT, np.ndarray):  # if defined above 
        MASK = np.logical_and(MASK, ~EXCL_ECOG_IPSILAT)  # unilat ecog array is positive for excl
        if verbose: print(f'total KEEPS in MASK after uni-lat-ipsi-lat-ecog '
                          f'exclusion: {np.sum(MASK) / 7 / EPHYS_FS} (sec, p/band)')


    ### SELECTION-4: APPLY MASKs on all ephys AND times
    ephys_3d[MASK == 0, :] = np.nan   # masks all excluded samples with nan
    ephys_time_2d[MASK == 0] = np.nan

    if verbose: print(f'total NaNs in 3d after MASK: '
                      f'{np.sum(np.isnan(ephys_3d)) / 7 / ephys_3d.shape[0] / EPHYS_FS}'
                      ' sec per 10-sec window (mean)')
    
    # masks windows to exclude based on NaNs with full-row-nans
    ephys_3d[nan_win_sel, :, :] = np.nan
    ephys_time_2d[nan_win_sel, :] = np.nan
    if verbose: print(f'total NaNs in 3d after NaN-window-masking: '
                        f'{np.sum(np.isnan(ephys_3d)) / 7 / ephys_3d.shape[0] / EPHYS_FS}'
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
                      f'{uniq_ephys_2d.shape} ({len(uniq_times) / EPHYS_FS} sec)')
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


def get_ssd_psd_from_array(
    ephys_arr, sfreq, SETTINGS,
    band_names, PSD_WIN_sec: int = 1,
    RETURN_PSD_1sec: bool = False,
    indiv_gamma_range=None,
):
    """
    extracts PSD based on specific SSDd
    frquency bands. input is selected ephys
    based on meta-data.

    Arguments:

    Returns:
        - new_f: array with PSD-freqs (4-35, 60-90)
        - new_psd: array containing PSDs corr to new_f
    """
    # skip too short samples
    if ephys_arr.shape[0] < sfreq: return [], []

    # create new freq array to return
    new_f = np.concatenate([np.arange(4, 35),
                            np.arange(60, 90)])
    
    # create NaN-array (1d for means, 2d for 1sec windows)
    if not RETURN_PSD_1sec:
        new_psd = np.array([np.nan] * len(new_f))
    elif RETURN_PSD_1sec:
        n_wins = ephys_arr.shape[0] // sfreq  # number of a sec windows fitting in data
        new_psd = np.array([[np.nan] * len(new_f)] * n_wins)

    # calculate per freq-band
    for i, band in enumerate(band_names):
        f_range = list(SETTINGS['SPECTRAL_BANDS'].values())[i]

        if SETTINGS['FT_VERSION'] == 'v8' and band == 'gammaPeak':
            f_range = indiv_gamma_range

        sig = ephys_arr[:, i]  # select band signal (SSD)
        sig = sig[~np.isnan(sig)]  # delete NaNs

        if RETURN_PSD_1sec:
            # reshape in 2d array with 1 sec windows
            sig = sig[:sfreq * n_wins]
            try:
                sig = sig.reshape((n_wins, sfreq))
            except ValueError:
                # if one sample short this can happen (trxy with one window less)
                n_wins -= 1
                sig = sig[:sfreq * n_wins]
                sig = sig.reshape((n_wins, sfreq))
                new_psd = new_psd[:n_wins, :]
                
        # calculate PSD, is 2d if RETURN_PSD_1sec is true
        freqs, ps = welch(sig, sfreq, nperseg=sfreq * PSD_WIN_sec,)
        band_sel = [f >= f_range[0] and f < f_range[1]
                    for f in freqs]
        new_band_sel = [f >= f_range[0] and f < f_range[1]
                        for f in new_f]
        
        if not RETURN_PSD_1sec:
            new_psd[new_band_sel] = ps[band_sel]
        elif RETURN_PSD_1sec:
            new_psd[:, new_band_sel] = ps[:, band_sel]

    # check shape PSDs and freqs, check NaNs
    if not RETURN_PSD_1sec:
        assert len(new_psd) == len(new_f), (
            f'f {new_f.shape} and psd {new_psd.shape} not match'
        )
        assert ~np.any(np.isnan(new_psd), axis=0), (
            f'NaNs in new_psd ({sum(np.isnan(new_psd))})'
        )

    elif RETURN_PSD_1sec:
        assert new_psd.shape == (n_wins, len(new_f)), (
            f'f: {new_f.shape}, wins: {n_wins}, and psd: '
            f'{new_psd.shape} do not match'
        )
        assert ~np.any(np.isnan(new_psd), axis=0).any(), (
            'NaNs in new_psd (in # freqs columns: '
            f'{sum(np.any(np.isnan(new_psd), axis=0))})'
        )
    return new_f, new_psd


def get_ssd_coh_from_array(
    ephys_arr1, ephys_arr2, sfreq,
    band_names, SETTINGS=None, f_range_list = None, 
    PSD_WIN_sec: int = 1, RETURN_PSD_1sec: bool = False,
    indiv_gamma_range=None,
):
    """
    extracts Coherences based on specific SSDd
    frquency bands. input is selected ephys
    based on meta-data.

    Arguments:

    Returns:
        - new_f: array with PSD-freqs (4-35, 60-90)
        - new_psd: array containing PSDs corr to new_f
    """
    # skip too short samples
    if ephys_arr1.shape[0] < sfreq: return [], [], []

    if SETTINGS: f_range_list = list(SETTINGS['SPECTRAL_BANDS'].values())

    # create new freq array to return
    new_f = np.concatenate([np.arange(4, 35),
                            np.arange(60, 90)])
    
    # create NaN-array (1d for means, 2d for 1sec windows)
    if not RETURN_PSD_1sec:
        new_sqcoh = np.array([np.nan] * len(new_f))
        new_icoh = np.array([np.nan] * len(new_f))
    elif RETURN_PSD_1sec:
        win_samples = sfreq * PSD_WIN_sec
        n_wins = ephys_arr1.shape[0] // win_samples  # number of a sec windows fitting in data
        new_sqcoh = np.array([[np.nan] * len(new_f)] * n_wins)
        new_icoh = np.array([[np.nan] * len(new_f)] * n_wins)

    # calculate per freq-band
    for i, band in enumerate(band_names):
        f_range = f_range_list[i]

        if SETTINGS['FT_VERSION'] == 'v8' and band == 'gammaPeak':
            f_range = indiv_gamma_range

        sig1 = ephys_arr1[:, i]  # select band signal (SSD)
        sig2 = ephys_arr2[:, i]  # select band signal (SSD)
        nan_sel = np.logical_or(np.isnan(sig1), np.isnan(sig2))
        sig1 = sig1[~nan_sel]  # delete NaNs
        sig2 = sig2[~nan_sel]  # delete NaNs

        if RETURN_PSD_1sec:
            # reshape in 2d array with 1 sec windows
            sig1 = sig1[:win_samples * n_wins]
            sig2 = sig2[:win_samples * n_wins]
            try:
                sig1 = sig1.reshape((n_wins, win_samples))
                sig2 = sig2.reshape((n_wins, win_samples))
            except ValueError:
                # if one sample short this can happen (trxy with one window less)
                n_wins -= 1
                for sig in [sig1, sig2]:
                    sig = sig[:win_samples * n_wins]
                    sig = sig.reshape((n_wins, win_samples))
                new_sqcoh = new_sqcoh[:n_wins, :]
                new_icoh = new_icoh[:n_wins, :]

            # if no data left
            if sig1.shape[0] == 0 or sig2.shape[0] == 0:
                return [], [], []
            # if only one epoch
            elif len(sig1.shape) == 1 or len(sig2.shape) == 1:
                freqs, _, icoh, _, sqcoh = calc_coherence(
                    sig1=sig1, sig2=sig2, fs=sfreq,
                    nperseg=sfreq,
                )
                icoh = np.atleast_2d(icoh)
                sqcoh = np.atleast_2d(sqcoh)
                if icoh.shape[0] > icoh.shape[1]: icoh = icoh.T
                if sqcoh.shape[0] > sqcoh.shape[1]: sqcoh = sqcoh.T
            # if multiple epochs in array
            else:
                icoh_list, coh_list = [], []

                # calculate Coherences per epoch
                for s1, s2 in zip(sig1, sig2):
                    freqs, _, icoh, _, sqcoh = calc_coherence(
                        sig1=s1, sig2=s2, fs=sfreq,
                        nperseg=sfreq,
                    )
                    icoh_list.append(icoh)
                    coh_list.append(sqcoh)
                icoh = np.array(icoh_list)
                sqcoh = np.array(coh_list)
        
        # do not return single epochs
        else:        
            # calculate Coherences
            freqs, _, icoh, _, sqcoh = calc_coherence(
                sig1=sig1, sig2=sig2, fs=sfreq,
                nperseg=sfreq,
            )

        # select out relevant freqs related to ssd bands
        band_sel = [f >= f_range[0] and f < f_range[1]
                    for f in freqs]
        new_band_sel = [f >= f_range[0] and f < f_range[1]
                        for f in new_f]
        
        if not RETURN_PSD_1sec:
            new_sqcoh[new_band_sel] = sqcoh[band_sel]
            new_icoh[new_band_sel] = icoh[band_sel]
        elif RETURN_PSD_1sec:
            new_sqcoh[:, new_band_sel] = sqcoh[:, band_sel]
            new_icoh[:, new_band_sel] = icoh[:, band_sel]

    # check shape PSDs and freqs, check NaNs
    if not RETURN_PSD_1sec:
        assert len(new_sqcoh) == len(new_f), (
            f'f {new_f.shape} and psd {new_sqcoh.shape} not match'
        )
        assert ~np.any(np.isnan(new_sqcoh), axis=0), (
            f'NaNs in new_psd ({sum(np.isnan(new_sqcoh))})'
        )

    elif RETURN_PSD_1sec:
        assert new_sqcoh.shape == (n_wins, len(new_f)), (
            f'f: {new_f.shape}, wins: {n_wins}, and COH: '
            f'{new_sqcoh.shape} do not match'
        )
        assert ~np.any(np.isnan(new_sqcoh), axis=0).any(), (
            'NaNs in new_psd (in # freqs columns: '
            f'{sum(np.any(np.isnan(new_sqcoh), axis=0))})'
        )

    print(f'...COH: return shapes {len(new_f)}, {new_icoh.shape}, {new_sqcoh.shape}')

    return new_f, new_sqcoh, new_icoh


def get_hemisphere_movement_location(
    SRC, cond, sub=None, ecog_sides=None,
):
    """
    find hemisphere location related to
    movement (IPSI or CONTRA lateral)

    sub and ecog_sides only required for
    ecog localisations
    """
    # find none movement conditions
    if not 'tap' in cond and not 'move' in cond: return False, False

    # define left or right hemisphere of current data
    if SRC.startswith('lfp'):
        hemisph = SRC.split('_')[1]  # left or right (stn)
    else:
        hemisph = ecog_sides[sub]
    
    # define left or right body side movement
    if 'tapleft' in cond.lower() or 'dyskmoveleft' in cond.lower():
        move_side = 'left'
    elif 'tapright' in cond.lower() or 'dyskmoveright' in cond.lower():
        move_side = 'right'
    elif 'dyskmoveboth' in cond.lower():
        move_side = 'bilat'
    
    # define contra or ipsi lateral movement
    if hemisph == move_side:
        IP_CON = 'IPSI'
        # print(f'....{attr}, hemisph: {hemisph} x {move_side} move: {IP_CON}')
    elif move_side == 'bilat':
        IP_CON = 'BILAT'
    else: IP_CON = 'CONTRA'
    
    # define movement type
    if 'tap' in cond: MOVETYPE = 'TAP'
    elif 'dyskmove' in cond: MOVETYPE = 'INVOLUNT'
    
    # print(f'\n...in {cond}, {attr}, hemisphere {hemisph} and moveside '
    #         f'{move_side} lead to {MOVETYPE} {IP_CON}')
    
    return IP_CON, MOVETYPE

