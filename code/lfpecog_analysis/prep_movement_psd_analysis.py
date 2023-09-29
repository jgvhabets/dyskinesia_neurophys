"""
Analyze movement voluntary effect
"""
# import functions and packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch

# import own functions
# import lfpecog_analysis.load_SSD_features as load_ssd_fts
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle,
                                        correct_acc_class)




def create_sub_movement_psds(sub, data_version='v4.0', ft_version='v4',
                             states_to_save=['tap', 'rest_no_move',
                                             'free_no_move', 'free_move'],
                             custom_tappers=['105', '010', '017'],):
    # TODO: ADD CHECK FOR EXISTING DATA
    import lfpecog_features.get_ssd_data as ssd
    import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq

    # define main directory with stored merged data
    main_data_path = os.path.join(get_project_path('data'),
                                  'merged_sub_data', data_version)
    results_path = os.path.join(get_project_path('results'), 'features',
                                  f'SSD_feats_broad_{ft_version}',
                                  data_version,
                                  'windows_10s_0.5overlap_tapRest')
    if not os.path.exists(results_path): os.makedirs(results_path)
    
    if sub.startswith('1'): n_sources = 2
    elif sub.startswith('0'): n_sources = 3
    files = os.listdir(results_path)
    if sum([f'{sub}' in f for f in files]) == (4 * n_sources):
        print(f'...\nSKIP sub-{sub}, all files present')
        return
    
    # load all SSD timeseries for subject
    ssd_sub = ssd.get_subject_SSDs(sub=sub,
                                   incl_stn=True, incl_ecog=False,
                                   ft_setting_fname=f'ftExtr_spectral_{ft_version}.json',)
    data_dict = {}  # store all states in to save

    # define lateralities
    for lfp_side, acc_side in zip(['left', 'right',],
                                  ['right', 'left']):
        files = os.listdir(results_path)
        if sum([f'{sub}_lfp_{lfp_side}' in f for f in files]) == 4:
            print(f'...\nSKIP LFP-{lfp_side}, ACC-{acc_side} (sub-{sub}) (present)')
            continue
        
        print(f'...\nstart LFP-{lfp_side}, ACC-{acc_side} (sub-{sub})')
        # # exclude combinations without tapping performed
        # if (sub, acc_side) in [('017', 'right'),
        #                     #    ('017', 'left'),  # for left tapping 017 no_move has to be checked
        #                        ]:
        #     print(f'skip acc-side {acc_side} for sub-{sub} due to no tapping')

        # get lfp data
        lfp = getattr(ssd_sub, f'lfp_{lfp_side}')

        # get ACC-data
        sub_data_path = os.path.join(main_data_path, f'sub-{sub}')
        fname = (f'{sub}_mergedData_{data_version}'
                 f'_acc_{acc_side}.P')  # left and right contain both bilateral labels
        acc = load_class_pickle(os.path.join(sub_data_path, fname))

        # correct for incorrect timestamps in v4 subject 103 (Wrong day in json)
        if sub == '103' and max(acc.times) > 2e6:
            setattr(acc, 'times',
                    np.array(acc.times) - np.float64(27 * 24 * 60 * 60))
            print('corrected 103 ACC timings')

        # create 2d array with bands
        for i_bw, bw in enumerate(['delta', 'alpha', 'lo_beta',
                                   'hi_beta', 'gamma']):
            bw_sig, sig_times, _ = ssd_TimeFreq.get_cont_ssd_arr(
                subSourceSSD=lfp, bw=bw,
                winLen_sec=ssd_sub.settings['WIN_LEN_sec'],
            )
            if i_bw == 0: lfp_arr = np.atleast_2d(bw_sig)
            else: lfp_arr = np.concatenate([lfp_arr, np.atleast_2d(bw_sig)], axis=0)
        # merge into array
        lfp_arr = np.array(lfp_arr)

        # get tapping or movement times
        if ssd_sub.sub in custom_tappers:
            _, _, tap_bool = custom_tap_finding(
                acc, acc_side=acc_side, move_type='tap',
            )
            _, _, move_side_bool = custom_tap_finding(
                acc, acc_side=acc_side, move_type='move',
            )
            _, _, othermove_side_bool = custom_tap_finding(
                acc, acc_side=lfp_side, move_type='move',
            )
            move_total_bool = np.logical_or(move_side_bool,
                                            othermove_side_bool)

        else:
            i_col_sidetap = [i for i, c in enumerate(acc.colnames)
                            if c == f'{acc_side}_tap'][0]
            tap_bool = acc.data[:, i_col_sidetap]
            
            # to exclude contralat taps from no-tap (rest)
            i_col_sidemove = [i for i, c in enumerate(acc.colnames)
                            if c == f'{acc_side}_move'][0]
            move_side_bool = acc.data[:, i_col_sidemove]
            move_side_bool = np.logical_or(move_side_bool, tap_bool)

            # to find moments without any movement
            i_col_no_move = [i for i, c in enumerate(acc.colnames)
                            if c == 'no_move'][0]
            move_total_bool = ~acc.data[:, i_col_no_move]
        
        # exclude free-tasks for tap vs rest analysis
        i_col_task = [i for i, c in enumerate(acc.colnames)
                      if c == 'task'][0]
        
        if 'tap' in states_to_save:
            # get tap task data
            tap_lfp_arr, tap_sig_times = excl_specific_task(
                data_arr=lfp_arr.copy(),
                data_times=sig_times.copy(),
                task_arr=acc.data[:, i_col_task],
                task_times=acc.times,
                task_to_excl=['rest', 'free']
            )
            # select out actual tap-movements data
            data_dict['tap'] = select_taps_in_data(
                neural_data=tap_lfp_arr,
                neural_times=tap_sig_times,
                tap_bool=tap_bool,
                tap_bool_times=acc.times,
                return_nontap=False,
            )
            print('...added tap')

        if 'rest_no_move' in states_to_save:
            # get no-tap, only rest, no movement data
            rest_lfp_arr, rest_sig_times = excl_specific_task(
                data_arr=lfp_arr.copy(),
                data_times=sig_times.copy(),
                task_arr=acc.data[:, i_col_task],
                task_times=acc.times,
                task_to_excl=['tap', 'free']
            )
            # excl movement moments (take 2nd return var)
            _, data_dict['restNoMove'] = select_taps_in_data(
                neural_data=rest_lfp_arr,
                neural_times=rest_sig_times,
                tap_bool=tap_bool,
                tap_bool_times=acc.times,
                return_nontap=True,
                move_excl_bool=move_total_bool,
                move_excl_times=acc.times,
                neural_fs=lfp.fs,
                margin_around_tap_sec=2,
            )
            print('...added rest no move')
        
        if 'free_move' in states_to_save or 'free_no_move' in states_to_save:
            # get no-tap, only rest, no movement data
            free_lfp_arr, free_sig_times = excl_specific_task(
                data_arr=lfp_arr.copy(),
                data_times=sig_times.copy(),
                task_arr=acc.data[:, i_col_task],
                task_times=acc.times,
                task_to_excl=['tap', 'rest']
            )
            if 'free_no_move' in states_to_save:
                # excl movement moments (take 2nd return var)
                _, data_dict['freeNoMove'] = select_taps_in_data(
                    neural_data=free_lfp_arr,
                    neural_times=free_sig_times,
                    tap_bool=tap_bool,
                    tap_bool_times=acc.times,
                    return_nontap=True,
                    move_excl_bool=move_total_bool,
                    move_excl_times=acc.times,
                    neural_fs=lfp.fs,
                    margin_around_tap_sec=2,
                )
                print('...added free no move')
            if 'free_move' in states_to_save:
                # excl movement moments (take 2nd return var)
                data_dict['freeMove'] = select_taps_in_data(
                    neural_data=free_lfp_arr,
                    neural_times=free_sig_times,
                    tap_bool=move_side_bool,
                    tap_bool_times=acc.times,
                    return_nontap=False,
                )
                print('...added free move')

        print(f'Saving sub-{sub} lfp-{lfp_side}, states: {data_dict.keys()}')
        for state in data_dict.keys():
            print(f'{state}, shape: {data_dict[state].shape}')
            np.save(file=os.path.join(results_path,
                                    f'{sub}_lfp_{lfp_side}_{state}Sigs.npy'),
                    arr=data_dict[state], allow_pickle=True)





def excl_specific_task(data_arr, data_times,
                       task_arr, task_times, task_to_excl):
    # convert to arrays
    if isinstance(data_times, list): data_times = np.array(data_times)
    if isinstance(task_times, list): task_times = np.array(task_times)
    if isinstance(task_arr, list): task_arr = np.array(task_arr)
    
    # set true to keep, false to exclude
    if len(data_arr.shape) == 1: incl_bool = np.ones(len(data_arr))
    elif len(data_arr.shape) == 2: incl_bool = np.ones(data_arr.shape[-1])
    
    if isinstance(task_to_excl, str): task_to_excl = [task_to_excl,]

    for task in task_to_excl:
        # find indices and times of changes (starts and ends) in task
        task_changes = np.diff((task_arr == task).astype(int)) 
        idx_task_start = np.where(task_changes== 1)[0]
        idx_task_end = np.where(task_changes == -1)[0]
        t_task_start = task_times[idx_task_start]
        t_task_end = task_times[idx_task_end]

        for t1, t2 in zip(t_task_start, t_task_end):
            # find indices in neural data for tap start and end
            i1 = np.argmin(abs(data_times - t1))
            i2 = np.argmin(abs(data_times - t2))
            incl_bool[i1:i2] = 0

    # select neural data based on bool
    if len(data_arr.shape) == 1:
        data_arr = data_arr[incl_bool.astype(bool)]
    elif len(data_arr.shape) == 2:
        data_arr = data_arr[:, incl_bool.astype(bool)]
            
    data_times = data_times[incl_bool.astype(bool)]


    return data_arr, data_times


def select_taps_in_data(
    neural_data, neural_times, tap_bool, tap_bool_times,
    margin_around_tap_sec=1, neural_fs=2048, return_nontap=False,
    move_excl_bool=False, move_excl_times=False,
):
    assert neural_data.shape[-1] == len(neural_times), (
        'length neural data and times do not match'
    )
    assert len(tap_bool) == len(tap_bool_times), (
        'length tap_bool and times do not match'
    )
    # convert to arrays
    if isinstance(neural_data, list): neural_data = np.array(neural_data)
    if isinstance(neural_times, list): neural_times = np.array(neural_times)

    # find tap start end indeices
    tap_df = np.diff(tap_bool.astype(int))
    tap_starts = np.where(tap_df == 1)[0]  # gives indices
    tap_ends = np.where(tap_df == -1)[0]
    # convert indices into time (dopa_time seconds)
    tap_starts = tap_bool_times[tap_starts]
    tap_ends = tap_bool_times[tap_ends]

    # create boolean arrays to adjust
    data_tap_bool = np.zeros(neural_data.shape[-1])  # change to positive during taps
    if return_nontap:
        data_notap_bool = np.ones(neural_data.shape[-1])  # change to negative during tap plus pad
        pad_idx = int(margin_around_tap_sec * neural_fs)

    for t1, t2 in zip(tap_starts, tap_ends):
        # find indices in neural data for tap start and end
        i1 = np.argmin(abs(neural_times - t1))
        i2 = np.argmin(abs(neural_times - t2))
        # set selected time window to positive in bool
        data_tap_bool[i1:i2] = 1

        if return_nontap:
            data_notap_bool[i1 - pad_idx:i2 + pad_idx] = 0

    if return_nontap:
        # find tap start end indices for mvoement to be removed
        tap_df = np.diff(move_excl_bool.astype(int))
        tap_starts = np.where(tap_df == 1)[0]  # gives indices
        tap_ends = np.where(tap_df == -1)[0]
        # convert indices into time (dopa_time seconds)
        tap_starts = move_excl_times[tap_starts]
        tap_ends = move_excl_times[tap_ends]

        # set no_tap_bool to negative for contra-taps
        for t1, t2 in zip(tap_starts, tap_ends):
            # find indices in neural data for tap start and end
            i1 = np.argmin(abs(neural_times - t1))
            i2 = np.argmin(abs(neural_times - t2))
            # set selected time window to negative in bool
            data_notap_bool[i1 - pad_idx:i2 + pad_idx] = 0

    # select neural data based on bool
    if len(neural_data.shape) == 1:
        tap_data = neural_data[data_tap_bool.astype(bool)]
    elif len(neural_data.shape) == 2:
        tap_data = neural_data[:, data_tap_bool.astype(bool)]

    if return_nontap:
        if len(neural_data.shape) == 1:
            no_tap_data = neural_data[data_notap_bool.astype(bool)]
        elif len(neural_data.shape) == 2:
            no_tap_data = neural_data[:, data_notap_bool.astype(bool)]

    if not return_nontap: return tap_data
    
    if return_nontap: return tap_data, no_tap_data 


def get_sub_tapTimings(sub,
                       custom_tappers=['105', '017', '010'],
                       DATA_VERSION='v4.0'):
    # use merged-data v4.0 for creation of v4.2
    if DATA_VERSION == 'v4.2': DATA_VERSION = 'v4.0'
    
    sub_data_path = os.path.join(get_project_path('data'),
                                 'merged_sub_data',
                                  DATA_VERSION,
                                  f'sub-{sub}')
    fname = (f'{sub}_mergedData_{DATA_VERSION}'
             '_acc_left.P')  # side does not matter for already detected bool labels
    # load Acc-detected movement labels
    acc = load_class_pickle(os.path.join(sub_data_path, fname))
    acc = correct_acc_class(acc)

    temp = {}
    temp['times_min'] = acc.times / 60
    for col in ['left_tap', 'right_tap', 'task']:
        idx = np.where([c == col for c in acc.colnames])[0][0]
        temp[col] = acc.data[:, idx]
    
    if sub in custom_tappers:
        for side in ['left', 'right']:
            # load sided timeseries
            fname = (f'{sub}_mergedData_{DATA_VERSION}'
                        f'_acc_{side}.P')
            acc = load_class_pickle(os.path.join(sub_data_path, fname))
            acc = correct_acc_class(acc)
            # find taps based on custom function
            _, _, taps = custom_tap_finding(
                acc, acc_side=side, move_type='tap',
            )
            temp[f'{side}_tap'] = taps
    
    # correct specific task-label error
    if sub == '108':
        sel = np.logical_and(temp['times_min'] > 15,
                             temp['times_min'] < 20)
        temp['task'][sel] = 'free'

    # delete all non tap-task taps from tap-boolean
    sel = temp['task'] != 'tap'
    for s in ['left_tap', 'right_tap']: temp[s][sel] = 0

    # create sum bool for tap in one of the sides
    temp['tap_all'] = np.logical_or(temp['right_tap'],
                                    temp['left_tap'])
    
    return temp


def get_tap_times(tap_dict=False, sub=None, tap_border_sec=1,
                  return_in_secs=False, DATA_VERSION='v4.0',):
    """
    takes dict with tap results, returns start timings
    in minutes: start and end timings of every epoch
    """
    if not tap_dict:
        tap_dict = get_sub_tapTimings(sub=sub,
                                      DATA_VERSION=DATA_VERSION,)
    # find start and end timestamps of total tapping
    starts = np.where(np.diff(tap_dict['tap_all'].astype(int)) == 1)[0]
    ends = np.where(np.diff(tap_dict['tap_all'].astype(int)) == -1)[0]

    starts = tap_dict['times_min'][starts]
    ends = tap_dict['times_min'][ends]

    # include border into tapping epoch
    starts -= (tap_border_sec/60)
    ends += (tap_border_sec/60)

    if return_in_secs:
        starts *= 60
        ends *= 60

    assert len(starts) == len(ends), (
        f'TAP-STARTS (n={len(starts)}) and ENDS '
        f'(n={len(ends)}) should have equal length'
    )

    return starts, ends


def select_taps_out_window(win_times, starts, ends):
    """
    Input:
        - starts and ends: timestamps (in minutes?!)
            for blocks of total tapping
        - win_times: should be same unit as timestamps
    
    Returns:
        - win_sel: boolean array, true if data should
            be included (none tap data), false is
            data should be excluded based on tapping
    """
    win_start = win_times[0]
    win_end = win_times[-1]
    # bool array to return
    win_sel = np.ones_like(win_times)  # if no tap in window, all is true

    if any(starts[np.logical_and(starts > win_start,
                                 starts < win_end)]):
        # select which tap epochs start within this window
        tap_sel = np.logical_and(starts > win_start,
                                 starts < win_end)
        # loop over epoch starts and ends
        for t1, t2 in zip(starts[tap_sel], ends[tap_sel]):
            # select matching window based on times
            temp_sel = np.logical_and(win_times > t1,
                                      win_times < t2)
            win_sel[temp_sel] = 0  # set tap-matching window to false

    return win_sel.astype(bool)


def custom_tap_finding(acc_class, acc_side, move_type='tap',):
    """
    custom needed due to restless legs and
    very slow hand raising

    results in indices and times of taps
    with left hand only
    Tapping movement starts circa two seconds
    before and one second after times

    Inputs: 
        - acc: acc class from pickle acc 017
        - acc_side
        - moveType: tap or move
    
    Returns:
        - peak_i
        - peak_t
        - tap_bool
    """
    fs = int(acc_class.fs)
    print(f'custom tap finding: {acc_class.sub}, {acc_side}')
    task_i = np.where([c == 'task' for c in acc_class.colnames])[0][0]

    if acc_class.sub == '017':

        if acc_side == 'left' and move_type == 'tap':
            sel = acc_class.data[:, 4] == 'tap'  # 4 is task column
            d = acc_class.data[:, 1][sel]  # 1 because of ACC_L_X
            t = acc_class.times[sel]
            idx = np.arange(acc_class.data.shape[0])[sel]

            THR=.5e-7
            peak_idx, peak_props = find_peaks(d, height=THR, distance=int(fs*5))

            peak_t = t[peak_idx]
            peak_i = idx[peak_idx]

            # create boolean positive during tap
            tap_bool = np.zeros(acc_class.data.shape[0])

            for i in peak_i:
                try:
                    if acc_class.times[i+fs] - acc_class.times[i-(2*fs)] > 3:
                        tap_bool[i - (2*fs):i] = 1
                    else: tap_bool[i - (2*fs):i+fs] = 1
                except IndexError:
                    tap_bool[i - (2*fs):i] = 1
        
        elif acc_side == 'right' and move_type == 'tap':
            tap_bool = np.zeros(acc_class.data.shape[0])
            peak_i = None
            peak_t = None
        
        elif move_type == 'move':
            dat = acc_class.data[:, 1:4]
            
            svm = np.sqrt((dat[:, 0] ** 2 +
                        dat[:, 1] ** 2 +
                        dat[:, 2] ** 2).astype(np.float64))

            if acc_side == 'left': tap_bool = svm > .8e-6
            elif acc_side == 'right': tap_bool = svm > .1e-6

            peak_i = None
            peak_t = None

        return peak_i, peak_t, tap_bool

    elif acc_class.sub == '105':
        if acc_side == 'left':
            col_idx = np.where([c == 'left_tap' for c in acc_class.colnames])[0][0]
            left_tap = acc_class.data[:, col_idx]
            return None, None, left_tap
        
        elif acc_side == 'right':
            # right acc data requried
            R_col = np.where([c == 'ACC_R_Z' for c in acc_class.colnames])[0][0]
            right_tap = np.logical_and(
                acc_class.data[:, R_col] > .25e-6,
                acc_class.data[:, task_i] == 'tap'
            )

            return None, None, right_tap
    
    elif acc_class.sub == '010':

        svm = np.sqrt(acc_class.data[:, 1].astype(np.float64)**2 +
                      acc_class.data[:, 2].astype(np.float64)**2 +
                      acc_class.data[:, 3].astype(np.float64)**2)
        # find peaks
        x_peaks, props = find_peaks(svm, height=.25e-6, distance=3*acc_class.fs)
        # find indices of tap start and end
        peak_indices = []
        margin = int(acc_class.fs/4)
        for i in x_peaks:
            i_start, i_end = False, False
            temp_i = i
            # find start: first value below THR before peak
            while not i_start:
                temp_i -= 1
                if all(svm[temp_i-margin:temp_i] < .5e-7): i_start = temp_i
            # find end: first value below THR after peak
            temp_i = i
            while not i_end:
                temp_i += 1
                if all(svm[temp_i:temp_i+margin] < .5e-7): i_end = temp_i
            # add values to peak_list
            peak_indices.append([i, i_start, i_end])
        # convert to array (n-taps x [peak, start, end])
        peak_indices = np.array(peak_indices)  # double start and ends are possible with fused taps

        # find bool-arr (considers fused taps)
        tap_bool = np.zeros_like(svm)
        for i1, i2 in zip(np.unique(peak_indices[:, 1]),
                          np.unique(peak_indices[:, 2])):
            tap_bool[i1:i2] = 1
        # remove (OR include) non-tap-task moments
        if move_type == 'tap':
            tap_bool = np.logical_and(tap_bool.astype(bool),
                                    acc_class.data[:, task_i] == 'tap')
            peak_i = np.array([i for i in peak_indices[:, 0]
                            if acc_class.data[i, task_i] == 'tap'])
            peak_t = acc_class.data[peak_i, 0]
        elif move_type == 'move':
            tap_bool = np.logical_and(tap_bool.astype(bool),
                                    acc_class.data[:, task_i] != 'tap')
            peak_i = np.array([i for i in peak_indices[:, 0]
                            if acc_class.data[i, task_i] != 'tap'])
            peak_t = acc_class.data[peak_i, 0]
                    
        return peak_i, peak_t, tap_bool