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
                                        correct_acc_class,
                                        load_ft_ext_cfg,
                                        get_avail_ssd_subs)
from lfpecog_analysis.ft_processing_helpers import (
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS
)



def create_sub_movement_psds(sub, data_version='v4.0', ft_version='v4',
                             states_to_save=['tap', 'rest_no_move',
                                             'free_no_move', 'free_move'],
                             custom_tappers=['105', '010', '017'],):
    """
    saved numpy files witih movement vs no movement per tasks
    (only STN processed?)

    20.12 updated to create continuous raw-ssd-timeseries
    with parallel cdrs, movement, tap, and task data 
    """

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
    
    if len(states_to_save) == 1 and 'tap' in states_to_save:
            if np.logical_and(f'{sub}_lfp_left_tapSigs.npy' in files,
                              f'{sub}_lfp_right_tapSigs.npy' in files):
                print(f'skip tap-detection for sub-{sub}')
                return None
            

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
        for i_bw, bw in enumerate(ssd_sub.settings['SPECTRAL_BANDS'].keys()):
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
            # _, _, move_side_bool = custom_tap_finding(
            #     acc, acc_side=acc_side, move_type='move',
            # )
            # _, _, othermove_side_bool = custom_tap_finding(
            #     acc, acc_side=lfp_side, move_type='move',
            # )
            # move_total_bool = np.logical_or(move_side_bool,
            #                                 othermove_side_bool)

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
        
        if np.logical_and('tap' in states_to_save,
                          (f'{sub}_lfp_{lfp_side}_tapSigs.npy'
                           not in os.listdir(results_path))):
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

        if np.logical_and('rest_no_move' in states_to_save,
                          (f'{sub}_lfp_{lfp_side}_restNoMoveSigs.npy'
                           not in os.listdir(results_path))):
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
            if np.logical_and('free_no_move' in states_to_save,
                          (f'{sub}_lfp_{lfp_side}_freeNoMoveSigs.npy'
                           not in os.listdir(results_path))):
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
            if np.logical_and('free_move' in states_to_save,
                          (f'{sub}_lfp_{lfp_side}_freeMoveSigs.npy'
                           not in os.listdir(results_path))):
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
    """
    returns data and times after excluding specific tasks
    """
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
                       DATA_VERSION='v4.0',
                       ONLY_KEEP_TAP_TAPS: bool = True,):
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

    # create dict with times, task, and taps
    temp = {'dopa_time': acc.times,
            'times_min': acc.times / 60}
    for col in ['left_tap', 'right_tap', 'task']:
        idx = np.where([c == col for c in acc.colnames])[0][0]
        temp[col] = acc.data[:, idx]
    # correct tap bools for custom tappers
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
    if ONLY_KEEP_TAP_TAPS:
        sel = temp['task'] != 'tap'
        for s in ['left_tap', 'right_tap']: temp[s][sel] = 0

    # create sum bool for tap in one of the sides
    temp['tap_all'] = np.logical_or(temp['right_tap'],
                                    temp['left_tap'])
    
    return temp


def get_tap_times(tap_dict=False, sub=None, tap_border_sec=1,
                  return_in_secs=False, DATA_VERSION='v4.0',):
    """
    Returns start and end timings for all taps,
    regardless of body side. 
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
    gives BOOL to remove data in between the given
    timestamps

    Input:
        - starts and ends: timestamps for blocks of tapping
        - win_times: ephys times, should be same unit as timestamps
    
    Returns:
        - win_sel: boolean array, the moments BETWEEN the
            start- and end-stamps are set to ZERO,
            the rest is ONES
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


def create_move_specific_ephys(
    sub: str,
    FT_VERSION='v6',
    CATEG_CDRS: bool = False,
    custom_tappers=['105', '017', '010'],
    TAP_BORDER_sec=0,
    ADD_TO_CLASS=False, self_class=None,
    verbose: bool = False,
):
    """
    get boolean arrays for movement labels corresponding to ephys
        
    Arguments:  
        - FT_VERSION: ..
        - TAP_BORDER_sec: seconds that will be included in bool
            prior and after start/end of single tap-epoch
        - custom_tappers: default list
    only includes taps DURING TAP TASK

    """
    # import here to prevent circular import loading prepMovePSD elsewhere
    from lfpecog_features.get_ssd_data import get_subject_SSDs

    SETTINGS = load_ft_ext_cfg(FT_VERSION=FT_VERSION)
    DATA_VERSION = SETTINGS['DATA_VERSION']
    # SUBS = get_avail_ssd_subs(DATA_VERSION=SETTINGS["DATA_VERSION"],
    #                           FT_VERSION=FT_VERSION)
    # use merged-data v4.0 for creation of v4.2
    if DATA_VERSION == 'v4.2': DATA_VERSION = 'v4.0'
    WINLEN_SEC = SETTINGS['WIN_LEN_sec']
    # exec_subs = ['107', ] #['101', '105', '008', '010', '021']
    
    # for sub in SUBS:
    print(f'\nstart sub-{sub}')
    # possibly exclude subjects
    print('...loading SSDs')
    # get ephys data and times
    ssd_sub = get_subject_SSDs(sub=sub,
                                incl_stn=True,
                                incl_ecog=True,
                                settings=SETTINGS,)
    if ADD_TO_CLASS: setattr(self_class,
                             'ssd_sub',
                             ssd_sub)

    # POTENTIALLY LOOP OVER EPHYS SOURCES
    src = 'lfp_left'
    temp_ssd = getattr(ssd_sub, src)
    print(f'uses only {src} for mask finding!!!')

    # create timestamps for every ephys sample in 2d array (2048 Hz)
    ephys_time_arr = np.array([
        np.arange(t, t + WINLEN_SEC, 1 / temp_ssd.fs)
        for t in temp_ssd.times
    ])
    nan_arr = np.isnan(temp_ssd.lo_beta)
    if ADD_TO_CLASS:
        setattr(self_class, 'ephys_time_arr', ephys_time_arr)
        setattr(self_class, 'fs', temp_ssd.fs)

    # get acc data for sub (side irrelevant for labels)
    print('...loading ACC')
    sub_data_path = os.path.join(
        get_project_path('data'), 'merged_sub_data',
        DATA_VERSION, f'sub-{sub}')
    fname = f'{sub}_mergedData_{DATA_VERSION}_acc_left.P'  # side does not matter for already detected bool labels
    accl = load_class_pickle(os.path.join(sub_data_path, fname))
    accl = correct_acc_class(accl)

    # get movement bools based on acc data (512 Hz)
    MOVE_BOOLS = {'no_move': accl.data[:, -1],
                  'any_move': np.sum(accl.data[:, -5:-1], axis=1) > 0,
                  'left_tap': accl.data[:, -5],
                  'left_allmove': (accl.data[:, -5] + accl.data[:, -3]) > 0,
                  'right_tap': accl.data[:, -4],
                  'right_allmove': (accl.data[:, -4] + accl.data[:, -2]) > 0}
    # exclude non-tap-task TAPs later (they contain other movement)

    # CREATE TAP BOOLS PER MOVEMENT/BODYSIDE
    if ADD_TO_CLASS:
        move_masks = {}
        for BOOL_SEL in MOVE_BOOLS.keys():
            print(f'...create movement-mask ({BOOL_SEL}) for ephys')
            starts, ends = get_start_end_times_move_epochs(
                bool_to_incl=MOVE_BOOLS[BOOL_SEL],
                acc_times_arr=accl.times,
            )
            move_masks[BOOL_SEL] = mask_movement_on_times(
                ephys_time_arr=ephys_time_arr, nan_arr=nan_arr,
                starts=starts, ends=ends, verbose=verbose,
            )
            # print to check portion in orig bool and 2d array
            print(f'move_mask CHECK for : {BOOL_SEL}')
            print(np.nansum(move_masks[BOOL_SEL]) /
                  (move_masks[BOOL_SEL].shape[0] *
                   move_masks[BOOL_SEL].shape[1] - np.sum(np.isnan(move_masks[BOOL_SEL])))
            )
            print(sum(MOVE_BOOLS[BOOL_SEL]) / len(MOVE_BOOLS[BOOL_SEL]))


    else:
        BOOL_SEL = 'any_move'
        print(f'...create movement-mask ({BOOL_SEL}) for ephys')
        starts, ends = get_start_end_times_move_epochs(
            bool_to_incl=MOVE_BOOLS[BOOL_SEL],
            acc_times_arr=accl.times,
        )
        move_mask = mask_movement_on_times(
            ephys_time_arr=ephys_time_arr, nan_arr=nan_arr,
            starts=starts, ends=ends,
            verbose=verbose)

    # # print to check portion in orig bool and 2d array
    # print(np.nansum(move_mask) / (move_mask.shape[0] * move_mask.shape[1] - np.sum(np.isnan(move_mask))))
    # print(sum(MOVE_BOOLS[BOOL_SEL]) / len(MOVE_BOOLS[BOOL_SEL]))

    # CREATE TASK MASK (corresponding to ephys 2d-data)
    print('...create task-mask for ephys')
    task_mask_times = get_mask_timings(
        orig_labels=accl.data[:, 4],
        orig_times=accl.times,
        MASK='TASK',
    )
    # currently codes: rest=0, tap=1, free=2
    task_mask = create_ephys_mask(ephys_time_arr=ephys_time_arr,
                                    ephys_win_times=temp_ssd.times,
                                    mask_times=task_mask_times,
                                    MASK='TASK',)
    
    # CREATE CDRS MASK (corresponding to ephys 2d-data)
    print('...create CDRS-mask for ephys')
    cdrs = find_select_nearest_CDRS_for_ephys(
        sub=sub, side='bilat',  
        ft_times=np.array(temp_ssd.times) / 60,
        INCL_CORE_CDRS=True,
        cdrs_rater='Patricia',
    )
    if CATEG_CDRS:
        cdrs = categorical_CDRS(cdrs,
                                cutoff_mildModerate=3.5,
                                cutoff_moderateSevere=7.5,)
    # get actual mask
    lid_mask_times = get_mask_timings(
        orig_labels=cdrs,
        orig_times=temp_ssd.times,
        MASK='LID',
    )
    # currently codes: rest=0, tap=1, free=2
    lid_mask = create_ephys_mask(ephys_time_arr=ephys_time_arr,
                                    ephys_win_times=temp_ssd.times,
                                    mask_times=lid_mask_times,
                                    MASK='LID',)
    
    if not ADD_TO_CLASS:
        assert (temp_ssd.lo_beta.shape == task_mask.shape
                == lid_mask.shape == move_mask.shape), (
            'one of masks or ephys data does not match shapes'
        )
        return move_mask, task_mask, lid_mask
    
    if ADD_TO_CLASS:
        assert (temp_ssd.lo_beta.shape == task_mask.shape
                == lid_mask.shape == list(move_masks.values())[0].shape), (
            'one of masks or ephys data does not match shapes: '
            # f'SSD-data: {}, task: {task_mask.shape}, '
        )
        
        setattr(self_class, 'move_masks', move_masks)
        setattr(self_class, 'task_mask', task_mask)
        setattr(self_class, 'lid_mask', lid_mask)
        print('added all masks to class')



def get_start_end_times_move_epochs(
    bool_to_incl, acc_times_arr
):
    """
    replacement for tap times, now handles all
    movement bools

    finds timings (seconds in dopatime) of starts
    and ends of positive movement-epochs
    """
    # find start and end timestamps (sec) of total tapping
    starts = np.where(np.diff(bool_to_incl.astype(int)) == 1)[0]
    ends = np.where(np.diff(bool_to_incl.astype(int)) == -1)[0]

    # add starts and ends as indices
    if bool_to_incl[0]:
        starts = np.concatenate([np.atleast_1d([0]), starts])
    if bool_to_incl[-1]:
        ends = np.concatenate([ends, np.atleast_1d([-1])])

    # convert to times
    starts = acc_times_arr[starts]
    ends = acc_times_arr[ends]  # still 512 Hz        

    return starts, ends


def mask_movement_on_times(
    ephys_time_arr, starts, ends, nan_arr=None,
    verbose: bool = False,
):
    """
    creates a bool nd-array (like ephys data shape) that
    indicates whether the ephys data should be included
    for the given movement boolean used to get the
    starts and ends times

    Arguments:
        - ephys_time_arr: nd.array corresponding to 2d ephys-array
            (with SSD timeseries)
        - starts: timestamps of beginnings of movement-states,
            generated with get_start_end_times_move_epochs
        - ends: same
        - nan_arr: if given, has same shape as time_arr and
            corresponds with nans in ephys data
    """
    if isinstance(nan_arr, np.ndarray):
        assert nan_arr.shape == ephys_time_arr.shape, (
            'nan_arr and time_arr have different shapes'
        )
    
    move_mask = np.zeros_like(ephys_time_arr)
    skipped_wins = 0
    
    for i_row, row in enumerate(ephys_time_arr):
        if verbose: print(row[0], row[-1])
        # find starts and ends within window
        temp_starts = starts[np.logical_and(starts>row[0],
                                            starts<row[-1])]
        temp_ends = ends[np.logical_and(ends>row[0],
                                        ends<row[-1])]
        if verbose: print(temp_starts, temp_ends)
        temp_starts = [np.argmin(abs(row - t)) for t in temp_starts]
        temp_ends = [np.argmin(abs(row - t)) for t in temp_ends]

        # when no starts or endings were found in window
        if len(temp_ends) == len(temp_starts) == 0:
            skipped_wins += 1
            if len(starts[starts < row[0]]) == 0: continue  # no start prior, bool stays negative
            if len(ends[ends < row[0]]) == 0:
                move_mask[i_row, :] = 1  # ongoing positive bool (only start found)
                continue  # no start prior, bool stays negative
            # find closest prior start or end
            dist_start = min(abs(starts[starts < row[0]] - row[0]))
            dist_end = min(abs(ends[ends < row[0]] - row[0]))
            if dist_start < dist_end: move_mask[i_row, :] = 1  # ongoing positive bool
            # ongoing negative bool (is already zero)

        # when multiple starts and ends were found
        while len(temp_starts) > 0 and len(temp_ends) > 0:
            if verbose: print('while',temp_starts, temp_ends)
            if verbose: print('masked', sum(move_mask[i_row]) / 2048)
            # fill bool for ongoing start
            if temp_ends[0] < temp_starts[0]:
                move_mask[i_row, :temp_ends[0]] = 1
                if verbose: print(f'masks first end: {temp_ends[0]/2048}')
                temp_ends = temp_ends[1:]
                continue
            # fill bool for ongoing end
            if temp_starts[-1] > temp_ends[-1]:
                move_mask[i_row, temp_starts[-1]:] = 1
                if verbose: print(f'masks last start: {10-(temp_starts[-1]/2048)}')
                temp_starts = temp_starts[:-1]

                continue
            # in between matching starts-ends
            for i1, i2 in zip(temp_starts, temp_ends):
                move_mask[i_row, i1:i2] = 1
                if verbose: print(f'mask between {i1}:{i2} = {(i2-i1)/2048}')
            temp_starts, temp_ends = [], []
            break
        # fill open starts or endings that are left
        if len(temp_starts) > 0: move_mask[i_row, temp_starts[0]:] = 1
        if verbose: print('masked', sum(move_mask[i_row]) / 2048)
        if len(temp_ends) > 0: move_mask[i_row, :temp_ends[0]:] = 1
        if verbose: print('masked', sum(move_mask[i_row]) / 2048)

        if verbose: print(sum(move_mask[i_row]), sum(move_mask[i_row]) / 2048)
        if verbose: print()

    if verbose: print(f'SKIPPED WINDOWS: {skipped_wins}')

    # set nan values in ephys to nan in move bool
    if isinstance(nan_arr, np.ndarray):
        move_mask[nan_arr] = np.nan

        
    return move_mask


def get_mask_timings(orig_labels,
                        orig_times,
                        MASK: str = 'TASK'):
    
    if MASK.upper() == 'TASK':
        label_times = {'rest': [],
                    'tap': [],
                    'free': []}

    elif MASK.upper() in ['CDRS', 'LID']:
        label_times = {
            c: [] for c in np.unique(orig_labels)
        }

    assert orig_labels[0] in list(label_times.keys()), (
            'orig_labels does not contain tasks/cdrs'
        )

    for i_task, task in enumerate(orig_labels):
        # set start task (no switch)
        if i_task == 0:
            start_t = orig_times[0]
            continue
        # include end task (no switch)
        elif i_task == len(orig_labels)-1:
            end_t = orig_times[i_task]
            label_times[orig_labels[i_task-1]].append([start_t, end_t])
            continue
        # only proceed when there is a change
        if task == orig_labels[i_task-1]: continue
        # add current start and time to correct task
        end_t = orig_times[i_task]
        label_times[orig_labels[i_task-1]].append([start_t, end_t])
        # print(f'add [{start_t}, {end_t}] to {orig_labels[i_task-1]}')

        # start new task period with current time
        start_t = orig_times[i_task]

    return label_times


def create_ephys_mask(ephys_time_arr,
                      ephys_win_times,
                      mask_times,
                      MASK: str = 'TASK'):
    # create 2d array mask corr to ephys data
    mask_arr = np.zeros_like(ephys_time_arr)

    for i_task, task in enumerate(mask_times.keys()):
        # define coding for task OR cdrs score
        if MASK == 'TASK': task_code = i_task
        elif MASK in ['LID', 'CDRS']: task_code = task
        # set code to all windows amtching the mask-time category
        task_bool = [
            any([np.logical_and(t>=t_couple[0], t<=t_couple[1])
                 for t_couple in mask_times[task]])
            for t in ephys_win_times
        ]  # is true if the window time was in ANY of the specifc task's timings

        mask_arr[task_bool, :] = task_code  # code the whole selected windows for task
        print(f'masked-category {task} is coded: {task_code}')
    
    return mask_arr