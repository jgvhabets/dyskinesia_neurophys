"""
Get accelerometer related derivatives
"""

# import 
from os import listdir
from os.path import join
from pandas import DataFrame
import numpy as np
from collections import namedtuple
from scipy.signal import resample_poly
import json

import matplotlib.pyplot as plt

from utils.utils_fileManagement import (
    load_class_pickle, get_project_path,
    make_object_jsonable
)
from lfpecog_features.moveDetection_preprocess import (
    signalvectormagn
)

def get_n_and_length_taps(win_tap_bool, acc_fs):
    """
    Get the number and duration of taps
    within a accelerometer window

    Input:
        - win_tap_bool: boolean array with tap  
            detected values (e.g. 'tap_left' column
            from data classes)
        - acc_fs: sample freq
    
    Returns:
        - n_taps: int
        - durations_sec: list with durations,
            empty list if no taps
    """
    tap_starts = np.where(np.diff(win_tap_bool) == 1)[0]
    tap_ends = np.where(np.diff(win_tap_bool) == -1)[0]

    # correct for incomplete taps at beginning or end
    if win_tap_bool[-1] == 1:
        tap_ends = np.concatenate([tap_ends.ravel(), [len(win_tap_bool) - 1]])
    if win_tap_bool[0] == 1:
        tap_starts = np.concatenate([[0], tap_starts.ravel()])

    # get number of taps
    n_taps = sum(np.diff(win_tap_bool) == 1)

    # get durations in seconds
    durations_sec = [
        (t2 - t1) / acc_fs for t1, t2
        in zip(tap_starts, tap_ends)
    ]

    return n_taps, durations_sec


Sides = namedtuple('Sides', 'left right')

def load_acc_and_task(
    sub, dataversion='v3.1',
    resample_freq=0,
):
    """
    Imports ACC classes with:
        'sub', 'data_version', 'data', 'colnames', 'times', 'fs'

    Arguments:
        - sub
        - data_version
        - resample_freq: if 0, no resampling is done

    Returns:
        - acc_both: namedtuple with left and right, containing
            dataframes with tri-axial acc-data
        - labels: dataframe containing all none-acc, none-
            ephys-columns
    """
    pickle_path = join(get_project_path('data'),
                               'merged_sub_data',
                               dataversion,
                               f'sub-{sub}',)
    files = listdir(pickle_path)

    for f in files:
        if f.endswith('.P') and 'acc' in f:
            if 'left' in f: left = f
            if 'right' in f: right = f
    # load first, to extract labels
    left = load_class_pickle(join(pickle_path, left))
    fs_left = left.fs

    label_cols = np.where(['acc' not in c.lower() and 'time' not in c.lower()
                         for c in left.colnames])[0]
    index_col = np.where(['time' in c.lower()
                         for c in left.colnames])[0][0]
    labels = DataFrame(data=left.data[:, label_cols],
                          columns=np.array(left.colnames)[label_cols],
                          index=left.data[:, index_col])

    acc_cols = np.where(['ACC' in c.upper()
                         for c in left.colnames])[0]
    left = DataFrame(data=left.data[:, acc_cols],
                        columns=np.array(left.colnames)[acc_cols],
                        index=left.data[:, index_col])
    
    right = load_class_pickle(join(pickle_path, right))
    fs_right = right.fs
    acc_cols = np.where(['ACC' in c.upper()
                         for c in right.colnames])[0]
    index_col = np.where(['time' in c.lower()
                         for c in right.colnames])[0][0]
    right = DataFrame(data=right.data[:, label_cols],
                         columns=np.array(right.colnames)[label_cols],
                         index=right.data[:, index_col])
    
    if resample_freq > 0:
        
        labels_time = resample_poly(x=labels.index.values,
                                    up=1,
                                    down=int(fs_left/resample_freq))
        labels_data = resample_poly(x=labels.values, up=1,
                                    down=int(fs_left/resample_freq))
        labels = DataFrame(labels_data, columns=labels.keys(),
                           index=labels_time)
        
        left_time = resample_poly(x=left.index.values, up=1,
                                  down=int(fs_left/resample_freq))
        left_data = resample_poly(x=left.values, up=1,
                                  down=int(fs_left/resample_freq))
        left = DataFrame(left_data, columns=left.keys(),
                         index=left_time)
        
        right_time = resample_poly(x=right.index.values, up=1,
                                   down=int(fs_right/resample_freq))
        right_data = resample_poly(x=right.values, up=1,
                                   down=int(fs_right/resample_freq))
        right = DataFrame(right_data, columns=right.keys(),
                         index=right_time)

    acc_both = Sides(left, right)
    
    return acc_both, labels


def find_task_during_time(
    timing, task_labels, task_times
):
    """
    Arguments:
        - timing (in dopa_time), timestamp
        - task_labels (array): task-column within
            extracted labels df
        - task_times (array): timestamp corresponding
            to task labels (index of labels_df)
    
    Returns:
        - task_found (str): rest, tap, or free
    """
    i = np.argmin(abs(task_times - timing))
    task_found = task_labels[i]

    return task_found


def get_task_timings(label_df, task):
    """
    Get start and end timings of tasks within
    a recording of a subject

    Arguments:
        - label_df (df): result of load_acc_and_task
        - task: str of task to give timings for
    
    Returns:
        - tuple with starts (list) and ends (list)
            of timings in dopa_time (sec)
    """

    label_arr = label_df['task'].values
    label_times = label_df.index.values

    starts = list(np.where(np.diff((label_arr == task).astype(int)) == 1)[0])
    ends = list(np.where(np.diff((label_arr == task).astype(int)) == -1)[0])

    if label_arr[0] == task: starts = [0] + starts
    if label_arr[-1] == task: ends.append(len(label_arr) - 1)

    starts = label_times[starts]
    ends = label_times[ends]

    return starts, ends


def select_task_in_features(
    ft_times, label_df, task
):
    """
    Arguments:
        - ft_times: times of feature values (in seconds!)
        - label_df: subject specific label dataframe
            imported in load_acc_and_task()
        - task (string): task to select on
    """
    # convert ft_times in seconds if needed
    if max(ft_times) < 200: ft_times *= 60

    fts_during_task = np.zeros((len(ft_times)))

    if isinstance(task, str): task = [task,]

    # repeat for all tasks given to include
    for t in task:

        starts, ends = get_task_timings(label_df, t)

        for s, e in zip(starts, ends):

            sel = np.logical_and(ft_times > s,
                                ft_times < e)

            fts_during_task[sel] = 1
    
    return fts_during_task


def define_OFF_ON_times(
    feat_times, cdrs_scores, cdrs_times,
    max_OFF_minutes=15,
    max_OFF_cdrs=0,
    min_ON_minutes=45,
    max_ON_CDRS=5,
    incl_tasks='all',
    labels_df=None,
    sub=None, data_version=None,
):
    """
    Finds binary medication moments OFF and ON, based on
    minutes after L-Dopa intake, and present CDRS scores.

    Arguments:
        - feat_times: array of timestamps corresponding
            to specific features of interest that should
            be selected (or splitted)
        - cdrs_scores, cdrs_times: arrays containing the
            total CDRS scores and corresponding timestamps
        - ... variables to set borders to OFF and ON inclusion
        - incl_tasks: string or list with task(s) to include
        - labels_df: subject specific DataFrame resulting from
            load_acc_and_task(), to select on tasks (if not default 'all')
    
    Returns:
        - off_times_sel: boolean array with length of feat_times,
            indicating which feature (time) meets the 
            requirements for inclusion in the OFF-condition
        - on_times_sel: idem for ON-condition
    """
    if isinstance(feat_times, list): feat_times = np.array(feat_times)
    # ensure all times are in minutes
    if np.nanmax(feat_times) > 200:
        feat_times /= 60
    if np.nanmax(cdrs_times) > 200:
        cdrs_times /= 60
    
    # get closest cdrs score for every feat-time
    ftwin_cdrs_score = [cdrs_scores[np.argmin(abs(cdrs_times - t))]
                        for t in feat_times]
    
    off_times_sel = [np.logical_and(m < max_OFF_minutes,
                                    ftwin_cdrs_score[i] == max_OFF_cdrs)
                     for i, m in enumerate(feat_times)]
    
    on_times_sel = [np.logical_and(m > min_ON_minutes,
                                   ftwin_cdrs_score[i] <= max_ON_CDRS)
                     for i, m in enumerate(feat_times)]
    
    # perform selection on task
    if incl_tasks != 'all':
        if not isinstance(labels_df, DataFrame):
            _, labels_df = load_acc_and_task(
                sub=sub, dataversion='v3.0',
                resample_freq=500)
            # if time-windows need to be selected on '
            # tasks, labels_df has to DataFrame resulting'
            #  from load_acc_and_task()'

        task_sel = select_task_in_features(
            ft_times=feat_times, label_df=labels_df,
            task=incl_tasks)
        # combine selection on on/off moments and included tasks
        off_times_sel = np.logical_and(off_times_sel, task_sel)
        on_times_sel = np.logical_and(on_times_sel, task_sel)
    
    return off_times_sel, on_times_sel


def get_raw_acc_traces(sub, side, data_version):
    """
    Returns pickled dataclass containing merged 
    accelerometer sub-data (attr: data, colnames,
    fs, times)
    """

    sub_data_path = join(get_project_path('data'),
                                'merged_sub_data',
                                data_version,
                                f'sub-{sub}')
    # load Acc-detected movement labels
    fname = (f'{sub}_mergedData_{data_version}'
                f'_acc_{side}.P')
    acc = load_class_pickle(join(sub_data_path, fname))

    print('PM: add FREE movement labels via '
          'tapFind.specTask_movementClassifier()'
          'and tapFind.get_move_bool_for_timeArray()')

    return acc


def get_any_resolution_acc_rms(sub, epoch_times_sec,
                               epoch_length_s,
                               data_version='v4.0'):
    
    rms = {'left': [], 'right': []}

    for side in ['left', 'right']:
        
        dat = get_raw_acc_traces(sub=sub, side=side,
                                 data_version=data_version)
        acc_cols = ['ACC' in c for c in dat.colnames]
        svm_arr = signalvectormagn(dat.data[:, acc_cols])
        rms = calc_windowed_rms(acc_sig=svm_arr, acc_times=dat.times,
                                win_times=epoch_times_sec,
                                win_len=epoch_length_s)
    
    rms = np.array([rms['left'], rms['left']], axis=0)

    return rms


def calc_windowed_rms(acc_sig, acc_times,
                      win_times, win_len):
    """
    Calculate root mean square of acc-signal for
    defined windows with starting times and
    equal window lengths
    """
    rms = []
    for t in win_times:
        t2 = t + win_len
        win_sel = np.logical_and(acc_times > t,
                                 acc_times < t2)
        rms.append(np.sqrt(np.mean(acc_sig[win_sel] ** 2)))
    
    rms = np.array(rms)

    return rms


def get_acc_rms_for_windows(sub, acc_side, featClass,
                            Z_SCORE=True, SAVE_RMS=False,):

    # select feature data
    dat = featClass.FEATS[sub]
    win_times_sec = dat.index.values * 60

    # select and load acc-data
    if acc_side in ['left', 'right']:
        MEAN_ACC = False
    elif acc_side in ['both', 'mean']:
        acc_side = 'left'
        MEAN_ACC = True

    fname = (f'{sub}_mergedData_{featClass.DATA_VERSION}'
            f'_acc_{acc_side}.P')
    acc = load_class_pickle(join(get_project_path('data'),
                                 'merged_sub_data',
                                 featClass.DATA_VERSION,
                                 f'sub-{sub}', fname))

    # calculate rms per feat-window
    acc_axes = ['ACC_' in c for c in acc.colnames]
    svm = signalvectormagn(acc.data[:, acc_axes])
    rms = calc_windowed_rms(acc_sig=svm, win_times=win_times_sec,
                            acc_times=acc.times,
                            win_len=featClass.WIN_LEN_sec)
    
    if MEAN_ACC:
        fname = (f'{sub}_mergedData_{featClass.DATA_VERSION}'
                f'_acc_right.P')
        acc = load_class_pickle(join(get_project_path('data'),
                                     'merged_sub_data',
                                     featClass.DATA_VERSION,
                                     f'sub-{sub}', fname))

        # calculate rms per feat-window
        acc_axes = ['ACC_' in c for c in acc.colnames]
        svm = signalvectormagn(acc.data[:, acc_axes])
        rms2 = calc_windowed_rms(acc_sig=svm, win_times=win_times_sec,
                                 acc_times=acc.times,
                                 win_len=featClass.WIN_LEN_sec)

        # save if defined
        if SAVE_RMS:
            filepath = join(get_project_path('results'),
                            'features',
                            f'SSD_feats_broad_{featClass.FT_VERSION}',
                            featClass.DATA_VERSION,
                            f'windows_{featClass.WIN_LEN_sec}s_'
                            f'{featClass.WIN_OVERLAP_part}overlap',
                            f'windowed_ACC_RMS_{sub}.json')
            
            sub_rms = {'left': rms, 'right': rms2}
            sub_rms = make_object_jsonable(sub_rms)

            with open(filepath, 'w') as f: json.dump(sub_rms, f)

        rms += rms2
    
    if not Z_SCORE: return rms

    elif Z_SCORE:
        rms_z = (rms - np.mean(rms)) / np.std(rms)
        return rms_z


def select_tf_on_movement(feat10sClass, sub,
                          tf_values_arr, tf_times_arr,
                          SELECT_ON_ACC_RMS,
                          RMS_Z_THRESH=-0.5,
                          RETURN_MOVE_SEL_BOOL=False,):
    """
    Function used within plot_descriptive_SSD_PSDs
    to select time-frequency values and times
    based on RMS-ACC movement.
    Currently selects one-second windows based on
    closest 10-second RMS-ACC values. Future 1-sec
    RMS resolution needs long computational time
    to prepare and store generated data.

    Inputs:
        - feat10sClass: priorly imported FeatLidClass
            with features, labels, and acc-rms
        - sub: current sub id
        - tf_values_arr, tf_times_arr (minutes, 1Hz): array with
            1d or 2d-values, for STN processing this
            are the keys of the dicts with match/nonmatch
            keys, generated within plot_STN_PSD_vs_LID()
        - SELECT_ON_ACC_RMS: either INCL_MOVE or EXCL_MOVE
        - RMS_Z_THRESH: z-score for mvoement threshold, defaults
            to -0.5.
    
    Returns:
        - tf_values_arr: corrected
        - tf_times_arr: corrected
    """
    allowed_selections = ['INCL_MOVE', 'EXCL_MOVE']
    assert SELECT_ON_ACC_RMS in allowed_selections, (
        F'SELECT_ON_ACC_RMS NOT IN {allowed_selections}'
    )
    # print(f'extract RMS for sub {sub}, {match_key}')
    # # calculate rms per feat-window (second resolution)
    # epoch_rms = get_any_resolution_acc_rms(
    #     sub=sub, epoch_times_sec=tf_times['match'] * 60,
    #     epoch_length_s=1, data_version=DATA_VERSION
    # )

    # fast way: RMS per 10 seconds
    rms = feat10sClass.ACC_RMS[sub]  # std zscored mean-rms
    rms_move = (rms > RMS_Z_THRESH).astype(int)
    rms_time = feat10sClass.FEATS[sub].index.values * 60  # in minutes, convert to seconds
    move_mask = np.zeros_like(tf_times_arr)

    for rms_t, rms_bool in zip(rms_time, rms_move):
        tf_i1 = np.argmin(abs((tf_times_arr * 60) - rms_t))  # convert to seconds for rounding accuracy
        move_mask[tf_i1:tf_i1 + feat10sClass.WIN_LEN_sec] = rms_bool
    
    print(f'Movement selection ({SELECT_ON_ACC_RMS}) for sub-{sub}:\n'
          f'rms-shape: {rms.shape}, mask-shape: {move_mask.shape}, '
          f'nr of movement-moments: {sum(move_mask)}, tf-arr-shape: {tf_values_arr.shape}')
    
    # select based on found windows
    if SELECT_ON_ACC_RMS == 'INCL_MOVE':
        move_sel = move_mask.astype(bool)
    elif SELECT_ON_ACC_RMS == 'EXCL_MOVE':
        move_sel = ~move_mask.astype(bool)
    
    tf_times_arr = tf_times_arr[move_sel]
    if tf_values_arr.shape[0] > tf_values_arr.shape[1]:
        tf_values_arr = tf_values_arr[move_sel, :]
    else:
        tf_values_arr = tf_values_arr[:, move_sel]

    # return corrected values and times, if required 
    # return bool for adhoc movement selection 
    if RETURN_MOVE_SEL_BOOL: return tf_values_arr, tf_times_arr, move_sel

    else: return tf_values_arr, tf_times_arr


def select_tf_on_movement_10s(feat10sClass, sub,
                              tf_values_arr, tf_times_arr,
                              SELECT_ON_ACC_RMS,
                              RMS_Z_THRESH=-0.5,
                              RETURN_MOVE_SEL_BOOL=False,
                              plot_figure=False,):
    """
    Function used within plot_COHs_spectra
    to select time-frequency values and times
    based on RMS-ACC movement.
    Selects per 10-SECOND WINDOW!

    Inputs:
        - feat10sClass: priorly imported FeatLidClass
            with features, labels, and acc-rms
        - sub: current sub id
        - tf_values_arr, tf_times_arr (minutes, 1Hz): array with
            1d or 2d-values, for STN processing this
            are the keys of the dicts with match/nonmatch
            keys, generated within plot_COH_spectra()
        - SELECT_ON_ACC_RMS: either INCL_MOVE or EXCL_MOVE
        - RMS_Z_THRESH: z-score for mvoement threshold, defaults
            to -0.5.
    
    Returns:
        - tf_values_arr: corrected
        - tf_times_arr: corrected
    """
    allowed_selections = ['INCL_MOVE', 'EXCL_MOVE']
    assert SELECT_ON_ACC_RMS in allowed_selections, (
        f'SELECT_ON_ACC_RMS NOT IN {allowed_selections}'
    )
    rms = feat10sClass.ACC_RMS[sub]  # std zscored mean-rms
    
    # individual threshold adjustment (based on visualisations)
    if sub=='010': RMS_Z_THRESH = -.3
    
    rms_move = (rms > RMS_Z_THRESH).astype(int)
    rms_time = feat10sClass.FEATS[sub].index.values * 60  # in minutes, convert to seconds
    move_mask = np.zeros_like(tf_times_arr)

    for i_t, t in enumerate(tf_times_arr):
        try: i_rms = np.where(rms_time == t)[0][0]
        except: i_rms = np.argmin(abs(rms_time - t))
        move_mask[i_t] = rms_move[i_rms]
    
    if 'INCL' in SELECT_ON_ACC_RMS: move_sel = move_mask.astype(bool)
    elif 'EXCL' in SELECT_ON_ACC_RMS: move_sel = ~move_mask.astype(bool)

    # plot individual selection overview (before data is selected out)
    if plot_figure:
        plot_move_selection(tf_times_arr, move_sel,
                            rms_time, rms, sub,
                            RMS_Z_THRESH=RMS_Z_THRESH,
                            IN_EX_CLUDE=SELECT_ON_ACC_RMS)
    # invert if necessary (check with value and time shapes)    
    if tf_values_arr.shape[0] == len(tf_times_arr):
        tf_values_arr = tf_values_arr[move_sel, :]
    elif tf_values_arr.shape[1] == len(tf_times_arr):
        tf_values_arr = tf_values_arr[:, move_sel]
    tf_times_arr = tf_times_arr[move_sel]

    # return corrected values and times, if required 
    # return bool for adhoc movement selection 
    if RETURN_MOVE_SEL_BOOL: return tf_values_arr, tf_times_arr, move_sel

    else: return tf_values_arr, tf_times_arr


def plot_move_selection(tf_times_arr, move_sel,
                        rms_time, rms, sub,
                        IN_EX_CLUDE, RMS_Z_THRESH):
    
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.hist(rms)
    # plt.savefig(join(get_project_path('figures'),
    #                  'ft_exploration', 'movement',
    #                  'rms_move_selection',
    #                  f'hist_rms{sub}'),
    #             dpi=150, facecolor='w',)
    # plt.close()

    if IN_EX_CLUDE == 'INCL_MOVE':
        lab_true, lab_false = ('Movement', 'No Movement')
    elif IN_EX_CLUDE == 'EXCL_MOVE':
        lab_true, lab_false = ('No Movement', 'Movement')

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # plot ACC
    ax.plot(rms_time / 60, rms, color='gray', alpha=.5,
            label='ACC RMS (mean)',)
    
    # plot movement selection
    y1, y2 = ax.get_ylim()
    ax.fill_between(tf_times_arr / 60, y1=y1, y2=y2,
                    where=move_sel, label=lab_true,
                    color='indigo', alpha=.3,)
    ax.fill_between(tf_times_arr / 60, y1=y1, y2=y2,
                    where=~move_sel, label=lab_false,
                    color='gold', alpha=.3,)
    
    ax.set_xlabel('Time after LDopa (min)', size=14,)
    ax.set_ylabel('z-scored ACC RMS (a.u.)', size=14,)
    ax.legend(frameon=False, fontsize=14,)
    ax.set_title(f'movement selection sub-{sub}\n'
                 f'(z-RMS cutoff: {RMS_Z_THRESH}: '
                 f'n-ft samples: {len(move_sel)} ({sum(move_sel)}%),'
                 f' n-rms samples in plot: {len(rms)})')

    ax.tick_params(size=14, labelsize=14,)

    plt.tight_layout()
    plt.savefig(join(get_project_path('figures'),
                     'ft_exploration', 'movement',
                     'rms_move_selection',
                     f'rms_moveSel_{sub}'),
                dpi=150, facecolor='w',)
    plt.close()

