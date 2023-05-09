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

from utils.utils_fileManagement import (
    load_class_pickle, get_project_path
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