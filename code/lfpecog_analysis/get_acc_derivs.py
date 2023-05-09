"""
Get accelerometer related derivatives
"""

# import 
from os import listdir
from os.path import join
from pandas import DataFrame
import numpy as np
from collections import namedtuple

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

def load_acc(
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


from scipy.signal import resample_poly