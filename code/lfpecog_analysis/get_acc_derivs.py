"""
Get accelerometer related derivatives
"""

# import 
import numpy as np

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