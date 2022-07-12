'''Accelerometer preprocess functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

# Import own functions
from lfpecog_features.tapping_impact_finder import find_impacts

def run_preproc_acc(
    dat_arr,
    fs: int,
    to_detrend: bool=True,
    to_remove_outlier=True,
    to_check_magnOrder: bool=True,
    to_check_polarity: bool=True,
    verbose: bool=True
):
    """
    Preprocess accelerometer according to defined steps

    Input:
        - 
    """
    # if len(dat_arr.shape) == 1:
    #     temp = np.zeros((1, dat_arr.shape[0]))
    #     temp[0, :] = dat_arr
    #     dat_arr = temp

    # print('start preprocess')
    main_ax_index = find_main_axis(dat_arr)

    if to_check_magnOrder: dat_arr = check_order_magnitude(
        dat_arr, main_ax_index)

    if to_detrend: dat_arr = detrend_bandpass(dat_arr, fs)

    if to_check_polarity: dat_arr = check_polarity(
        dat_arr, main_ax_index, fs, verbose)

    if to_remove_outlier: dat_arr = remove_outlier(
        dat_arr, main_ax_index, fs, verbose)
    
    # print('end preprocess')

    return dat_arr, main_ax_index


def find_main_axis(dat_arr):
    """
    Select acc-axis which recorded tapping the most

    Input:
        - dat_arr (arr): triaxial acc signals
    Returns:
        - main_ax_index (int): [0, 1, or 2], axis
            with most tapping activity detected
    """
    if len(dat_arr.shape) == 2:
        maxs = np.max(dat_arr, axis=1)
        mins = abs(np.min(dat_arr, axis=1))

    elif len(dat_arr.shape) == 1:
        maxs = np.max(dat_arr, )
        mins = abs(np.min(dat_arr,))
    else:
        return print('Check shape data array inserted')

    main_ax_index = np.argmax(maxs + mins)

    return main_ax_index


def detrend_bandpass(
    dat_array, fs: int, lowcut: int=1, highcut: int=100, order=5
):
    """
    Apply bandpass filter to detrend drift in acc-data, effect
    is based on highpass effect.
    """
    nyq = fs / 2
    b, a = butter(
        order,
        [lowcut / nyq, highcut / nyq],
        btype='bandpass'
    )
    filt_dat = filtfilt(b,a, dat_array)

    return filt_dat


def remove_outlier(dat_arr, main_ax_index, fs, verbose):
    """
    Removes large outliers, empirical threshold testing
    resulted in using a percentile multiplication.
    Replaces outliers and the half second around them
    with np.nan's.
    """
    main_ax = dat_arr[main_ax_index]
    buff = int(fs / 4)
    thresh = 10 * np.percentile(main_ax, 99)

    outliers = np.logical_or(
        main_ax < -thresh, main_ax > thresh)
    if np.sum(outliers) == 0: return dat_arr

    if verbose: print(
        f'{np.sum(outliers)} outlier-timepoints to remove'
    )
    remove_i = np.zeros_like((main_ax))
    
    for i, outl in enumerate(outliers):
        if outl: remove_i[i - buff:i + buff] = [1] * 2 * buff
    
    dat_arr[:, remove_i.astype(bool)] = np.nan

    return dat_arr


def check_order_magnitude(dat_arr, main_ax_index):
    """
    Checks and corrects if the order of magnitude of
    the acc-signal is in range [0 - 10] m/s/s, equal
    to range in g (9.81 m/s/s).
    Occasionaly sensor recordings are in order 1e-6,
    or 1e6.
    """
    if np.percentile(dat_arr[main_ax_index], 99) < 1e-2:
    
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] / 1e-6
    
    elif np.percentile(dat_arr[main_ax_index], 99) > 1e2:
    
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * 1e-6

    return dat_arr



import matplotlib.pyplot as plt

def check_polarity(
    dat_arr, main_ax_index: int, fs: int, verbose):
    """
    Check whether accelerometer was placed correctly.
    Correct is defined as when upwards movement is
    recorded as positive acceleration.
    """
    
    main_ax = dat_arr[main_ax_index]

    # plt.figure(figsize=(24,12))
    # plt.plot(dat_arr.T, alpha=.5)
    # plt.plot(np.diff(main_ax))
    # plt.axhline(np.percentile(main_ax, 99))
    # plt.show()
    _, impacts = find_impacts(main_ax, fs)
    # impacts = find_peaks(
    #     np.diff(main_ax),
    #     height=np.percentile(main_ax, 99)
    # )[0]
    assert len(impacts) > 0, 'No impacts-peaks detected'

    count = 0    
    for pos in impacts:
        area_pre = main_ax[
            pos - int(fs / 10):pos - int(fs / 50)]
        posRMS = sum([area_pre[area_pre > 0]][0] ** 2)
        negRMS = sum([area_pre[area_pre < 0]][0] ** 2)

        if  posRMS > negRMS:
            count += 1

    if (count / impacts.shape[0]) > .5:
        if verbose: print('Pos/Neg switched')
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * -1

    return dat_arr


