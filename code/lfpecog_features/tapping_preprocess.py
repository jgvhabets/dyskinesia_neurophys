'''Accelerometer preprocess functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def run_preproc_acc(
    dat_arr,
    fs: int,
    to_detrend: bool=True,
    to_remove_outlier=True,
    to_check_magnOrder: bool=True,
    to_check_polarity: bool=True
):
    """
    Preprocess accelerometer according to defined steps
    """
    main_ax_index = find_main_axis(dat_arr)
    
    if to_check_magnOrder: dat_arr = check_order_magnitude(
        dat_arr, main_ax_index)

    if to_detrend: dat_arr = detrend_bandpass(dat_arr, fs)

    if to_check_polarity: dat_arr = check_polarity(
        dat_arr, main_ax_index, fs)

    if to_remove_outlier: dat_arr = remove_outlier(
        dat_arr, main_ax_index, fs)

    return dat_arr


def find_main_axis(dat_arr):
    """
    Select acc-axis which recorded tapping the most
    """
    maxs = np.max(dat_arr, axis=1)
    mins = abs(np.min(dat_arr, axis=1))
    main_ax_index = np.argmax(maxs + mins)
    # main_axis = dat_arr[main_ax_index]

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


def remove_outlier(dat_arr, main_ax_index, fs):
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

    print(f'{np.sum(outliers)} outlier-timepoints to remove')
    remove_i = np.zeros_like((main_ax))
    
    for i, outl in enumerate(outliers):
        if outl: remove_i[i - buff:i + buff] = [1] * 2 * buff
    
    dat_arr[:, remove_i.astype(bool)] = np.nan

    return dat_arr


def check_order_magnitude(dat_arr, main_ax_index):
    """
    Checks and corrects if the order of magnitude of
    the acc-signal is between 0 instead of 1e-6.
    """
    # print('before', dat_arr[:, :10])
    if np.percentile(dat_arr[main_ax_index], 99) < 1e-2:
    
        print('small magnitude detected')
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] / 1e-6
    
    elif np.percentile(dat_arr[main_ax_index], 99) > 1e2:
    
        print('large magnitude detected')
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * 1e-6
    
    else:
        print('normal range')
    
    # print('after', dat_arr[:, :10])

    return dat_arr


def check_polarity(dat_arr, main_ax_index: int, fs: int):
    """
    Check whether accelerometer was placed correctly.
    Correct is defined as when upwards movement is
    recorded as positive acceleration.
    """
    print(f'Index chosen: {main_ax_index}')
    main_ax = dat_arr[main_ax_index]
    impacts = find_peaks(
        np.diff(main_ax),
        height=np.percentile(main_ax, 99)
    )[0]

    count = 0    
    for pos in impacts:
        area_pre = main_ax[
            pos - int(fs / 10):pos - int(fs / 50)]
        posRMS = sum([area_pre[area_pre > 0]][0] ** 2)
        negRMS = sum([area_pre[area_pre < 0]][0] ** 2)

        if  posRMS > negRMS:
            count += 1

    if (count / impacts.shape[0]) > .5:
        # print('Pos/Neg switched')
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * -1

    return dat_arr


