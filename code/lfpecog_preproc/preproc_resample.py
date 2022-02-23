# Import general packages and functions
from array import array
import numpy as np
from scipy.signal import resample_poly
import mne

def resample(
    data: array, Fs_orig: int, Fs_new: int
):
    """
    Function performs the downsampling of recorded
    (neurophys) data to the desired frequency for
    feature extraction.
    Function is written for downsampling since
    this will always be the case.

    Arguments:
        - data (array): 3d array with data, first dimension
        are windows, second dim are rows, third dim are
        the data points over time within one window
        - Fs_origin (int): original sampling freq
        - Fs_new (int): desired sampling freq

    Returns:
        - newdata (dict): dict containing similar
        data arrays as input data, however downsampled
        and therefore less datapoints per window
    """
    down = int(Fs_orig / Fs_new)  # factor to down sample
    if len(data.shape) == 3:
        newdata = np.zeros((
            data.shape[0],
            data.shape[1],
            int(data.shape[2] / down),
        ))
        time = data[:, 0, :]  # time row over all windows
        newtime = time[:, ::down][:, :newdata.shape[2]]
        newdata[:, 0, :] = newtime  # alocate new times in new data array
        newdata[:, 1:, :] = resample_poly(
            data[:, 1:, :], up=1, down=down, axis=2
        )[:, :, :newdata.shape[2]]  # fill signals rows with signals
    
    if len(data.shape) == 2:
        newdata = np.zeros((
            data.shape[0],
            data.shape[1] // down,
        ))
        time = data[0, :]  # time row over all windows
        newtime = time[::down][:newdata.shape[1]]
        newdata[0, :] = newtime  # alocate new times in new data array
        # newdata[1:, :] = resample_poly(
        #     data[1:, :], up=1, down=down, axis=1
        # )[:, :newdata.shape[1]]  # fill signals rows with signals
        newdata[1:, :] = mne.filter.resample(
            data[1:, :], up=1, down=down, axis=1
        )[:, :newdata.shape[1]]  # fill signals rows with signals

    return newdata


