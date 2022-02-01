# Import general packages and functions
import numpy as np
from scipy.signal import resample_poly

def resample(
    data: dict, group: str, Fs_orig: int, Fs_new: int
):
    """
    Function performs the downsampling of recorded
    (neurophys) data to the desired frequency for
    feature extraction.
    Function is written for downsampling since
    this will always be the case.

    Arguments:
        - data (dict): dict containing the 3d arrays
        with data of the different source-groups
        (e.g. lfp-l, lfp-r, ecog)
        - group (str): name of group, corresponding
        to keys in the data dictionary
        - Fs_origin (int): original sampling freq
        - Fs_new (int): desired sampling freq

    Returns:
        - newdata (dict): dict containing similar
        data arrays as input data, however downsampled
        and therefore less datapoints per window
    """
    data = data[group]
    down = int(Fs_orig / Fs_new)  # factor to down sample
    newdata = np.zeros((data.shape[0], data.shape[1],
                        int(data.shape[2] / down)))
    time = data[:, 0, :]  # all time rows from all windows
    newtime = time[:, ::down]  # all windows, only times on down-factor
    newdata[:, 0, :] = newtime  # alocate new times in new data array
    newdata[:, 1:, :] = resample_poly(
        data[:, 1:, :], up=1, down=down, axis=2
    )  # fill signals rows with signals

    return newdata


