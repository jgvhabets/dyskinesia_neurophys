# Import general packages and functions
from array import array
import numpy as np
from scipy.signal import resample_poly
import mne


def resample_for_dict(
    dataDict,
    chNamesDict,
    settings,
    orig_Fs
):
    """
    Run resample() for data dictionaries
    """
    groups_Fs = {}

    for group in dataDict.keys():

        n_timerows = sum([
            'time' in n for n in chNamesDict[group]
        ])

        if group[:3] == 'acc':
            dtype = 'acc'

        elif group[:3] in ['lfp', 'eco']:
            dtype = 'ephys'

        else:
            print(f'/tGROUP {group} NOT RESAMPLED BEC'
            'AUSE NO RESAMPLE FREQ WAS GIVEN (json)')
            continue
    
        dataDict[group] = resample(
            data=dataDict[group],
            n_timeRows=n_timerows,
            Fs_orig=orig_Fs,
            Fs_new=settings[dtype]['resample_Fs'],
        )
        # new freq sample can vary between datatypes
        groups_Fs[group] = settings[dtype]['resample_Fs']

    return dataDict, groups_Fs

def resample(
    data: array, n_timeRows:int,
    Fs_orig: int, Fs_new: int
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
        - n_timeRows (int): number of rows in array
            containing time stamps based on names
        - Fs_origin (int): original sampling freq
        - Fs_new (int): desired sampling freq

    Returns:
        - newdata (dict): dict containing similar
        data arrays as input data, however downsampled
        and therefore less datapoints per window
    """
    down = Fs_orig / Fs_new  # factor to down sample

    if len(data.shape) == 3:

        newdata = np.zeros((
            data.shape[0],
            data.shape[1],
            int(data.shape[2] / down),
        ))

        time = data[:, :n_timeRows, :]
        newtime = time[:, :, ::down][:, :newdata.shape[2]]
        newdata[:, :n_timeRows, :] = newtime

        newdata[:, n_timeRows:, :] = resample_poly(
            data[:, n_timeRows:, :], up=1, down=down, axis=2
        )[:, :, :newdata.shape[2]]


    if len(data.shape) == 2:

        newdata = np.zeros((
            data.shape[0],
            int(data.shape[1] / down),
        ))
        if n_timeRows > 0: newtime = np.zeros((n_timeRows, newdata.shape[-1]))

        if n_timeRows > 0: time = data[:n_timeRows, :]
        if n_timeRows > 0: 
            newtime[0] = np.arange(
                start=0, stop=time[0, -1], step=(1 / Fs_new)
            )[:newdata.shape[-1]]
        if n_timeRows > 0: 
            newtime[1] = np.arange(
                start=time[1, 0], stop=time[1, -1], step=(1 / Fs_new)
            )[:newdata.shape[-1]]

        if n_timeRows > 0: newdata[:n_timeRows, :] = newtime
        
        newdata[n_timeRows:, :] = mne.filter.resample(
            data[n_timeRows:, :], up=1, down=down, axis=1
        )[:, :newdata.shape[1]]

    return newdata


