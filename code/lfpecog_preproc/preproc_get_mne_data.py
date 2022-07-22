"""
ECoG-LFP Preprocessing Helper-Functions to Load
and Preprocess.
"""
# Import general packages and functions
from array import array
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from typing import Any
import mne_bids

def get_data_and_channels(
    rawRun, runInfo,
):
    """
    Loading and importing (getting) of
    data via MNE-functions
    """
    data_objs, data_arrays = {}, {}
    ch_names = {}

    for g in runInfo.data_groups:

        data_objs[g] = getattr(rawRun, g).load_data()  # MNE's load_data()
        (ch_arr, ch_t) = data_objs[g].get_data(return_times=True)  # MNE's actual getting of data
        ch_t = np.reshape(ch_t, (1, len(ch_t)))
        data_arrays[g] = np.vstack((ch_t, ch_arr))
    
        ch_names[g] = data_objs[g].info['ch_names']

        if g[:3] == 'acc': ch_names[g] = [n[:7] for n in ch_names[g]]
        
        ch_names[g] = ['run_time', 'dopa_time'] + ch_names[g]

        print(f'\nCHANNEL NAMES: GROUP {g}: {ch_names[g]}'
                f'\nDATASHAPE: {data_arrays[g].shape}')
        
    return data_arrays, ch_names


def remove_flatlines_empties(
    data: dict, chNames: dict, fs,
    thresh: float=.66, reportPath='',
):
    """
    Removes single channels with more than
    *threshold* % flatlines, and removes full
    groups which are empty thereafter.
    Does not remove the empty groups from the
    data_groups attributes in runInfo-Class,
    so data.keys() should be used for updated
    group list.

    Inputs:
        - data: dict containing loaded data per group
        - chNames: dict with corr channel names
        - fs: sample frequency
        - thresh: threshold for flatline part
    
    Returns:
        - data: cleaned data dict
        - chNames: claned channelNames dict
    """
    if reportPath: report = ''

    for g in data:
        flat_chs = []
        
        for ch in np.arange(1, data[g].shape[0]):

            sec_starts = np.arange(0, len(data[g][ch]), fs)
            seconds_array = np.array(
                [data[g][ch][int(s + 1)] - data[g][ch][int(s)] for s in sec_starts]
            )
            count_flats = sum(seconds_array == 0)  # count zeros

            if count_flats / len(seconds_array) > thresh:
                flat_chs.append(ch)

        if len(flat_chs) > 0:

            del_names = []
            for f_c in flat_chs: del_names.append(chNames[g][f_c])
            for c in del_names: chNames[g].remove(c)
            np.delete(data[g], flat_chs, axis=0)
            # delete rows (on rownumber) in once, to prevent changing row nrs during deletion!
            
            report = report + f'\nRemoved from {g}, FLATLINE channels: {del_names}'

            ### TODO: INCLUDE VISUALISATION OF REMOVED CHANNELS !!!
    
    print(f'\n\tREPORT UPDATE: {report}')

    if reportPath:

        with open(reportPath, 'a') as f:

            f.write(report)
            f.close()

    data, chNames = delete_empty_groups(
        data, chNames, reportPath
    )

    return data, chNames


def delete_empty_groups(
    dataDict: dict, chNameDict: dict, reportPath='',
):
    """
    """
    empty_groups = []

    for group in dataDict.keys():

        if dataDict[group].shape[-2] <= 1:
            empty_groups.append(group)
    
    for group in empty_groups:

        del(dataDict[group], chNameDict[group])

    report = f'Empty Group(s) removed: {empty_groups}'
    if reportPath:

        with open(reportPath, 'a') as f:

            f.write(report)
            f.close()


    return dataDict, chNameDict