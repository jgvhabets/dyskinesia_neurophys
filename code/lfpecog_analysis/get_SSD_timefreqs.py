"""
Convert SSD-timeseries into combined
timefrequency arrays and Power Spectra
"""

import numpy as np
from pandas import DataFrame
from collections import namedtuple
from scipy.signal import welch
import json
from os.path import join, exists
from os import makedirs, listdir

from utils.utils_fileManagement import (
    get_project_path, make_object_jsonable
)
from lfpecog_features.get_ssd_data import get_subject_SSDs


def get_all_ssd_timeFreqs(
    SUBS, DATA_VERSION='v4.0', FT_VERSION='v4',
    WIN_LEN=10, WIN_OVERLAP=0.5, SSD_BROAD=True,
):
    TFs = {}

    for sub in SUBS:
        psd_sub = get_SSD_timeFreq(
            sub=sub, DATA_VERSION=DATA_VERSION, FT_VERSION=FT_VERSION,
            WIN_LEN=WIN_LEN, WIN_OVERLAP=WIN_OVERLAP,
            SSD_BROAD=SSD_BROAD,    
        )
        
        tf_sub = {}
        
        for src in psd_sub.keys():
            tf_sub[src] = TimeFreqTuple(
                np.array(psd_sub[src]['values']),
                np.array(psd_sub[src]['freqs']),
                np.array(psd_sub[src]['times'])
            )

        TFs[sub] = tf_sub
        print(f'...loaded subject-{sub} Time-Frequency data')
        
    return TFs


TimeFreqTuple = namedtuple('TimeFreqTuple',
                           ['values', 'freqs', 'times'])


def get_SSD_timeFreq(
    sub, DATA_VERSION='v4.0', FT_VERSION='v4',
    WIN_LEN=10, WIN_OVERLAP=0.5, SSD_BROAD=True,    
):
    dict_out = {}

    ssd_folder = 'SSD_feats'
    if SSD_BROAD: ssd_folder += '_broad'
    path = join(get_project_path('results'),
                            'features', ssd_folder,
                            DATA_VERSION,
                            f'windows_{WIN_LEN}s_{WIN_OVERLAP}overlap_PSD',
                            )
    filename = f'SSD_PSD_{sub}.json'
    
    if not exists(path):
        makedirs(path)
        create_PSDs = True
    elif np.logical_and(exists(path),
                        filename not in listdir(path)):
        create_PSDs = True
    else:
        create_PSDs = False

    if create_PSDs: 
        print(f'START CREATING SSD PSDs for sub-{sub} in get_SSD_timeFreq()')
        ssd_subClass = get_subject_SSDs(
            sub=sub,
            incl_stn=True,
            incl_ecog=True,
            ft_setting_fname=f'ftExtr_spectral_{FT_VERSION}.json',
        )
        
        for source in ssd_subClass.ephys_sources:
            ssd_source_dat = getattr(ssd_subClass, source)
            timefreqArr, tf_times, min_f, max_f = create_SSD_timeFreqArray(ssd_source_dat)
            tf_values = timefreqArr.values
            tf_freqs = np.array(timefreqArr.index)

            source_dict = {'values': tf_values,
                           'times': tf_times,
                           'freqs': tf_freqs}
            source_dict = make_object_jsonable(source_dict)
            dict_out[source] = source_dict
        # save dict as json
        with open(join(path, filename), 'w') as f:
            json.dump(dict_out, f)
        
        return dict_out    
    
    elif not create_PSDs:
        print(f'load existing powers for sub-{sub} ({filename})')

        # load dict from json
        with open(join(path, filename), 'r') as f:
            dict_out = json.load(f)
        
        return dict_out



def get_cont_ssd_arr(subSourceSSD, bw,
                     winLen_sec=10,):
    """
    
    Returns:
        - new_arr: 1d array with pasted full data
        - new_timestamps: 1d array with timestamps parallel to data
        - new_times_sec: only full seconds of new data
    """
    
    windows = getattr(subSourceSSD, bw)
    wintimes = subSourceSSD.times
    new_fs = subSourceSSD.fs
    win_t_diffs =  list(np.diff(subSourceSSD.times)) + [0]

    new_arr, new_times_sec, new_timestamps = [], [], []
    skip_next = False

    for windat, wintime, t_diff in zip(windows, wintimes, win_t_diffs):
        
        if not skip_next:
            if len(new_times_sec) > 0:
                assert wintime > new_times_sec[-1], (
                    f'wintime {wintime} earlier than last time added {new_times_sec[-1]}'
                )
            # add if no nans in window
            if not np.isnan(windat).any():
                new_arr.extend(windat)
                new_times_sec.extend(np.arange(wintime, wintime + winLen_sec))
                new_timestamps.extend(np.arange(wintime, wintime + winLen_sec, (1 / new_fs)))
            # if nans in window
            else:
                skip_next = False  # no data added, try to add next window
                continue

            if t_diff <= (winLen_sec / 2): skip_next = True
            else: skip_next = False

        elif skip_next:
            skip_next = False

    new_arr = np.array(new_arr)
    new_times_sec = np.array(new_times_sec)

    assert sum(np.diff(new_times_sec) < 0) == 0, (
        'negative time differences present'
    )

    return new_arr, new_timestamps, new_times_sec


def create_SSD_timeFreqArray(subSourceSSD, win_len_sec=1,):
    """
    Input:
        - subSourceSSD: should be e.g. ssdXXX.lfp_left
    """
    fs = subSourceSSD.fs
    bands = subSourceSSD.settings['SPECTRAL_BANDS']
    min_f = min([min(r) for r in bands.values()])
    max_f = max([max(r) for r in bands.values()])

    for i, bw in enumerate(bands.keys()):

        # convert windows with overlap into continuous array
        cont_arr, cont_time_arr, cont_time_secs = get_cont_ssd_arr(
            subSourceSSD=subSourceSSD, bw=bw
        )
        # reshape into windows of seconds for spectral decomp
        reshaped_arr = cont_arr.reshape((len(cont_arr) // fs, fs))
        # calculate welch with 1 Hz bins per window
        f, px = welch(reshaped_arr, fs=fs,
                             nperseg=int(fs * win_len_sec),
                             axis=1,)
        # add current bandwidth to sum-array
        if i == 0:
            sum_timefreq = DataFrame(
                data=np.zeros((max_f + 1 - min_f, px.shape[0])),
                columns=cont_time_secs,
                index=np.arange(min_f, max_f + 1)
            )
        # define freq-ranges for band
        bw_range = bands[bw]
        if bw == 'hi_beta': bw_range[1] = 50
        if bw == 'gamma': bw_range[0] = 50
        # select and add PSD values for ranges
        sel = px.T[bw_range[0]:bw_range[1] + 1, :]
        sum_timefreq.loc[bw_range[0]:bw_range[1]] = sel

    return sum_timefreq, cont_time_secs, min_f, max_f


def correct_timeFreq_baseline(tf_values, tf_times,
                              perc_change=True):
    """
    Input:
        - tf_values: coming in 1 Hz
        - tf_times: parallel to 1 Hz
    """
    minute_bl_timing = 0

    baseline_sel = (tf_times / 60) < minute_bl_timing

    # accept baseline timing if 120 seconds data are present
    while sum(baseline_sel) < 120:
        minute_bl_timing += 3  # add 3 minutes and re-calculate
        baseline_sel = (tf_times / 60) < minute_bl_timing
        print(f'test {minute_bl_timing} cut off, sum: {sum(baseline_sel)}')



    assert sum(baseline_sel) > 0, ('No baseline found')

    bl_values = np.mean(tf_values[:, baseline_sel],
                        axis=1)
    
    for col in np.arange(tf_values.shape[1]):
        tf_values[:, col] = (tf_values[:, col] - bl_values)
        if perc_change:
            tf_values[:, col] = (tf_values[:, col] / bl_values) * 100

    print(f'baseline timing: {minute_bl_timing}')

    return tf_values