"""
Convert SSD-timeseries into combined
timefrequency arrays and Power Spectra
"""
import numpy as np
from pandas import DataFrame, read_csv
from collections import namedtuple
from scipy.signal import welch
import json
from os.path import join, exists
from os import makedirs, listdir

from utils.utils_fileManagement import (
    get_project_path, make_object_jsonable
)
from lfpecog_features.get_ssd_data import get_subject_SSDs
from lfpecog_preproc.preproc_import_scores_annotations import get_ecog_side
from lfpecog_analysis.psd_analysis_classes import select_coh_values, get_ssd_coh_from_array
from lfpecog_features.feats_spectral_features import calc_coherence
# from lfpecog_analysis.process_connectivity import (
#     get_conn_values_sub_side
# )   # necessary for GET_CONNECTIVITY = True in get_all_sdd_timeFreqs()


def get_all_ssd_timeFreqs(
    SUBS, DATA_VERSION='v4.0', FT_VERSION='v4',
    WIN_LEN=10, WIN_OVERLAP=0.5, SSD_BROAD=True,
    FORCE_PSD_CREATION=False,
    GET_CONNECTIVITY=False,
    verbose: bool = True,
):
    TFs = {}

    for sub in SUBS:
        TFs[sub] = {}

        if not GET_CONNECTIVITY:
            # get all timefreq SSD data per sub
            psd_sub = get_SSD_timeFreq(
                sub=sub, DATA_VERSION=DATA_VERSION,
                FT_VERSION=FT_VERSION,
                WIN_LEN=WIN_LEN, WIN_OVERLAP=WIN_OVERLAP,
                SSD_BROAD=SSD_BROAD,
                FORCE_PSD_CREATION=FORCE_PSD_CREATION,  
            )
            # create data tuple per source            
            for src in psd_sub.keys():
                TFs[sub][src] = TimeFreqTuple(
                    np.array(psd_sub[src]['values']),
                    np.array(psd_sub[src]['freqs']),
                    np.array(psd_sub[src]['times'])
                )

            if verbose: print(f'...loaded subject-{sub} Time-Frequency data')
        
        
        elif isinstance(GET_CONNECTIVITY, str):
            if sub.startswith('1'): continue

            assert GET_CONNECTIVITY in ['mic', 'trgc'], (
                f'GET_CONNECTIVITY ({GET_CONNECTIVITY}) '
                f'should be "mic" OR "trgc"'
            )
            
            CONN_FT_PATH = join(
                get_project_path('results'),
                'features',
                'connectivity',
                f'windows_{WIN_LEN}s_{WIN_OVERLAP}overlap'
            )
            for src in ['ipsilateral', 'contralateral']:
                # get connectivity values, times, freqs
                (values,
                 times,
                 freqs) = get_conn_values_sub_side(sub=sub,
                                                   stn_side=src,
                                                   CONN_FT_PATH=CONN_FT_PATH,
                                                   conn_method=GET_CONNECTIVITY)
                TFs[sub][src] = TimeFreqTuple(np.array(values),
                                              np.array(freqs),
                                              np.array(times))
                
            if verbose: print(f'...loaded subject-{sub} Connectivity Time-Frequency data')
 
    return TFs


TimeFreqTuple = namedtuple('TimeFreqTuple',
                           ['values', 'freqs', 'times'])


def get_SSD_timeFreq(
    sub, DATA_VERSION='v4.0', FT_VERSION='v4',
    WIN_LEN=10, WIN_OVERLAP=0.5, SSD_BROAD=True,
    FORCE_PSD_CREATION=False,   
):
    dict_out = {}

    ssd_folder = 'SSD_feats'
    if SSD_BROAD: ssd_folder += '_broad'
    ssd_folder += f'_{FT_VERSION}'
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
    
    if FORCE_PSD_CREATION: create_PSDs = True

    if create_PSDs: 
        print(f'START CREATING SSD PSDs for sub-{sub} (fts: {FT_VERSION}) in get_SSD_timeFreq()')
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
        print(f'load existing powers for sub-{sub} (fts: {FT_VERSION}, {filename})')

        # load dict from json
        with open(join(path, filename), 'r') as f:
            dict_out = json.load(f)
        
        return dict_out


def get_COH_timeFreq(
    sub, DATA_VERSION='v4.0', FT_VERSION='v4',
    WIN_LEN=10, WIN_OVERLAP=0.5, SSD_BROAD=True,
    FORCE_PSD_CREATION=False,
    COH_TYPE: str = 'sqCOH',
):
    dict_out = {}

    ssd_folder = 'SSD_feats'
    if SSD_BROAD: ssd_folder += '_broad'
    ssd_folder += f'_{FT_VERSION}'
    path = join(get_project_path('results'),
                            'features', ssd_folder,
                            DATA_VERSION,
                            f'windows_{WIN_LEN}s_{WIN_OVERLAP}overlap_COH',)
    filename = f'SSD_COH_{sub}.json'
    
    if not exists(path):
        makedirs(path)
        create_PSDs = True
    elif np.logical_and(exists(path),
                        filename.replace('COH', COH_TYPE) not in listdir(path)):
        create_PSDs = True
    else:
        create_PSDs = False
    
    if FORCE_PSD_CREATION: create_PSDs = True

    if create_PSDs:
        print(f'START CREATING SSD COHs ({COH_TYPE}) for sub-{sub} (fts: {FT_VERSION}) in get_COH_timeFreq()')
        ssd_subClass = get_subject_SSDs(
            sub=sub,
            incl_stn=True,
            incl_ecog=True,
            ft_setting_fname=f'ftExtr_spectral_{FT_VERSION}.json',
        )
        COH_sources = {'STNs': ['lfp_left', 'lfp_right']}
        if sub.startswith('0'):
             ecog_side = get_ecog_side(sub)
             COH_sources['STNECOG'] = [f'lfp_{ecog_side}', f'ecog_{ecog_side}']
        
        for coh_source, source_list in COH_sources.items():
            ssd_sig1 = getattr(ssd_subClass, source_list[0])
            ssd_sig2 = getattr(ssd_subClass, source_list[1])

            sqcohArr, icohArr, tf_times, min_f, max_f = create_COH_timeFreqArray(ssd_sig1, ssd_sig2)
            
            for tfArr in [sqcohArr, icohArr]:
                tf_values = tfArr.values
                tf_freqs = np.array(tfArr.index)

                source_dict = {'values': tf_values,
                               'times': tf_times,
                               'freqs': tf_freqs}
                source_dict = make_object_jsonable(source_dict)
                dict_out[coh_source] = source_dict
            # save dict as json
            with open(join(path, filename.replace('COH', coh_source)), 'w') as f:
                json.dump(dict_out, f)
        
        return dict_out    
    
    elif not create_PSDs:

        print(f'load existing powers for sub-{sub} (fts: {FT_VERSION}, {filename})')

        # load dict from json
        with open(join(path, filename.replace('COH', COH_TYPE)), 'r') as f:
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



def create_COH_timeFreqArray(
    ssd_sig1, ssd_sig2,
    COH_single_epochs_sec=1, COH_windows_sec=10,):
    """
    Input:
        - subSourceSSD: should be e.g. ssdXXX.lfp_left
    """
    fs = ssd_sig1.fs
    bands = ssd_sig1.settings['SPECTRAL_BANDS']
    min_f = min([min(r) for r in bands.values()])
    max_f = max([max(r) for r in bands.values()])

    for i, bw in enumerate(bands.keys()):
        print(f'create_COH_timefreq: {bw}')
        arr1, arr2, time_wins = [], [], []
        # convert windows with overlap into continuous array
        cont_arr1, cont_time_arr1, cont_time_secs1 = get_cont_ssd_arr(
            subSourceSSD=ssd_sig1, bw=bw, winLen_sec=10,
        )
        cont_arr2, cont_time_arr2, cont_time_secs2 = get_cont_ssd_arr(
            subSourceSSD=ssd_sig2, bw=bw, winLen_sec=10,
        )
        sig1, sig2 = select_coh_values(
            cont_arr1, cont_arr2, cont_time_arr1, cont_time_arr2
        )
        cont_time = cont_time_arr1[np.isin(cont_time_arr1, cont_time_arr2)]
        cont_time_secs = cont_time_secs1[np.isin(cont_time_secs1, cont_time_secs2)]

        n_samples_win = COH_windows_sec * fs
        t0 = round(cont_time[0])
        while t0 < (cont_time[-1] - COH_windows_sec):
            win_sel = np.logical_and(cont_time > t0, cont_time < (t0 + COH_windows_sec))
            if sum(win_sel) < (.5 * n_samples_win): continue
            win1 = [np.nan] * n_samples_win
            win2 = win1.copy()
            win1[:sum(win_sel)] = sig1[win_sel]
            win2[:sum(win_sel)] = sig2[win_sel]
            arr1.append(win1)
            arr2.append(win2)
            time_wins.append(t0)
            t0 += COH_windows_sec
        
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        sqcoh_all, icoh_all = [], []

        for sig1, sig2 in zip(arr1, arr2):
            sig1 = sig1[~np.isnan(sig1)]
            sig2 = sig2[~np.isnan(sig2)]
            # calc coherence
            freqs, _, icoh, _, sqcoh = calc_coherence(
                sig1=sig1, sig2=sig2, fs=fs,
                nperseg=fs*COH_single_epochs_sec,
            )
            icoh = np.atleast_2d(icoh)
            sqcoh = np.atleast_2d(sqcoh)
            if icoh.shape[0] > icoh.shape[1]: icoh = icoh.T
            if sqcoh.shape[0] > sqcoh.shape[1]: sqcoh = sqcoh.T

            sqcoh_all.append(sqcoh.ravel())
            icoh_all.append(icoh.ravel())
        
        sqcoh_all = np.array(sqcoh_all)
        icoh_all = np.array(icoh_all)
        
        # add current bandwidth to sum-array
        if i == 0:
            sqcoh_timefreq = DataFrame(
                data=np.zeros((max_f + 1 - min_f, sqcoh_all.shape[0])),
                columns=time_wins,
                index=np.arange(min_f, max_f + 1)
            )
            icoh_timefreq = sqcoh_timefreq.copy()
        # define freq-ranges for band
        bw_range = bands[bw]
        if bw == 'hi_beta': bw_range[1] = 50
        if bw in ['gamma', 'gamma1']: bw_range[0] = 50
        # select and add PSD values for ranges
        sel = sqcoh_all.T[bw_range[0]:bw_range[1] + 1, :]
        sqcoh_timefreq.loc[bw_range[0]:bw_range[1]] = sel
        # add icoh
        sel = icoh_all.T[bw_range[0]:bw_range[1] + 1, :]
        icoh_timefreq.loc[bw_range[0]:bw_range[1]] = sel

    return sqcoh_timefreq, icoh_timefreq, time_wins, min_f, max_f


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
        print(f'create_SSD_timefreq: {bw}')
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
        if bw in ['gamma', 'gamma1']: bw_range[0] = 50
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


def get_coh_tf_per_sub(sub, COH_type, COH_source,
                       FT_VERSION, DATA_VERSION,
                       WIN_LEN_s=10, WIN_OVERLAP=0.5):
    """
    Input:
        - COH_type: imag_coh or sq_coh
        - COH_source: STN_STN or STN_ECOG 

    Returns:
        - coh_values,
        - coh_times,
        - coh_freqs
    """
    assert COH_type in ['sq_coh', 'imag_coh'], 'wrong COH_type'
    assert COH_source in ['STN_STN', 'STN_ECOG'], 'wrong COH_source'
    
    ft_path = join(get_project_path('results'),
                        'features',
                        f'SSD_feats_broad_{FT_VERSION}',
                        DATA_VERSION,
                        f'windows_{WIN_LEN_s}s_{WIN_OVERLAP}overlap')

    coh_sub_files = [f for f in listdir(ft_path) if
    sub in f and COH_type in f and COH_source in f]

    freqs = np.arange(4, 91)
    coh_sub_df = DataFrame(columns=freqs)

    for f in coh_sub_files:
        temp_cohs = read_csv(join(ft_path, f),
                                header=0, index_col=0)
        for c in temp_cohs.keys():
            coh_sub_df[int(float(c))] = temp_cohs[c]

    coh_times = coh_sub_df.index.values
    coh_freqs = np.array(coh_sub_df.keys())
    coh_values = coh_sub_df.values

    return coh_values, coh_times, coh_freqs
