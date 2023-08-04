"""
Analyze movement voluntary effect
"""
# import functions and packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch

# import own functions
import lfpecog_analysis.load_SSD_features as load_ssd_fts
import lfpecog_features.get_ssd_data as ssd
import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq
import utils.utils_fileManagement as utilsFiles
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle,
                                        save_class_pickle,
                                        mergedData,
                                        correct_acc_class)


def plot_overview_tap_detection(
    acc, fsize=14, SAVE_FIG=False,
):

    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # plot both tap data columns
    ax.plot(acc.times/60, acc.data[:, 5], label=acc.colnames[5])
    ax.plot(acc.times/60, acc.data[:, 6], label=acc.colnames[6])

    ax.plot(acc.times/60, acc.data[:, 4], lw=5, alpha=.6,
            label='task')

    ax.set_ylim(0, 2.5)

    ax.legend(loc='upper right', ncol=3,
            fontsize=fsize,)

    ax.set_title(f'Sub-{acc.sub}: Acc-Tap-Detection',
            loc='left', weight='bold',
            size=fsize,)
    ax.set_xlabel('Time after L-Dopa intake (minutes)',
                  size=fsize,)

    for sp in ['right', 'top']: ax.spines[sp].set_visible(False)

    plt.tick_params(axis='both', size=fsize,
                    labelsize=fsize)
    plt.tight_layout()
    
    if SAVE_FIG:
        fig_name = f'acc_tap_detect_sub{acc.sub}'
        save_path = os.path.join(get_project_path('figures'),
                                'ft_exploration',
                                acc.data_version,
                                'movement', 'tap_detect',
                                fig_name)
        plt.savefig(save_path, facecolor='w', dpi=150,)
        plt.close()

    else:
        plt.show()


def create_sub_movement_psds(sub, data_version='v4.0', ft_version='v4',):

    # define main directory with stored merged data
    main_data_path = os.path.join(get_project_path('data'),
                                  'merged_sub_data', data_version)
    results_path = os.path.join(get_project_path('results'), 'features',
                                  'SSD_feats_broad', data_version,
                                  'windows_10s_0.5overlap_tapRest')
    
    # load all SSD timeseries for subject
    ssd_sub = ssd.get_subject_SSDs(sub=sub,
                                   incl_stn=True, incl_ecog=False,
                                   ft_setting_fname=f'ftExtr_spectral_{ft_version}.json',)
    
    # get ACC-labels
    sub_data_path = os.path.join(main_data_path, f'sub-{sub}')
    fname = (f'{sub}_mergedData_{data_version}'
             '_acc_left.P')  # left and right contain both bilateral labels
    acc = load_class_pickle(os.path.join(sub_data_path, fname))

    # correct for incorrect timestamps in v4 subject 103 (Wrong day in json)
    if sub == '103' and max(acc.times) > 2e6:
        setattr(acc, 'times',
                np.array(acc.times) - np.float64(27 * 24 * 60 * 60))
        print('corrected 103 ACC timings')

    # define lateralities
    for lfp_side, acc_side in zip(['left', 'right',],
                                  ['right', 'left']):
        # exclude combinations without tapping performed
        if (sub, acc_side) in [('017', 'right'),
                            #    ('017', 'left'),  # for left tapping 017 no_move has to be checked
                               ]:
            print(f'skip acc-side {acc_side} for sub-{sub} due to no tapping')

        # get lfp data
        lfp = getattr(ssd_sub, f'lfp_{lfp_side}')
        # create 2d array with bands
        for i_bw, bw in enumerate(['delta', 'alpha', 'lo_beta',
                                   'hi_beta', 'gamma']):
            bw_sig, sig_times, _ = ssd_TimeFreq.get_cont_ssd_arr(
                subSourceSSD=lfp, bw=bw,
                winLen_sec=ssd_sub.settings['WIN_LEN_sec'],
            )
            if i_bw == 0: lfp_arr = np.atleast_2d(bw_sig)
            else: lfp_arr = np.concatenate([lfp_arr, np.atleast_2d(bw_sig)], axis=0)
        # merge into array
        lfp_arr = np.array(lfp_arr)

        # get tapping times
        if ssd_sub.sub == '017':
            peak_idx, peak_t, tap_bool = custom_tap_finding_017(acc)  # 017 only left taps
        else:
            i_col_sidetap = [i for i, c in enumerate(acc.colnames)
                            if c == f'{acc_side}_tap'][0]
            tap_bool = acc.data[:, i_col_sidetap]
            
            # to exclude contralat taps from no-tap (rest)            
            i_col_no_move = [i for i, c in enumerate(acc.colnames)
                            if c == 'no_move'][0]
            move_bool = ~acc.data[:, i_col_no_move]
        
        # exclude free-tasks for tap vs rest analysis
        i_col_task = [i for i, c in enumerate(acc.colnames)
                      if c == 'task'][0]
        lfp_arr, sig_times = excl_specific_task(
            data_arr=lfp_arr,
            data_times=sig_times,
            task_arr=acc.data[:, i_col_task],
            task_times=acc.times,
            task_to_excl='free'
        )

        # split data in tap vs no-tap
        tap_data = select_taps_in_data(
            neural_data=lfp_arr,
            neural_times=sig_times,
            tap_bool=tap_bool,
            tap_bool_times=acc.times,
            return_nontap=False,
        )

        # get no-tap, only rest, no movement data
        lfp_arr, sig_times = excl_specific_task(
            data_arr=lfp_arr,
            data_times=sig_times,
            task_arr=acc.data[:, i_col_task],
            task_times=acc.times,
            task_to_excl='tap'
        )

        _, no_move_data = select_taps_in_data(
            neural_data=lfp_arr,
            neural_times=sig_times,
            tap_bool=tap_bool,
            tap_bool_times=acc.times,
            return_nontap=True,
            move_excl_bool=move_bool,
            move_excl_times=acc.times,
            neural_fs=lfp.fs,
            margin_around_tap_sec=2,
        )

        print(f'Saving sub-{sub} lfp-{lfp_side}: tap-shape:'
              f' {tap_data.shape}, NO TAP: {no_move_data.shape}')
        np.save(file=os.path.join(results_path,
                                  f'{sub}_lfp_{lfp_side}_tapSigs.npy'),
                arr=tap_data, allow_pickle=True)
        np.save(file=os.path.join(results_path,
                                  f'{sub}_lfp_{lfp_side}_restSigs.npy'),
                arr=no_move_data, allow_pickle=True)





def excl_specific_task(data_arr, data_times,
                       task_arr, task_times, task_to_excl):
    # convert to arrays
    if isinstance(data_times, list): data_times = np.array(data_times)
    if isinstance(task_times, list): task_times = np.array(task_times)
    if isinstance(task_arr, list): task_arr = np.array(task_arr)
    
    task_changes = np.diff((task_arr == task_to_excl).astype(int)) 
    idx_task_start = np.where(task_changes== 1)[0]
    idx_task_end = np.where(task_changes == -1)[0]

    t_task_start = task_times[idx_task_start]
    t_task_end = task_times[idx_task_end]

    # set true to keep, false to exclude
    if len(data_arr.shape) == 1: incl_bool = np.ones(len(data_arr))
    elif len(data_arr.shape) == 2: incl_bool = np.ones(data_arr.shape[-1])
    

    for t1, t2 in zip(t_task_start, t_task_end):
        # find indices in neural data for tap start and end
        i1 = np.argmin(abs(data_times - t1))
        i2 = np.argmin(abs(data_times - t2))

        incl_bool[i1:i2] = 0

    # select neural data based on bool
    if len(data_arr.shape) == 1:
        data_arr = data_arr[incl_bool.astype(bool)]
    elif len(data_arr.shape) == 2:
        data_arr = data_arr[:, incl_bool.astype(bool)]
            
    data_times = data_times[incl_bool.astype(bool)]


    return data_arr, data_times


def select_taps_in_data(
    neural_data, neural_times, tap_bool, tap_bool_times,
    margin_around_tap_sec=1, neural_fs=2048, return_nontap=False,
    move_excl_bool=False, move_excl_times=False,
):
    assert neural_data.shape[-1] == len(neural_times), (
        'length neural data and times do not match'
    )
    assert len(tap_bool) == len(tap_bool_times), (
        'length tap_bool and times do not match'
    )
    # convert to arrays
    if isinstance(neural_data, list): neural_data = np.array(neural_data)
    if isinstance(neural_times, list): neural_times = np.array(neural_times)

    # find tap start end indeices
    tap_df = np.diff(tap_bool.astype(int))
    tap_starts = np.where(tap_df == 1)[0]  # gives indices
    tap_ends = np.where(tap_df == -1)[0]
    # convert indices into time (dopa_time seconds)
    tap_starts = tap_bool_times[tap_starts]
    tap_ends = tap_bool_times[tap_ends]

    # create boolean arrays to adjust
    data_tap_bool = np.zeros(neural_data.shape[-1])  # change to positive during taps
    if return_nontap:
        data_notap_bool = np.ones(neural_data.shape[-1])  # change to negative during tap plus pad
        pad_idx = int(margin_around_tap_sec * neural_fs)

    for t1, t2 in zip(tap_starts, tap_ends):
        # find indices in neural data for tap start and end
        i1 = np.argmin(abs(neural_times - t1))
        i2 = np.argmin(abs(neural_times - t2))
        # set selected time window to positive in bool
        data_tap_bool[i1:i2] = 1

        if return_nontap:
            data_notap_bool[i1 - pad_idx:i2 + pad_idx] = 0

    if return_nontap:
        # find tap start end indices for mvoement to be removed
        tap_df = np.diff(move_excl_bool.astype(int))
        tap_starts = np.where(tap_df == 1)[0]  # gives indices
        tap_ends = np.where(tap_df == -1)[0]
        # convert indices into time (dopa_time seconds)
        tap_starts = move_excl_times[tap_starts]
        tap_ends = move_excl_times[tap_ends]

        # set no_tap_bool to negative for contra-taps
        for t1, t2 in zip(tap_starts, tap_ends):
            # find indices in neural data for tap start and end
            i1 = np.argmin(abs(neural_times - t1))
            i2 = np.argmin(abs(neural_times - t2))
            # set selected time window to negative in bool
            data_notap_bool[i1 - pad_idx:i2 + pad_idx] = 0

    # select neural data based on bool
    if len(neural_data.shape) == 1:
        tap_data = neural_data[data_tap_bool.astype(bool)]
    elif len(neural_data.shape) == 2:
        tap_data = neural_data[:, data_tap_bool.astype(bool)]

    if return_nontap:
        if len(neural_data.shape) == 1:
            no_tap_data = neural_data[data_notap_bool.astype(bool)]
        elif len(neural_data.shape) == 2:
            no_tap_data = neural_data[:, data_notap_bool.astype(bool)]

    if not return_nontap: return tap_data
    
    if return_nontap: return tap_data, no_tap_data 




def custom_tap_finding_017(acc_l_017):
    """
    custom needed due to restless legs and
    very slow hand raising

    results in indices and times of taps
    with left hand only
    Tapping movement starts circa two seconds
    before and one second after times

    Inputs: 
        - acc: acc class from pickle acc 017
    """
    fs = int(acc_l_017.fs)

    d_og = acc_l_017.data[:, 1]  # 1 because of ACC_L_X
    t_og = acc_l_017.times
    idx_og = np.arange(len(d_og))
    task = acc_l_017.data[:, 4]  # 4 is task column

    sel = task == 'tap'

    d = d_og[sel]
    t = t_og[sel]
    idx = idx_og[sel]

    THR=.5e-7
    peak_idx, peak_props = find_peaks(d, height=THR, distance=int(fs*5))

    peak_t = t[peak_idx]
    peak_i = idx[peak_idx]

    # create boolean positive during tap
    tap_bool = np.zeros(len(d_og))
    for i in peak_i:
        try:
            if t_og[i+fs] - t_og[i-(2*fs)] > 3:
                tap_bool[i - (2*fs):i] = 1
            else: tap_bool[i - (2*fs):i+fs] = 1
        except IndexError:
            tap_bool[i - (2*fs):i] = 1


    # plt.scatter(t[peak_idx], np.ones(len(peak_idx)) * 1e-7,
    #             color='red', s=10)

    # plt.plot(t_og, d_og, color='green', alpha=.5)

    # plt.xlim(14, 15)
    # plt.show()

    return peak_i, peak_t, tap_bool