"""
Main function to call burst-extraction
in neurophysiological signals.

Takes all arguments and calls executive
function in bursts_funcs.py
"""

# import public packages and functions
import numpy as np
import pandas as pd
from itertools import compress
import json
from os.path import join, exists
from os import makedirs
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# import own functions
from lfpecog_features.bursts_funcs import get_burst_features
from utils.utils_fileManagement import (
    get_beta_project_path, make_object_jsonable
)
from lfpecog_features.get_ssd_data import get_subject_SSDs
import lfpecog_features.bursts_funcs as burstFuncs
import lfpecog_features.feats_helper_funcs as ftHelpers
from lfpecog_analysis.get_acc_task_derivs import define_OFF_ON_times
from lfpecog_analysis.load_SSD_features import ssdFeatures
from lfpecog_preproc.preproc_import_scores_annotations import get_ecog_side


def get_bursts_main(
    merged_df,
    col_names,
    fs,
    winLen_minutes = 3,
    winOverlap_part = 1,
    segLen_sec = .5,
    minWin_available = .5,  # skip window if less than this part of potential window is available
    burstCutoffPerc = 75,
    min_shortBurst_sec = 0.1,
    min_longBurst_sec = 0.2,  # consider more detailed distribution
):
    """
    
    """
    # set burst-calculation variables
    bp_dict = {
        'alpha': [8, 13],
        'beta': [13, 30],
        'lowBeta': [13, 20],
        'highBeta': [20, 30],
        'midGamma': [60, 90]
    }

    # determine resulting variables
    winLen_sec = winLen_minutes * 60
    winLen_samples = winLen_sec * fs
    segLen_samples = segLen_sec * fs

    burstFtArray, winTimes, winTasks = {}, {}, {}

    for col in col_names:

        burstFtArray[col] = {}

        print(f'START {col}')

        for freq_name in bp_dict:

            print(f'start {freq_name}')

            # define range of minutes to search for windows
            startWin = round(merged_df.index[0] / winLen_sec) - 1
            stopWin = round(merged_df.index[-1] / winLen_sec) + 1
            # plus / minus one to prevent missing window by rounding

            winStartTimes = []
            burstFtList = []

            for nWin in np.arange(startWin, stopWin + 1, winOverlap_part):
                
                iStart = nWin * winLen_sec
                iStop = (nWin + 1) * winLen_sec

                winDat = merged_df.loc[
                    iStart:iStop, col
                ].values

                if winDat.shape[0] < (minWin_available * winLen_samples):
                    # not enough data present
                    # print(f'\twindow {nWin} excluded\n')
                    continue
                
                # if enough data present in window
                winStartTimes.append(int(iStart))
                
                nShortB, nLongB, rateShort, rateLong = get_burst_features(
                    sig=winDat, fs=fs,
                    burst_freqs=bp_dict[freq_name],
                    cutoff_perc=burstCutoffPerc,
                    threshold_meth='full-window',
                    min_shortBurst_sec=min_shortBurst_sec,
                    min_longBurst_sec=min_longBurst_sec,
                    envelop_smoothing=True,
                )
                burstFtList.append(
                    [nShortB, nLongB, rateShort, rateLong]
                )
                ftNames = ['n_short', 'n_long', 'rate_short', 'rate_long']

            burstFtArray[col][freq_name] = np.array(burstFtList)
            winTimes[col] = winStartTimes

            winTasks[col] = []

            for t in winStartTimes:
                
                winTasks[col].append(merged_df.loc[t:t + fs]['task'].values[0])


    return burstFtArray, winTimes, winTasks, ftNames


from lfpecog_plotting.plotHelpers import get_colors
from lfpecog_plotting.beta_burstrate_plotting import plot_beta_bursts

def get_burst_features_SSD(
    sub: str,
    SMOOTH_MILLISEC: int = 250,
    THRESH_ORIGIN: str = 'off',
    SPLIT_OFF_ON: bool = False,
    TO_SAVE_FIG: bool = True,
    FIG_DIR: str = None,
    sub_SSD_class = None,
    TO_PLOT_FIG: bool = False,
    TO_STORE_BURST_LENGTHS: bool = True,
    LOAD_STORED_RESULTS: bool = False
):
    """
    Arguments:
        - sub: e.g. '008'
        - SMOOTH_MILLISEC: milliseconds window for envelop smoothing
        - THRESH_ORIGIN: off or on or combi
        - TO_SAVE_FIG: to save or not
        - FIG_DIR: needed if figure should be saved
        - sub_SSD_class: result from ssd.get_subject_SSDs(),
            if given, this will decrease computational time
        - LOAD_STORED_RESULTS: make use previsouly created brust-values
    """
    print(f'start {sub}, burst feature-extraction')

    if not LOAD_STORED_RESULTS:
        burst_count = {}  # empty dict to store values
        
        # import SSD data if not defined
        if sub_SSD_class:
            ssd_dat = sub_SSD_class
        else:
            ssd_dat = get_subject_SSDs(
                sub=sub, incl_stn=True,
                ft_setting_fname='ftExtr_spectral_v1.json',)

        for beta_bw in ['lo_beta', 'hi_beta']:
            burst_count[beta_bw] = {}
            # load thresholds from JSON
            burst_thresh_dir = join(
                get_beta_project_path('results'),
                'bursts', 'thresholds')
            thrsh_fname = f'burst_thresholds_{beta_bw}.json'
            thrsh_path = join(burst_thresh_dir, thrsh_fname)

            with open(thrsh_path, 'r') as t:
                thresh = json.load(t)
            
            # if sub not in JSON: add to json and load again
            if not sub in thresh.keys():
                store_and_plot_beta_burst_thresholds(
                    sub=sub, ssd_dat=ssd_dat)
            with open(thrsh_path, 'r') as t:
                thresh = json.load(t)

            assert sub in thresh.keys(), (
                f'{sub} has no {beta_bw} thresholds saved'
                f' in {thrsh_path}')
                
            thresh = thresh[sub]
            
            # extract burst characteristics

            for dType in thresh.keys():  # loop over lfp_left/right and ecog_side
                
                burst_count[beta_bw][dType] = {
                    'short': [],  # store n-short / n-seconds bursts per window
                    'long': [],  # store n-long / n-seconds bursts per window
                    'burst_lengths': [],  # store lists with burst-lengths per window
                }

                # select SSD'd data to extract from
                tempdat = getattr(ssd_dat, dType)
                fs = tempdat.fs
                sig_array = getattr(tempdat, beta_bw)  # 2d array of shape n-windows, n-samples (per window) 
                burst_count[beta_bw][dType]['win_times'] = tempdat.times.copy()  # store timestamps corresponding to n-windows

                for sig in sig_array:
                    # loops over all windows                
                    env = abs(hilbert(sig))  # env from SSD'd timeseries of one window

                    if SMOOTH_MILLISEC > 0:
                        env = ftHelpers.smoothing(sig=env, fs=fs,
                                                win_ms=SMOOTH_MILLISEC)
                    
                    (n_short, n_long,
                    short_rate, long_rate,
                    burst_lengths
                    ) = burstFuncs.calc_bursts_from_env(
                        envelop=env,
                        fs=fs,
                        burst_thr=thresh[dType][THRESH_ORIGIN],
                        min_shortBurst_sec=.1,
                        min_longBurst_sec=.6,
                    )
                    burst_count[beta_bw][dType]['short'].append(short_rate)
                    burst_count[beta_bw][dType]['long'].append(long_rate)
                    burst_count[beta_bw][dType]['burst_lengths'].append(burst_lengths)
        
        if TO_STORE_BURST_LENGTHS:
            
            store_dir = join(get_beta_project_path('results'), 'bursts')
            if not exists(store_dir): makedirs(store_dir)
            fname = f'sub{sub}_bursts.json'
            
            # save values in json
            for bw in burst_count.keys():
                for dtype in burst_count[bw].keys():
                    burst_count[bw][dtype] = make_object_jsonable(
                        burst_count[bw][dtype])
            
            # add important settings variables to save
            burst_count['smooth_millisec'] = int(SMOOTH_MILLISEC)  # convert to json writable int
            burst_count['threshold_origin'] = THRESH_ORIGIN  # string is json writable
            burst_count['win_len_sec'] = float(tempdat.settings['WIN_LEN_sec'])
            burst_count['data_version'] = getattr(ssd_dat, list(thresh.keys())[0]).settings['DATA_VERSION']  # string is json writable

            with open(join(store_dir, fname), 'w') as f:
                json.dump(burst_count, f)
    
    elif LOAD_STORED_RESULTS:
        bursts_path = join(get_beta_project_path('results'),
                           'bursts', f'sub{sub}_bursts.json')

        with open(bursts_path, 'r') as f:
            burst_count = json.load(f)
                

    if not TO_PLOT_FIG and not TO_SAVE_FIG:
        return burst_count

    else:
        # plot results (only executed when not save AND not plot)
        plot_beta_bursts(
            sub=sub, burst_count=burst_count,
            SPLIT_OFF_ON=SPLIT_OFF_ON,
            TO_PLOT_FIG=TO_PLOT_FIG, TO_SAVE_FIG=TO_SAVE_FIG,
            FIG_DIR=FIG_DIR
        )

        return burst_count
    
   

def store_and_plot_beta_burst_thresholds(
    sub: str, ssd_dat, PERCENTILE: int = 75,
    INCL_TASKS='rest', data_version='v3.0',
):
    # define folders to store
    fig_dir = join(
        get_beta_project_path('figures'),
        'bursts', 'thresholds')
    burst_results_dir = join(
        get_beta_project_path('results'),
        'bursts')

    # get off/on definitions for subject
    fts = ssdFeatures(sub_list=[sub,])  # load features for CDRS scores
    win_sel = {}
    win_sel['off'], win_sel['on'] = define_OFF_ON_times(
        feat_times=ssd_dat.lfp_left.times.copy(),
        cdrs_scores=getattr(fts, f'sub{sub}').scores.total.copy(),
        cdrs_times=getattr(fts, f'sub{sub}').scores.times.copy(),
        min_ON_minutes=50,
        incl_tasks=INCL_TASKS,
        sub=sub, data_version=data_version,
    )
    
    threshs = {sub: {}}  # dict to store values

    for bw in ['lo_beta', 'hi_beta']:
        # create json/fig filename for sub/bandwidth
        json_fname = f'burst_thresholds_{bw}.json'
        json_path = join(burst_results_dir, json_fname)
        fig_fname = f'{sub}_{bw.upper()}_burst_75percentiles'
        
        # new figure per source and bandwitdh
        fig, axes = plt.subplots(1, 3, figsize=(8, 4))

        for i_src, src in enumerate(
            ['lfp_left', 'lfp_right',
             f'ecog_{get_ecog_side(sub)}']
        ):
            threshs[sub][src] = {}  # to store as json
            beta_ssd = getattr(ssd_dat, src)  # get SSD timeseries
            beta_ssd = getattr(beta_ssd, bw).copy()

            beta_env = abs(hilbert(beta_ssd))  # convert all beta-timeseries to analytic signal

            for cond in ['off', 'on']:

                env = beta_env[win_sel[cond]]  # select off or on windows
                percs = []  # store percentiles per window

                for w in np.arange(sum(win_sel[cond])):
                    thrs = np.nanpercentile(env[w], PERCENTILE)
                    percs.append(thrs)

                threshs[sub][src][cond] = np.nanmean(percs)

                axes[i_src].hist(percs, alpha=.5,
                        label=f'{cond.upper()}, mean: '
                        f'{np.nanmean(percs).round(3)}')
                
            axes[i_src].set_xlabel("75-percentile (SSD'd envelop)")
            axes[i_src].set_ylabel('# observations (windows)')
            axes[i_src].legend()
            axes[i_src].set_title(f'{src}')

        plt.suptitle(f'sub {sub} - 75-percentile of {bw}-envelope',
                    weight='bold',)
        plt.tight_layout()
        plt.savefig(join(fig_dir, fig_fname), dpi=150,
                    facecolor='w')
        plt.close()

        if exists(json_path):
            with open(json_path, 'r') as json_file:
                json_treshs = json.load(json_file)
        else:
            json_treshs = {}
        # fill new found values in dict
        json_treshs[sub] = threshs[sub]
        # write updated dict to json
        with open(json_path, 'w') as json_file:
            json.dump(json_treshs, json_file)
