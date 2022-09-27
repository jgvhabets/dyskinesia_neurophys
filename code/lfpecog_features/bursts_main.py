"""
Main function to call burst-extraction
in neurophysiological signals.

Takes all arguments and calls executive
function in bursts_funcs.py
"""

# import public packages and functions
import numpy as np
import pandas as pd

# import own functions
from lfpecog_features.bursts_funcs import get_burst_features

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
            startWin = round(merged_df.index[0] / 180) - 1
            stopWin = round(merged_df.index[-1] / 180) + 1
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


