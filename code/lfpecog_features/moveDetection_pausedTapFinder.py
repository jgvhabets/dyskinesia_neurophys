'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from ast import Index
from signal import SIG_IGN
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import variation
from scipy.ndimage import uniform_filter1d
from datetime import datetime, timedelta
from itertools import compress

import matplotlib.pyplot as plt

# Import own functions
import lfpecog_features.moveDetection_preprocess as movePrep

def pausedTapDetector(
    subdat, tap_move_distance=1, svm_thr=1e-7,
):
    """
    Detect the moments of finger-raising and -lowering
    during a fingertapping task.
    Function detects the axis with most variation and then
    first detects several large/small pos/neg peaks, then
    the function determines sample-wise in which part of a
    movement or tap the acc-timeseries is, and defines the
    exact moments of finger-raising, finger-lowering, and
    the in between stopping moments. 

    Input:
        - subdat (Class): resulting from subjectData()
        - tap_move_distance: n seconds which have to between
            tap and other movement; too close other movements
            get excluded
    
    Return:
        - outlists (list of lists):
            for tap-lists: each tap-list contains 4
            dopa-times or dopa-indices [start-tap, 1st-peak,
            2nd-peak, end-tap]
            for move-lists: one move-list contains the
            peak-dopa-times or peak-dopa-indices of other
            movements
    """
    # find present acc-sides
    acc_sides = [
        s for s in [
            'left', 'right'
        ] if f'acc_{s}' in vars(subdat).keys()
    ]

    assert len(acc_sides) == 2, print(
        f'\n\t{len(acc_sides)} ACC sides found for sub '
        f'{subdat.sub}, Class-keys: {vars(subdat).keys()}'
    )

    out_lists = {}  # store output

    for side in acc_sides:
        print(
            'start movement detection sub '
            f'{subdat.sub} {side} side'
        )
        # empty lists to store final outputs
        for v in ['t', 'i']:
            for m in ['tap', 'move']:
                out_lists[f'{side}_{m}_{v}'] = []

        fs = int(getattr(subdat, f'acc_{side}').fs)
        accDf = getattr(subdat, f'acc_{side}').data
        ax_sel = [
            k for k in accDf.keys() if 'ACC' in k
        ]
        if len(ax_sel) == 0:
            ax_sel = [
                k for k in accDf.keys() if k in ['X', 'Y', 'Z']
            ]

        acc_arr = accDf[ax_sel].values

        # PM check for polarity with prep-function

        mainAx_ind = movePrep.find_main_axis(acc_arr)

        times = accDf['dopa_time'].values
        
        for task in ['tap', 'rest']:
            # perform per task-block
            taskBlock_inds = movePrep.find_task_blocks(accDf, task)  # gives DF-indices

            tap_i_list, tap_t_list = [], []  # start new list per side (per sub)
            move_i_list, move_t_list = [], []  # start new list per side (per sub)
            # move_template = [np.nan] * 5  # STOP/smallPos/largePos/largeNeg/STOP

            for bN, (iB1, iB2) in enumerate(
                zip(taskBlock_inds[0], taskBlock_inds[1])
            ):

                # print(f'START block #{bN}')
                sig3ax = acc_arr[iB1:iB2, :]
                sig = sig3ax[:, mainAx_ind]
                # sigdiff = np.diff(sig)  # add diff of signal
                sigInds = np.arange(iB1, iB2)
                sigTimes = times[iB1:iB2]
                svm = movePrep.signalvectormagn(sig3ax)
                smoothsvm = uniform_filter1d(
                    svm, size=int(fs / 4)
                )

                # # Find peaks to help movement detection
                largePos = find_peaks(
                    sig,
                    height=5e-7,   #np.max(sig) * .5,
                    distance=fs,  # 1 s
                )[0]
                smallPeaks = find_peaks(
                    svm,
                    height=2.5e-7,    #(abs(np.min(sig)) * .1, abs(np.min(sig)) * .4),
                    distance=fs,  # 10 ms
                )[0]
                largeNeg = find_peaks(
                    -1 * sig,  # convert pos/neg for negative peaks
                    height=5e-7,   #abs(np.min(sig)) * .4,  # first value is min, second is max
                    distance=fs,  # 10 ms
                )[0]
                
                # define peak-timings of TAPS
                if np.logical_or(
                    largePos == [], largeNeg == []
                ):
                    # not both large Pos and Neg peaks present
                    if largePos: otherLargePeaks = largePos
                    elif largeNeg: otherLargePeaks = largeNeg
                    else: otherLargePeaks = []

                else:  # both large Pos and Neg present

                    otherLargePeaks = []

                    for posP in largePos:
                        # check distance to closest negative peak
                        if min(abs(posP - largeNeg)) > (fs * .5):
                            # large peak without close negative peak
                            otherLargePeaks.append(posP)  # store for other movement

                        else:  # negative peak close enough to be a TAP
                        
                            # only include tap (pos and neg-peak) if they are within 0.5 sec
                            negP = largeNeg[np.argmin(abs(posP - largeNeg))]
                            # allocate two large peaks in tap-list
                            if posP > negP: temp_tap = [np.nan, negP, posP, np.nan]
                            if posP < negP: temp_tap = [np.nan, posP, negP, np.nan]

                            # search for beginning and end of tapping movement
                            # based on signal vector activity
                            temp_tap[0], temp_tap[-1] = find_local_act_borders(
                                posP, smoothsvm, svm_thr
                            )

                            try:
                                tap_t_list.append([sigTimes[t] for t in temp_tap])
                                tap_i_list.append([sigInds[t] for t in temp_tap])
                            
                            except ValueError:
                                continue  # skip where nan is left bcs of no border
                            
                            except IndexError:
                                print('TAP:',temp_tap)  # PM: exclusion of real-taps?
                                continue
                
                # exclude small Peaks close to Tapping-Peaks
                min_gap = tap_move_distance * fs
                
                if np.logical_and(
                    len(smallPeaks) == 0,
                    len(otherLargePeaks) == 0
                ): continue  # not small/other movements

                sel_peaks = []

                for p in smallPeaks:
                    # ignore moves too close to taps
                    try:
                        min_dist = min(abs(p - largePos))
                        if min_dist > min_gap:
                            sel_peaks.append(p)

                    except ValueError:
                        # if no large peaks in 
                        if largePos == []:
                            sel_peaks.append(p)
                
                for p in otherLargePeaks: sel_peaks.append(p)

                # find movement borders
                temp_moves = []
                for p in sel_peaks:
                    
                    i_s, i_e = find_local_act_borders(
                        p, smoothsvm, svm_thr
                    )
                    if (i_e - i_s) > ( 5 * fs): continue  # if window is too long (missing data)

                    temp_moves.append([i_s, i_e])
    
                temp_moves = remove_lists_with_NaN(temp_moves)

                try:
                    move_t_list.extend([sigTimes[t] for t in temp_moves])
                    move_i_list.extend([sigInds[t] for t in temp_moves])
                except ValueError:
                    continue  # skip where nan is left bcs of no border
                
                except IndexError:
                    print(temp_moves)
                    continue


            out_lists[f'{side}_tap_t'].extend(tap_t_list)
            out_lists[f'{side}_tap_i'].extend(tap_i_list)
            out_lists[f'{side}_move_t'].extend(move_t_list)
            out_lists[f'{side}_move_i'].extend(move_i_list)

    return out_lists


def remove_lists_with_NaN(
    list_of_lists
):
    sel = [~np.isnan(l).any() for l in list_of_lists]

    newlist = list(compress(list_of_lists, sel))

    return newlist


def find_local_act_borders(
    peak_i, svm_sig, svmThr
):
    pre_i = np.nan
    post_i = np.nan

    i = peak_i
    while np.isnan(pre_i):
        
        i -= 1
        try:
            if svm_sig[i] < svmThr: pre_i = i
        except IndexError:
            break  # skip if border is out of data
    
    i = peak_i
    while np.isnan(post_i):
        
        i += 1
        try:
            if svm_sig[i] < svmThr: post_i = i
        except IndexError:
            break  # skip if border is out of data
    
    return pre_i, post_i


def saveAllEphysRestblocks(
    ephysdata, fs, restTimes, dopaIntakeTime, runStart,
    savedir, ephysname, runname, winlen=1024,
):
    """
    Select ephys-data that correspond with detected
    accelerometer-rest moments. Prerequisite is that
    neurophysiology and accelerometer data come from
    identical time period. Sample frequencies can
    differ, but total time-period measured must be equal!
    Epochs are saved per window-length (default is 1024
    samples), this leads to not considering some data
    which falls out of this windows.

    Input:
        - ephysdata (arr): one-dimensional data array
            containing 1 timeseries of neurophys data

        - dopaIntakeTime (str): timepoint of L-Dopa
            intake in format 'YYYY-MM-DD HH-MM'

    """
    # timestamps from start (in sec)
    ephystime = np.arange(0, len(ephysdata), 1 / fs)
    # create empty nd-array to store rest data in epochs
    # of defined window length
    tempdat = np.zeros((1, winlen))
    rowTimes = []  # list to store times corresponding to data rows

    for timeIdx in restTimes[1:-1]:  # skip first and last
        # find acc-Rest-times in ephys-timestamps
        neuInd1 = np.where(ephystime == timeIdx[0])[0][0]
        neuInd2 = np.where(ephystime == timeIdx[1])[0][0]
        dat_sel = ephysdata[neuInd1:neuInd2]
        n_wins = len(dat_sel) // winlen
        dat_sel = np.reshape(
            dat_sel[:n_wins * winlen],
            (n_wins, winlen),
            order='C',  # fills row after row
        )
        tempdat = np.vstack([tempdat, dat_sel])
        # add timestamp of start-time for every included row
        for i in np.arange(n_wins):
            rowTimes.append(timeIdx[0] + (i * winlen / fs))

    rowTimes = np.round(rowTimes, 4)  # round .019999 values
    tempdat = tempdat[1:, :]  # exclude first row of zeros

    dopaIN = datetime.strptime(dopaIntakeTime, '%Y-%m-%d %H:%M')

