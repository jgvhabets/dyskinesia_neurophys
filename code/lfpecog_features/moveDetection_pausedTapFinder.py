'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from ast import Index
from signal import SIG_IGN
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import variation
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
        - tap_move_distance: # seconds which have to between
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

                # # Find peaks to help movement detection
                # peaksettings = {
                #     'peak_dist': .5,
                #     'cutoff_time': 2,
                # }

                # smallPos = find_peaks(
                #     sig,
                #     height=(svm_thr, .5e-6),    #(np.max(sig) * .1, np.max(sig) * .6),  # first value is min, second is max
                #     distance=fs,  # 1 s
                #     # prominence=np.max(sig) * .1,
                #     # wlen=fs * .5,
                # )[0]
                largePos = find_peaks(
                    sig,
                    height=.5e-6,   #np.max(sig) * .5,
                    distance=fs,  # 1 s
                )[0]
                smallPeaks = find_peaks(
                    svm,  # convert pos/neg for negative peaks
                    height=svm_thr,    #(abs(np.min(sig)) * .1, abs(np.min(sig)) * .4),
                    distance=fs,  # 10 ms
                    # prominence=abs(np.min(sig)) * .05,
                    # wlen=40,
                )[0]
                largeNeg = find_peaks(
                    -1 * sig,
                    height=.5e-6,   #abs(np.min(sig)) * .4,  # first value is min, second is max
                    distance=fs,  # 10 ms
                    # prominence=np.min(yEpoch) * .1,
                    # wlen=40,
                )[0]

                # hWin =   # tap is until half a second
                
                # define peak-timings
                if np.logical_and(
                    largePos != [], largeNeg != []
                ):

                    for posP in largePos:

                        if min(abs(posP - largeNeg)) < (fs * .5):
                        
                            # only include tap (pos and neg-peak) if they are within 0.5 sec
                            negP = largeNeg[np.argmin(abs(posP - largeNeg))]
                            if posP > negP: temp_tap = [np.nan, negP, posP, np.nan]
                            if posP < negP: temp_tap = [np.nan, posP, negP, np.nan]

                            # search for beginning and end of tapping movement
                            # based on signal vector activity
                            i = posP
                            while np.isnan(temp_tap[3]):

                                i += 1
                                try:
                                    if svm[i] < svm_thr: temp_tap[3] = i
                                except IndexError:
                                    break  # skip tap without inactive border
                            
                            i = posP
                            while np.isnan(temp_tap[0]):

                                i -= 1
                                if svm[i] < svm_thr: temp_tap[0] = i

                            try:
                                tap_t_list.append([sigTimes[t] for t in temp_tap])
                                tap_i_list.append([sigInds[t] for t in temp_tap])
                            except ValueError:
                                continue  # skip where nan is left bcs of no border
                            except IndexError:
                                continue
                
                # exclude small Peaks close to Tapping-Peaks
                maxgap = tap_move_distance * fs
                
                if len(smallPeaks) > 0:
                    sel_peaks = []
                    # try:
                    for p in smallPeaks:
                        try:
                            dist = abs(p - largePos)
                            if min(dist) > maxgap: sel_peaks.append(p)
                        except ValueError:
                            if largePos == []:
                                sel_peaks.append(p)
        
                    move_t_list.extend([sigTimes[t] for t in sel_peaks])
                    move_i_list.extend([sigInds[t] for t in sel_peaks])


                # plt.figure(figsize=(24, 12))
                # plt.plot(
                #     sig, label=f'sig ({mainAx_ind})', c='k', ls='dotted', alpha=.5,
                # )
                # # plt.plot(
                # #     sig3ax, alpha=.5, label=['x','y','z']
                # # )
                # # plt.plot(np.diff(svm), alpha=.5, label='sigSVM')
                # plt.plot(svm, alpha=.3, c='r', label='svm')
                # scatY = [.5, 2, -2]
                # labels = ['small Peaks (svm)', 'large POS', 'large NEG']
                # for n, peaks in enumerate([smallPeaks, largePos, largeNeg]):
                #     plt.scatter(
                #         peaks, [scatY[n] * 1e-6] * len(peaks),
                #         label=labels[n], s=70,)
                
                # plt.axhline(svm_thr, label=f'svm thr @ {svm_thr}')
                # plt.axhline(-svm_thr, label=f'svm thr @ {svm_thr}')
                # plt.xlim(20000, 60000)
                # plt.ylim(-2.1e-6, 2e-6)
                # plt.legend(loc='lower center', ncol=3, fontsize=14)
                # plt.title(f'{side} B # {bN}')
                # plt.show()

            out_lists[f'{side}_tap_t'].extend(tap_t_list)
            out_lists[f'{side}_tap_i'].extend(tap_i_list)
            out_lists[f'{side}_move_t'].extend(move_t_list)
            out_lists[f'{side}_move_i'].extend(move_i_list)

        # out_lists[f'{side}_tap_t'] = tap_t_list
        # out_lists[f'{side}_tap_i'] = tap_i_list
        # out_lists[f'{side}_move_t'] = move_t_list
        # out_lists[f'{side}_move_i'] = move_i_list    
    
    return out_lists

    
            
            
            # act_win = fs * 10
            # state='rest'

            # for n, y in enumerate(sig):

            #     if np.isnan(y): continue
                
            #     if np.logical_and(
            #         n < hWin,
            #         n > (len(sig) - hWin)
            #     ): continue

            #     win = svm[int(n - hWin):int(n + hWin)]
            #     win_active = sum(win > svm_thr) > hWin  # more than half of window is active

            #     if state == 'rest':
            #         print(n)
            #         print(win > svm_thr)
            #         sum(win > svm_thr)
            #         print(win_active)

            #         if win_active:

            #             print('from rest 2 active')

            #             state = 'move'
            #             move_i = move_template.copy()
            #             move_t = move_template.copy()
            #             move_i[0] = sigInds[n]
            #             move_t[0] = sigTimes[n]
                    
            #         else: continue  # state stays rest
                
            #     else:   # state is move
                    
            #         if ~ win_active:  # less than window is active

            #             move_i[-1] = sigInds[n]
            #             move_t[-1] = sigTimes[n]

            #             if np.logical_and(
            #                 ~np.isnan(move_i[2]),
            #                 ~np.isnan(move_i[3])
            #             ):  # contains large POS and NEG

            #                 tap_i_list.append(move_i)
            #                 tap_t_list.append(move_t)
            #                 print(move_t, move_i)

            #                 state = 'rest'
                        
            #             else:

            #                 move_i_list.append(move_i)
            #                 move_t_list.append(move_t)

            #                 state = 'rest'
                    
            #         else:  # window is active

            #             if n in smallPos:

            #                 move_i[1] = sigInds[n]
            #                 move_t[1] = sigTimes[n]
                        
            #             elif n in largePos:

            #                 move_i[2] = sigInds[n]
            #                 move_t[2] = sigTimes[n]

            #             elif n in largeNeg:

            #                 move_i[3] = sigInds[n]
            #                 move_t[3] = sigTimes[n]

        # # Lists to store collected indices and timestamps
        # tapi = []  # list to store indices of tap
        # movei = []  # list to store indices of other move
        # resti = []  # list to store indices of rest
        # resttemp = []  # list to temporarily collect rest-indices
        # tempi = []  # for during detection process
        # state = 'lowRest'

        # # Sample-wise movement detection
        # # Thresholds for movement detection
        # # posThr = .5e-7  # Try on new patients: adjust to person or tasks?!
        # # posThr = .5e-7
        # posSig = sig[sig > 1e-8]
        # posThr = np.percentile(posSig, 75)  # TRY OUT
        # print(posThr, len(largePos[0]), len(largeNeg[0]))
        # negThr = -posThr
        # for n, y in enumerate(sig[:-1]):

        #     if state == 'otherMov':
        #         if np.logical_and(
        #             y > posThr,
        #             # any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
        #             any([Y in largePos[0] for Y in range(n, n + int(fs))])  # TRY OUT
        #         ):  # from other Movement into tap-begin
        #             tempi.append(n)  # END of OTHER MOVE
        #             if (tempi[-1] - tempi[0]) > fs * .1:
        #                 movei.append(tempi)
        #             tempi = []  # start new index list
        #             state='upAcc1'
        #             tempi.append(n)  # START TIME Tap-UP
        #             print('start of TAP/MOV from OTHER MOVEM', y, n)
        #             maxUpAcc = np.max(sig[n:n + int(fs * .1)])
        #             continue

        #         try:
        #             next10 = sum([negThr < Y < posThr for Y in sig[range(
        #                 n, n + int(fs * .2)
        #             )]])
        #             if next10 > (fs * .2) * .8:
        #                 # End 'other move' if 8 / 10 next samples are inactive
        #                 tempi.append(n)  # END of OTHER MOVE
        #                 if (tempi[-1] - tempi[0]) > fs * .1:
        #                     movei.append(tempi)
        #                 tempi = []
        #                 state = 'lowRest'
        #                 print('ended MOVE bcs next10 inactive')
        #         except IndexError:  # prevent indexerror out of range for next10
        #             print('end of timeseries')
        #             continue

        #     elif state == 'lowRest':
        #         if np.logical_and(
        #             y > posThr,
        #             # any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
        #             any([Y in largePos[0] for Y in range(n, n + int(fs))])  # TRY OUT
        #         ):
        #             print('logAND is TRUE from rest', n)

        #             if resttemp:  # close and store active rest period
        #                 resttemp.append(n)  # Add second and last rest-ind
        #                 if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
        #                     resti.append(resttemp)  # add finished rest-indices
        #                 resttemp = []  # reset resttemp list
                    
        #             state='upAcc1'
        #             tempi.append(n)  # START TIME Tap-UP
        #             print('start of TAP/MOV', y, n)
        #             maxUpAcc = np.max(sig[n:n + int(fs * .1)])

        #         elif np.logical_or(
        #                 np.logical_or(n in smallPos[0], n in smallNeg[0]),
        #                 ~ (negThr < y < posThr)
        #         ):

        #             if resttemp:  # close and store active rest period
        #                 resttemp.append(n)  # Add second and last rest-ind
        #                 if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
        #                     resti.append(resttemp)  # add finished rest-indices
        #                 resttemp = []  # reset resttemp list

        #             state = 'otherMov'
        #             tempi.append(n)  # START TIME Othermovement
        #             # print('START TIME OTHER MOVEMENT')
                
        #         else:  # lowRest stays lowRest
        #             if not resttemp:  # if rest-temp list is empty
        #                 resttemp.append(n)  # start of rest period
                    
        #     elif state == 'upAcc1':
        #         if y == maxUpAcc:
        #             state='upAcc2'  # after acc-peak

        #     elif state == 'upAcc2':  # TRY OUT
        #         if y < 0:   #negThr < y < posThr:
        #             tempi.append(n)  # add moment of MAX UP-SPEED
        #     #         state='upDec0'
        #     #         maxUpDecc = np.min(sig[n:n + int(fs * .1)])

        #     # elif state == 'upDec0':  # prevent taking an early small neg peak
        #     #     if y == maxUpDecc:
        #             state = 'upDec1'

        #     elif state=='upDec1':
        #         # if n - tempi[0] > (fs * peaksettings['cutoff_time']):
        #         #     # if movement-up takes > defined cutoff time
        #         #     state = 'otherMov'  # reset to start-state
        #         #     movei.append(tempi)  # was untoggled?
        #         #     tempi = []  # was untoggled?
        #         # elif n in smallNeg[0]:
        #         if np.logical_and(
        #             y < negThr, sigdiff[n] > 0
        #         ):  # ACC is increasing after neg peak
        #             state='upDec2'
        #             print('after lowpeak(UP) coming back up', n)

        #     elif state == 'upDec2':
        #         if y > 0:  #negThr < y < posThr:
        #             state='highRest'
        #             tempi.append(n)  # end of Up
        #             print('endUP', n)

        #     elif state == 'highRest':
        #         if n - tempi[2] > (fs * peaksettings['cutoff_time']):
        #             # if highrest takes > defined cutoff time
        #             state = 'otherMov'  # reset to start-state
        #             movei.append(tempi)  # was untoggled?
        #             tempi = []  # was untoggled?
        #         elif np.logical_and(
        #             # np.logical_and(
        #                 y < negThr, sigdiff[n] < negThr#,
        #             # (any([Y in largeNeg[0] for Y in range(n, n + int(fs * .2))])
        #         ):
        #             state='downAcc1'
        #             print('START DOWN MOVEMENT', n)
        #             tempi.append(n)  # start of Tap-DOWN

        #     elif state == 'downAcc1':
        #         if n in largeNeg[0]:
        #             state='downAcc2'
        #         elif np.logical_and(
        #             y < negThr, sigdiff[n] > posThr
        #         ):
        #             state='downAcc2'
        #         elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
        #             state = 'otherMov'  # reset to start-state
        #             movei.append(tempi)
        #             tempi = []

        #     elif state == 'downAcc2':
        #         if np.logical_and(
        #             y < 0,
        #             sigdiff[n] > posThr
        #         ):
        #             print('Fastest DOWN MOVEMENT', n)
        #             tempi.append(n)  # fastest point in TAP-DOWN
        #             state='downDec1'

        #         elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
        #             state = 'otherMov'  # reset to start-state
        #             movei.append(tempi)
        #             tempi = []
                    
        #     elif state=='downDec1':
        #         if n in largePos[0]:
        #             state='downDec2'
        #         elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
        #             state = 'otherMov'  # reset to start-state
        #             movei.append(tempi)
        #             tempi = []
                    
        #     elif state == 'downDec2':
        #         if negThr < y < posThr:
        #             state='lowRest'
        #             tempi.append(n)  # end point of DOWN-TAP
        #             tapi.append(tempi)
        #             tempi = []

        # # remove otherMovements directly after tap
        # # print(tapi)
        # if tapi and movei:  # only when both exist
        #     endTaps = [tap[-1] for tap in tapi]
        #     movei_sel = []
        #     for tap in movei:
        #         if min([abs(tap[0] - end) for end in endTaps]) > (.2 * fs):
        #             movei_sel.append(tap)
        #     movei = movei_sel

        # # convert detected indices-lists into timestamps
        # tapTimes = []  # list to store timeStamps of tap
        # moveTimes = []  # alternative list for movements
        # restTimes = []  # list to sore rest-timestamps
        # for tap in tapi: tapTimes.append(sigTimes[tap])
        # for tap in movei: moveTimes.append(sigTimes[tap])
        # for tap in resti: restTimes.append(sigTimes[tap])

    # return tapTimes, moveTimes, restTimes


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

