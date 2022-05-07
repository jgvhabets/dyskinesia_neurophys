'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import variation
from datetime import datetime, timedelta



# def conduct  -> PUT OVERALL CONDUCT FUNCTION AND BLOCKSAVE FUNCTION
# IN DIFFERENT .PY-FILE

def tapDetector(
    x, y, z, fs, task, side='right',
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
        - x, y, z (arr): all three one-dimensional data-
            arrays containing one acc-axis each. Exact
            labeling x/y/z is not important. Should have equal
            lengths. Typically timeseries from one run.
        - fs (int): corresponding sample frequency
        - side (string): side where acc-data origin from
    
    Return:
        - tapTimes (list of lists): each list contains 4 timestamps
            (in seconds from array-start) indicating moments of:
            [finger-raise start, finger raise end,
            finger-lowering start, finger-lowering end]
        - moveTimes, restTimes: idem but then for 'other
            movements' and rest periods (of > 1 sec), each list
            contains the first and last timestamp of move/rest
            period.
    """
    # input sanity checks
    assert len(x) == len(y), f'Arrays X and Y should have equal lengths'
    assert len(x) == len(z), f'Arrays X and Z should have equal lengths'
    assert side in ['left', 'right'], f'Side should be left or right'
    assert task in ['paused', 'continuous'], (f'Task should be paused '
        f'or continuous')

    timeStamps = np.arange(0, len(x), 1 / fs)  # timestamps from start (in sec)
    ax_arrays = [x, y, z]
    # Find axis with most variation
    maxVar = np.argmax([variation(arr) for arr in ax_arrays])
    # maxRMS = np.argmax([sum(arr) for arr in ax_arrays])
    sig = ax_arrays[maxVar]  # acc-signal to use
    if sig.max() > 1e-4: sig = sig * 1e-6
    
    # Find peaks to help movement detection
    peaksettings = {
        'continuous': {
            'peak_dist': 0.1,
            'cutoff_time': .25,
        },
        'paused': {
            'peak_dist': .5,
            'cutoff_time': 2,
        }
    }
    if task == 'paused':
        smallPos = find_peaks(
            sig,
            height=(np.max(sig) * .1, np.max(sig) * .6),  # first value is min, second is max
            distance=fs * .2,  # 1 s
            prominence=np.max(sig) * .1,
            wlen=fs * .5,
        )
        largePos = find_peaks(
            sig,
            height=np.max(sig) * .5,
            distance=fs,  # 1 s
        )
        smallNeg = find_peaks(
            -1 * sig,  # convert pos/neg for negative peaks
            height=(-.5e-7, abs(np.min(sig)) * .4),
            distance=fs * 0.2,  # 10 ms
            prominence=abs(np.min(sig)) * .05,
            # wlen=40,
        )
        largeNeg = find_peaks(
            -1 * sig,
            height=abs(np.min(sig)) * .4,  # first value is min, second is max
            distance=fs * .5,  # 10 ms
            # prominence=np.min(yEpoch) * .1,
            # wlen=40,
        )

    # Lists to store collected indices and timestamps
    tapi = []  # list to store indices of tap
    movei = []  # list to store indices of other move
    resti = []  # list to store indices of rest
    resttemp = []  # list to temporarily collect rest-indices
    tempi = []  # for during detection process
    state = 'lowRest'

    # Sample-wise movement detection
    if task == 'paused':
        # Thresholds for movement detection
        posThr = .5e-7  # Try on new patients: adjust to person or tasks?!
        negThr = -.5e-7
        for n, y in enumerate(sig):

            if state == 'otherMov':
                if np.logical_and(
                    y > posThr,
                    any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
                ):  # from other Movement into tap-begin
                    tempi = []  # start new index list
                    state='upAcc1'
                    tempi.append(n)  # START TIME Tap-UP
                    continue

                try:
                    next10 = sum([negThr < Y < posThr for Y in sig[range(
                        n, n + int(fs * .2)
                    )]])
                    if next10 > (fs * .2) * .8:
                        # End 'other move' if 8 / 10 next samples are inactive
                        tempi.append(n)  # END of OTHER MOVE
                        if (tempi[-1] - tempi[0]) > fs * .1:
                            movei.append(tempi)
                        tempi = []
                        state = 'lowRest'
                except IndexError:  # prevent indexerror out of range for next10
                    # print('end of timeseries')
                    continue

            elif state == 'lowRest':
                if np.logical_and(
                    y > posThr,
                    any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
                ):

                    if resttemp:  # close and store active rest period
                        resttemp.append(n)  # Add second and last rest-ind
                        if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
                            resti.append(resttemp)  # add finished rest-indices
                        resttemp = []  # reset resttemp list
                    
                    state='upAcc1'
                    tempi.append(n)  # START TIME Tap-UP

                elif np.logical_or(
                        np.logical_or(n in smallPos[0], n in smallNeg[0]),
                        ~ (negThr < y < posThr)
                ):

                    if resttemp:  # close and store active rest period
                        resttemp.append(n)  # Add second and last rest-ind
                        if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
                            resti.append(resttemp)  # add finished rest-indices
                        resttemp = []  # reset resttemp list

                    state = 'otherMov'
                    tempi.append(n)  # START TIME Othermovement
                
                else:  # lowRest stays lowRest
                    if not resttemp:  # if rest-temp list is empty
                        resttemp.append(n)  # start of rest period
                    
            elif state == 'upAcc1':
                if n in smallPos[0]:  # PAUSED
                    state='upAcc2'  # after acc-peak, movement still accelerating
                    # print('peakpos1', n)
            elif state == 'upAcc2':
                if y < 0:   #negThr < y < posThr:
                    state='upDec1'
                    ### TODO -> ADD THIS MOMENT AS MAX SPEED MOMENT
            elif state=='upDec1':
                if n - tempi[0] > (fs * peaksettings[task]['cutoff_time']):
                    # if movement-up takes > defined cutoff time
                    state = 'otherMov'  # reset to start-state
                    movei.append(tempi)  # was untoggled?
                    tempi = []  # was untoggled?
                elif n in smallNeg[0]:
                    state='upDec2'
                    # print('peakneg1', n)
            elif state == 'upDec2':
                if y > 0:  #negThr < y < posThr:
                    state='highRest'
                    tempi.append(n)  # end of Up
                    # print('endUP', n)
            elif state == 'highRest':
                if n - tempi[1] > (fs * peaksettings[task]['cutoff_time']):
                    # if highrest takes > defined cutoff time
                    state = 'otherMov'  # reset to start-state
                    movei.append(tempi)  # was untoggled?
                    tempi = []  # was untoggled?
                elif np.logical_and(
                    y < negThr,
                    any([Y in largeNeg[0] for Y in range(n, n + int(fs * .2))])
                ):
                    state='downAcc1'
                    tempi.append(n)  # start of Tap-DOWN

            elif state == 'downAcc1':
                if n in largeNeg[0]:
                    state='downAcc2'
                elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                    # if down-move takes > defined cutoff time
                    state = 'otherMov'  # reset to start-state
                    movei.append(tempi)  # newly added
                    tempi = []  # newly added

            elif state == 'downAcc2':
                if np.logical_or(
                    negThr < y < posThr,
                    y > posThr
                ):
                    state='downDec1'
                elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                    # if down-move takes > defined cutoff time
                    state = 'otherMov'  # reset to start-state
                    movei.append(tempi)  # newly added
                    tempi = []  # newly added
                    
            elif state=='downDec1':
                if n in largePos[0]:
                    state='downDec2'
                elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                    # if down-move takes > defined cutoff time
                    state = 'otherMov'  # reset to start-state
                    movei.append(tempi)  # newly added
                    tempi = []  # newly added
                    
            elif state == 'downDec2':
                if negThr < y < posThr:
                    state='lowRest'
                    tempi.append(n)  # end of DOWN
                    tapi.append(tempi)
                    tempi = []

        # remove otherMovements directly after tap
        if tapi and movei:  # only when both exist
            endTaps = [tap[-1] for tap in tapi]
            movei_sel = []
            for tap in movei:
                if min([abs(tap[0] - end) for end in endTaps]) > (.25 * fs):
                    movei_sel.append(tap)
            movei = movei_sel

    # convert detected indices-lists into timestamps
    tapTimes = []  # list to store timeStamps of tap
    moveTimes = []  # alternative list for movements
    restTimes = []  # list to sore rest-timestamps
    for tap in tapi: tapTimes.append(timeStamps[tap])
    for tap in movei: moveTimes.append(timeStamps[tap])
    for tap in resti: restTimes.append(timeStamps[tap])

    return tapTimes, moveTimes, restTimes


def ContTapDetector(
    x, y, z, fs, task, side='right',
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
        - x, y, z (arr): all three one-dimensional data-
            arrays containing one acc-axis each. Exact
            labeling x/y/z is not important. Should have equal
            lengths. Typically timeseries from one run.
        - fs (int): corresponding sample frequency
        - side (string): side where acc-data origin from
    
    Return:
        - tapTimes (list of lists): each list contains 4 timestamps
            (in seconds from array-start) indicating moments of:
            [finger-raise start, finger raise end,
            finger-lowering start, finger-lowering end]
        - moveTimes, restTimes: idem but then for 'other
            movements' and rest periods (of > 1 sec), each list
            contains the first and last timestamp of move/rest
            period.
    """
    # input sanity checks
    assert len(x) == len(y), f'Arrays X and Y should have equal lengths'
    assert len(x) == len(z), f'Arrays X and Z should have equal lengths'
    assert side in ['left', 'right'], f'Side should be left or right'
    assert task in ['paused', 'continuous'], (f'Task should be paused '
        f'or continuous')

    timeStamps = np.arange(0, len(x), 1 / fs)  # timestamps from start (in sec)
    ax_arrays = [x, y, z]
    # Find axis with most variation
    maxVar = np.argmax([variation(arr) for arr in ax_arrays])
    # maxRMS = np.argmax([sum(arr) for arr in ax_arrays])
    sig = ax_arrays[maxVar]  # acc-signal to use
    if sig.max() > 1e-4: sig = sig * 1e-6
    sigdf = np.diff(sig)
    # Thresholds for movement detection
    posThr = np.mean(sig)  # Try on new patients: adjust to person or tasks?!
    negThr = -np.mean(sig)
    
    # Find peaks to help movement detection
    peaksettings = {
        'peak_dist': 0.1,
        'cutoff_time': .25,
    }

    # find relevant positive peaks
    posPeaks = find_peaks(
        sig,
        height=(posThr, np.max(sig)),
        distance=fs * .05,  # settings[task]['peak_dist']
    )[0]
    # select Pos-peaks with surrounding >> Pos and Neg Diff
    endPeaks = [np.logical_or(
        any(sigdf[i -3:i + 3] < np.percentile(sig, 10)),  #np.percentile(sig, 90) * .5)
        any(sigdf[i -3:i + 3] > np.percentile(sig, 90))
    ) for i in posPeaks]
    endPeaks = posPeaks[endPeaks]
    # delete endPeaks from posPeaks
    for i in endPeaks:
        idel = np.where(posPeaks == i)
        posPeaks = np.delete(posPeaks, idel)
    # delete endPeaks which are too close after each other
    # by starting with std False before np.diff, the diff-scores
    # represent the distance to the previous peak
    tooclose = endPeaks[np.append(
        np.array(False), np.diff(endPeaks) < (fs / 6))]
    for p in tooclose:
        i = np.where(endPeaks == p)
        endPeaks = np.delete(endPeaks, i)
        posPeaks = np.append(posPeaks, p)

    smallNeg = find_peaks(
        -1 * sig,  # convert pos/neg for negative peaks
        height=(-.5e-7, abs(np.min(sig)) * .5),
        distance=fs * peaksettings['peak_dist'] * .5,
        prominence=abs(np.min(sig)) * .05,
        # wlen=40,
    )[0]

    largeNeg = find_peaks(
        -1 * sig,
        height=abs(np.min(sig)) * .4,  # first value is min, second is max
        distance=fs * peaksettings['peak_dist'],
        # prominence=np.min(yEpoch) * .1,
        # wlen=40,
    )[0]

    # Lists to store collected indices and timestamps
    tapi = []  # list to store indices of tap
    movei = []  # list to store indices of other move
    resti = []  # list to store indices of rest
    resttemp = []  # temp-list to collect rest-indices [1st, Last]
    starttemp = [np.nan] * 5  # for during detection process
    # [startUP, fastestUp, stopUP, startDown, stopDown]
    tempi = starttemp.copy()  # to start process
    state = 'lowRest'

    # Sample-wise movement detection        
    for n, y in enumerate(sig):

        if state == 'otherMov':
            if n in endPeaks:  # during other Move: end Tap
                tempi[-1] = n  # finish and store index list
                if (tempi[-1] - tempi[0]) > fs * .1:
                    movei.append(tempi)  # save if long enough
                state='lowRest'
                tempi = starttemp.copy()  # after end: start lowRest
                continue

            try:
                next10 = sum([negThr < Y < posThr for Y in sig[range(
                    n, n + int(fs * .2)
                )]])
                if next10 > (fs * .2) * .8:
                    # End 'other move' if 8 / 10 next samples are inactive
                    tempi[-1] = n  # END of OTHER MOVE
                    if (tempi[-1] - tempi[0]) > fs * .1:
                        movei.append(tempi)
                    tempi = starttemp.copy()  # after end: start lowRest
                    state = 'lowRest'
            except IndexError:  # prevent indexerror out of range for next10
                # print('end of timeseries')
                continue

        elif state == 'lowRest':
            if np.logical_and(
                y > posThr,  # if value is over pos-threshold
                sigdf[n] > np.percentile(sigdf, 75)  # AND diff is over Thr
                # any([Y in posPeaks for Y in range(n, n + int(fs * .2))])  # USED IN PAUSED
            ):
                if resttemp:  # close and store active rest period
                    resttemp.append(n)  # Add second and last rest-ind
                    if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
                        resti.append(resttemp)  # add finished rest-indices
                    resttemp = []  # reset resttemp list
                
                state='upAcc1'
                tempi[0] = n  # START TIME Tap-UP

            # elif np.logical_or(
            #         np.logical_or(n in posPeaks, n in smallNeg[0]),
            #         ~ (negThr < y < posThr)
            # ):

            #     if resttemp:  # close and store active rest period
            #         resttemp.append(n)  # Add second and last rest-ind
            #         if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
            #             resti.append(resttemp)  # add finished rest-indices
            #         resttemp = []  # reset resttemp list
                # state = 'otherMov'
                # tempi.append(n)  # START TIME Othermovement
            
            elif n in endPeaks:  # during lowRest, endPeak found
                resttemp.append(n)  # Add second and last rest-ind
                if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
                    resti.append(resttemp)  # add finished rest-indices
                resttemp = []  # reset resttemp list
                state='lowRest'
                tempi = starttemp.copy()  # after end: start lowRest
                continue

            else:  # lowRest stays lowRest
                if not resttemp:  # if rest-temp list is empty
                    resttemp.append(n)  # start of rest period
                
        elif state == 'upAcc1':
            if n in posPeaks:
                state='upAcc2'  # after acc-peak, movement still accelerating

        elif state == 'upAcc2':
            if y < 0:  # crossing zero-line, start of decelleration
                tempi[1] = n  # save n as FADTEST MOMENT UP
                state='upDec1'

        elif state=='upDec1':
            if n in smallNeg:
                state='upDec2'
        elif state == 'upDec2':
            if np.logical_or(y > 0, sigdf[n] < 0):  # acc is pos, or not increasing anymore
                state='highRest'  # end of UP-decell
                tempi[2]= n  # END OF UP !!!

######## TODO: GO FURTHER HERE (make ideal detect situation and assess later taps on completeness)

        elif state == 'highRest':
            # if n - tempi[1] > (fs * peaksettings[task]['cutoff_time']):
            #     # if highrest takes > defined cutoff time
            #     state = 'otherMov'  # reset to start-state
            #     movei.append(tempi)  # was untoggled?
            #     tempi = []  # was untoggled?
            if np.logical_and(
                y < negThr,
                sigdf[n] < np.percentile(sigdf, 25)  # already done
            ):
                state='downAcc1'
                tempi.append(n)  # start of Tap-DOWN

        elif state == 'downAcc1':
            if n in largeNeg[0]:
                state='downAcc2'
            elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                # if down-move takes > defined cutoff time
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)  # newly added
                tempi = []  # newly added

        elif state == 'downAcc2':
            if np.logical_or(
                negThr < y < posThr,
                y > posThr
            ):
                state='downDec1'
            elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                # if down-move takes > defined cutoff time
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)  # newly added
                tempi = []  # newly added
                
        elif state=='downDec1':
            if n in largePos[0]:
                state='downDec2'
            elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
                # if down-move takes > defined cutoff time
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)  # newly added
                tempi = []  # newly added
                
        elif state == 'downDec2':
            if negThr < y < posThr:
                state='lowRest'
                tempi.append(n)  # end of DOWN
                tapi.append(tempi)
                tempi = []

    # convert detected indices-lists into timestamps
    tapTimes = []  # list to store timeStamps of tap
    moveTimes = []  # alternative list for movements
    restTimes = []  # list to sore rest-timestamps
    for tap in tapi: tapTimes.append(timeStamps[tap])
    for tap in movei: moveTimes.append(timeStamps[tap])
    for tap in resti: restTimes.append(timeStamps[tap])

    return tapTimes, moveTimes, restTimes

def saveAllEphysRestblocks(
    ephysdata, fs, restTimes, dopaIntakeTime, runStart,
    savedir, ephysname, runname, winlen=1024,
):
# TODO: include dopa time
# TODO: include multiple runs in different start function
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
    # runStart into datetime delta from dopaIN (in seconds)
    # for every row runStartDopaDelta + rowSeconds --> DopaSeconds


    # saving data array  -> DONT SAVE DATA PER RUN, COLLET ALL RUNS IN
    # OVERALL FUNCTION AND SAVE ONE OVERALL FILES WITH DATA ROWS
    # AND A CORRESPONDING ROWTIMES FILE
    fname = f'{runname}_{neu_ch}_win{winlen}'
    # np.save(os.path.join(
    #     tempdir, 'restblocks', f'restBlocks_sub08_{fname}'), tempdat)
    # save list of rowtimes
    np.save(os.path.join(
        savedir, f'restTimes_sub08_{fname}'), rowTimes)


    # return tempdat, rowTimes

    ### Saving Rest Blocks
    # determine neurophys axis before function input, function input
    # only one timeseries
    for neusource in ['lfp_left', 'ecog', 'lfp_right']:
        SelAndSave_Restblocks(
            neu_data = getattr(SUB08.runs[run], f'{neusource}_arr'),
            fs = getattr(SUB08.runs[run], f'{neusource}_Fs'),
            neu_names = getattr(SUB08.runs[run], f'{neusource}_names'),
            restTimes=restTimes,
            runname=run[-6:], neu_ch_incl=['ECOG_L_1', 'LFP_L_3_4']
        )