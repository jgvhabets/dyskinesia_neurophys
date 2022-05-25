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
from itertools import compress



# def conduct  -> PUT OVERALL CONDUCT FUNCTION AND BLOCKSAVE FUNCTION
# IN DIFFERENT .PY-FILE

def pausedTapDetector(
    fs: int, x=[], y=[], z=[], side='right',
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
    if x != [] and y != []:
        assert len(x) == len(y), f'Arrays X and Y should'
        ' have equal lengths'
    if x != [] and z != []:
        assert len(x) == len(z), f'Arrays X and Z should'
        ' have equal lengths'
    if z != [] and y != []:
        assert len(y) == len(z), f'Arrays X and Z should'
        ' have equal lengths'
    assert side in ['left', 'right'], f'Side should be '
    'left or right'

    ax_arrs = []
    for ax in [x, y, z]:
        if ax != []: ax_arrs.append(ax)

    # Find axis with most variation
    maxVar = np.argmax([variation(arr) for arr in ax_arrs])
    # maxRMS = np.argmax([sum(arr) for arr in ax_arrays])
    sig = ax_arrs[maxVar]  # acc-signal to use
    # check data for pos/neg and order of magn
    # sig = check_PosNeg_and_Order(sig, fs)

    # # add differential  # OPTIMIZE MODEL LATER WITH DIFF
    sigdf = np.diff(sig)

    # timestamps from start (in sec)
    timeStamps = np.arange(0, len(sig), 1 / fs)
    
    # Find peaks to help movement detection
    peaksettings = {
        'peak_dist': .5,
        'cutoff_time': 2,
    }

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
    # Thresholds for movement detection
    # posThr = .5e-7  # Try on new patients: adjust to person or tasks?!
    # posThr = .5e-7
    posSig = sig[sig > 1e-8]
    posThr = np.percentile(posSig, 75)  # TRY OUT
    print(posThr, len(largePos[0]), len(largeNeg[0]))
    negThr = -posThr
    for n, y in enumerate(sig):

        if state == 'otherMov':
            if np.logical_and(
                y > posThr,
                # any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
                any([Y in largePos[0] for Y in range(n, n + int(fs))])  # TRY OUT
            ):  # from other Movement into tap-begin
                tempi.append(n)  # END of OTHER MOVE
                if (tempi[-1] - tempi[0]) > fs * .1:
                    movei.append(tempi)
                tempi = []  # start new index list
                state='upAcc1'
                tempi.append(n)  # START TIME Tap-UP
                print('start of TAP/MOV from OTHER MOVEM', y, n)
                maxUpAcc = np.max(sig[n:n + int(fs * .1)])
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
                    print('ended MOVE bcs next10 inactive')
            except IndexError:  # prevent indexerror out of range for next10
                print('end of timeseries')
                continue

        elif state == 'lowRest':
            if np.logical_and(
                y > posThr,
                # any([Y in smallPos[0] for Y in range(n, n + int(fs * .2))])
                any([Y in largePos[0] for Y in range(n, n + int(fs))])  # TRY OUT
            ):
                print('logAND is TRUE from rest', n)

                if resttemp:  # close and store active rest period
                    resttemp.append(n)  # Add second and last rest-ind
                    if (resttemp[1] - resttemp[0]) > fs:  # if rest > 1 sec
                        resti.append(resttemp)  # add finished rest-indices
                    resttemp = []  # reset resttemp list
                
                state='upAcc1'
                tempi.append(n)  # START TIME Tap-UP
                print('start of TAP/MOV', y, n)
                maxUpAcc = np.max(sig[n:n + int(fs * .1)])

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
                # print('START TIME OTHER MOVEMENT')
            
            else:  # lowRest stays lowRest
                if not resttemp:  # if rest-temp list is empty
                    resttemp.append(n)  # start of rest period
                
        elif state == 'upAcc1':
            if y == maxUpAcc:
                state='upAcc2'  # after acc-peak

        elif state == 'upAcc2':  # TRY OUT
            if y < 0:   #negThr < y < posThr:
                tempi.append(n)  # add moment of MAX UP-SPEED
        #         state='upDec0'
        #         maxUpDecc = np.min(sig[n:n + int(fs * .1)])

        # elif state == 'upDec0':  # prevent taking an early small neg peak
        #     if y == maxUpDecc:
                state = 'upDec1'

        elif state=='upDec1':
            # if n - tempi[0] > (fs * peaksettings['cutoff_time']):
            #     # if movement-up takes > defined cutoff time
            #     state = 'otherMov'  # reset to start-state
            #     movei.append(tempi)  # was untoggled?
            #     tempi = []  # was untoggled?
            # elif n in smallNeg[0]:
            if np.logical_and(
                y < negThr, sigdf[n] > 0
            ):  # ACC is increasing after neg peak
                state='upDec2'
                print('after lowpeak(UP) coming back up', n)

        elif state == 'upDec2':
            if y > 0:  #negThr < y < posThr:
                state='highRest'
                tempi.append(n)  # end of Up
                print('endUP', n)

        elif state == 'highRest':
            if n - tempi[2] > (fs * peaksettings['cutoff_time']):
                # if highrest takes > defined cutoff time
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)  # was untoggled?
                tempi = []  # was untoggled?
            elif np.logical_and(
                # np.logical_and(
                    y < negThr, sigdf[n] < negThr#,
                # (any([Y in largeNeg[0] for Y in range(n, n + int(fs * .2))])
            ):
                state='downAcc1'
                print('START DOWN MOVEMENT', n)
                tempi.append(n)  # start of Tap-DOWN

        elif state == 'downAcc1':
            if n in largeNeg[0]:
                state='downAcc2'
            elif np.logical_and(
                y < negThr, sigdf[n] > posThr
            ):
                state='downAcc2'
            elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)
                tempi = []

        elif state == 'downAcc2':
            if np.logical_and(
                y < 0,
                sigdf[n] > posThr
            ):
                print('Fastest DOWN MOVEMENT', n)
                tempi.append(n)  # fastest point in TAP-DOWN
                state='downDec1'

            elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)
                tempi = []
                
        elif state=='downDec1':
            if n in largePos[0]:
                state='downDec2'
            elif n - tempi[3] > (fs * peaksettings['cutoff_time']):
                state = 'otherMov'  # reset to start-state
                movei.append(tempi)
                tempi = []
                
        elif state == 'downDec2':
            if negThr < y < posThr:
                state='lowRest'
                tempi.append(n)  # end point of DOWN-TAP
                tapi.append(tempi)
                tempi = []

    # remove otherMovements directly after tap
    print(tapi)
    if tapi and movei:  # only when both exist
        endTaps = [tap[-1] for tap in tapi]
        movei_sel = []
        for tap in movei:
            if min([abs(tap[0] - end) for end in endTaps]) > (.2 * fs):
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


def check_PosNeg_and_Order(
    sig, fs,
):
    # check if pos/neg is switched in axis due to
    # different placement during experiment
    hop = 3
    impacts = []
    for n in np.arange(hop, fs * 5):
        if np.logical_and(
            any(np.diff(sig)[n - hop:n] >
                np.percentile(sig, 90)),
            any(np.diff(sig)[n- hop:n] <
                np.percentile(sig, 10))
        ):
            impacts.append(n)
    if len(impacts) < 10:
        numImp = len(impacts)
    else:
        numImp = 10
    # print(impacts, sig.shape)
    # if the mean around the impact moments is negative
    if np.mean([sig[impacts[i] - 2: impacts[i] + 2
            ] for i in np.arange(1, numImp)]) < 0:
        sig = sig * -1  # switch pos - neg
    
    # Check for order of magnitude
    # if not in order of 1e-6, then covert
    if sig.max() > 1e-4: sig = sig * 1e-6
    
    return sig
        

def continTapDetector(
    fs: int, x=[], y=[], z=[], side='right',
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
    if x != [] and y != []:
        assert len(x) == len(y), f'Arrays X and Y should'
        ' have equal lengths'
    if x != [] and z != []:
        assert len(x) == len(z), f'Arrays X and Z should'
        ' have equal lengths'
    if z != [] and y != []:
        assert len(y) == len(z), f'Arrays X and Z should'
        ' have equal lengths'
    assert side in ['left', 'right'], f'Side should be '
    'left or right'

    ax_arrs = []
    for ax in [x, y, z]:
        if ax != []: ax_arrs.append(ax)
    # Find axis with most variation
    maxVar = np.argmax([variation(arr) for arr in ax_arrs])
    # maxRMS = np.argmax([sum(arr) for arr in ax_arrays])
    sig = ax_arrs[maxVar]  # acc-signal to use
    # check data for pos/neg and order of magn
    sig = check_PosNeg_and_Order(sig, fs)
    
    # add differential of signal
    sigdf = np.diff(sig)
    # timestamps from start (in sec)
    timeStamps = np.arange(0, len(sig), 1 / fs)

    # Thresholds for movement detection
    posThr = np.mean(sig)
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
        any(sigdf[i -3:i + 3] < np.percentile(sig, 10)),
        any(sigdf[i -3:i + 3] > np.percentile(sig, 90))
    ) for i in posPeaks]
    endPeaks = posPeaks[endPeaks]
    # delete endPeaks from posPeaks
    for i in endPeaks:
        idel = np.where(posPeaks == i)
        posPeaks = np.delete(posPeaks, idel)
    # delete endPeaks which are too close after each other
    # by starting with std False before np.diff, the diff- 
    # scores represent the distance to the previous peak
    tooclose = endPeaks[np.append(
        np.array(False), np.diff(endPeaks) < (fs / 6))]
    for p in tooclose:
        i = np.where(endPeaks == p)
        endPeaks = np.delete(endPeaks, i)
        posPeaks = np.append(posPeaks, p)
    # double check endPeaks with np.diff
    hop = 3
    endP2 = []
    for n in np.arange(hop, sig.shape[0]):
        if np.logical_and(
            any(np.diff(sig)[n - hop:n] > np.percentile(sig, 90)),
            any(np.diff(sig)[n- hop:n] < np.percentile(sig, 10))
        ):  # if diff is above extremes within hop-distance
            endP2.append(n)
    endP2 = list(compress(endP2, np.diff(endP2) > hop))
    for p2 in endP2:  # add to endPeaks if not containing
        if min(abs(p2 - endPeaks)) > 5:
            endPeaks = np.append(endPeaks, p2)

    smallNeg = find_peaks(
        -1 * sig,  # convert pos/neg for negative peaks
        height=(-.5e-7, abs(np.min(sig)) * .5),
        distance=fs * peaksettings['peak_dist'] * .5,
        prominence=abs(np.min(sig)) * .05,
        # wlen=40,
    )[0]

    # largeNeg = find_peaks(
    #     -1 * sig,
    #     height=abs(np.min(sig)) * .4,
    #     # first value is min, second is max
    #     distance=fs * peaksettings['peak_dist'],
    #     # prominence=np.min(yEpoch) * .1,
    #     # wlen=40,
    # )[0]

    # Lists to store collected indices and timestamps
    tapi = []  # list to store indices of tap
    movei = []  # list to store indices of other move
    resti = []  # list to store indices of rest
    resttemp = []  # temp-list to collect rest-indices [1st, Last]
    starttemp = [np.nan] * 6  # for during detection process
    # [startUP, fastestUp, stopUP, 
    #  startDown, fastestDown, stopDown]
    tempi = starttemp.copy()  # to start process
    state = 'lowRest'

    # Sample-wise movement detection        
    for n, y in enumerate(sig[:-1]):

        if state == 'otherMov':
            # PM LEAVE OUT OTHER-MOV-STATE
            if n in endPeaks:  # during other Move: end Tap
                tempi[-1] = n  # finish and store index list
                if (tempi[-1] - tempi[0]) > fs * .1:
                    movei.append(tempi)  # save if long enough
                state='lowRest'
                tempi = starttemp.copy()  # after end: start lowRest
                continue

            try:
                next10 = sum([negThr < Y < posThr for Y in      sig[range(n, n + int(fs * .2)
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
                # print('save start UP', n)

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
                state='upAcc2'
                # acc getting less, veloc still increasing
                # print('acc-peakUP detected', n)

            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found

        elif state == 'upAcc2':
            if y < 0:  # crossing zero-line, start of decelleration
                tempi[1] = n  # save n as FASTEST MOMENT UP
                state='upDec1'
                # print('fastest point UP', n)

            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found

        elif state=='upDec1':
            if n in smallNeg:
                state='upDec2'

            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found

        elif state == 'upDec2':
            if np.logical_or(y > 0, sigdf[n] < 0):
                # if acc is pos, or goes into acceleration
                # phase of down movement
                state='highRest'  # end of UP-decell
                tempi[2]= n  # END OF UP !!!

            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found

        elif state == 'highRest':
            if np.logical_and(
                y < negThr,
                sigdf[n] < 0  #np.percentile(sigdf, 25)
                # from highRest: LOWERING starts when acc
                # gets below negative-threshold AND when
                # differential is negative
            ):
                state='downAcc1'
                tempi[3] = n  # START OF LOWERING
                # print('LOWERING START', n)
            
            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found

        # elif state == 'downAcc1':
        #     if n in largeNeg[0]:
        #         state='downAcc2'
        #     elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
        #         # if down-move takes > defined cutoff time
        #         state = 'otherMov'  # reset to start-state
        #         movei.append(tempi)  # newly added
        #         tempi = []  # newly added

        # elif state == 'downAcc2':
        elif state == 'downAcc1':
            if np.logical_and(
                y > 0,
                sigdf[n] > 0
            ):
            # if acceleration gets positive again and keeps
            # one increasing (sigdf) downwards acceleration
            # is finished -> ADD FASTEST DOWNW MOMENT
                state='downDec1'
                tempi[4] = n
                # print('fastest DOWN @', n)
            elif n in endPeaks:
                state = 'downDec2'
                # emergency out if endPeak is found


            # elif n - tempi[2] > (fs * peaksettings[task]['cutoff_time']):
            #     # if down-move takes > defined cutoff time
            #     state = 'otherMov'  # reset to start-state
            #     movei.append(tempi)  # newly added
            #     tempi = []  # newly added

        elif state == 'downDec1':
            if n in endPeaks:
                state = 'downDec2'

        elif state=='downDec2':
            if np.logical_or(
                y < 0,
                sigdf[n] < 0
            ):  # after large pos-peak, before around impact
            # artefectual peaks
                state='lowRest'
                tempi[5] = n
                # store current indices
                tapi.append(tempi)
                tempi = starttemp.copy()  # restart w/ 6*nan
    
    # drop first tap due to starting time
    tapi = tapi[1:]
    # convert detected indices-lists into timestamps
    tapTimes = []  # list to store timeStamps of tap
    # moveTimes = []  # alternative list for movements
    # restTimes = []  # list to sore rest-timestamps
    for tap in tapi: tapTimes.append(
        [timeStamps[I] for I in tap if I is not np.nan]
    )
    # for tap in movei: moveTimes.append(timeStamps[tap])
    # for tap in resti: restTimes.append(timeStamps[tap])

    return tapi, tapTimes, endPeaks


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