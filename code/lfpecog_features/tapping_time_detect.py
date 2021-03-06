'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import variation
from datetime import datetime, timedelta
from itertools import compress

# Import own functions
import lfpecog_features.tapping_preprocess as preprocess
from lfpecog_features.tapping_impact_finder import find_impacts


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
    for n, y in enumerate(sig[:-1]):

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
    # print(tapi)
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

        

def continTapDetector(
    acc_triax, main_ax_i: int, fs: int,
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
        - acc_triax (arr): tri-axial accelerometer data-
            array containing x, y, z.
        - main_ax_i (int): index of axis which detected
            strongest signal during tapping (0, 1, or 2)
        - fs (int): sample frequency in Hz
    
    Return:
        - tapi (list of lists): list with full-recognized taps,
            every list is one tap. Every list contains 6 moments
            of the tap: [start-UP, fastest-UP, end-UP,
            start-DOWN, fastest-DOWN, end-DOWN]
        - tapTimes (list of lists): lists per tap corresponding
            with tapi, expressed in seconds after start data array
        - endPeaks (array): indices of impact-peak which correspond
            to end of finger closing moment.
    """
    sig = acc_triax[main_ax_i]
    sigdf = np.diff(sig)
    timeStamps = np.arange(0, len(sig), 1 / fs)

    # Thresholds for movement detection
    posThr = np.nanmean(sig)
    negThr = -np.nanmean(sig)
    
    # Find peaks to help movement detection
    peaksettings = {
        'peak_dist': 0.1,
        'cutoff_time': .25,
    }

    _, impacts = find_impacts(sig, fs)  # use v2 for now

    posPeaks = find_peaks(
        sig,
        height=(posThr, np.nanmax(sig)),
        distance=fs * .05,
    )[0]

    for i in impacts:
        idel = np.where(posPeaks == i)
        posPeaks = np.delete(posPeaks, idel)

    negPeak = find_peaks(
        -1 * sig,
        height=-.5e-7,
        distance=fs * peaksettings['peak_dist'] * .5,
        prominence=abs(np.nanmin(sig)) * .05,
    )[0]

    # Lists to store collected indices and timestamps
    tapi = []  # list to store indices of tap
    empty_timelist = np.array([np.nan] * 7)
    # [startUP, fastestUp, stopUP, startDown, fastestDown, impact, stopDown]
    tempi = empty_timelist.copy()
    state = 'lowRest'
    post_impact_blank = int(fs / 1000 * 15)  # 10 msec
    blank_count = 0

    # Sample-wise movement detection        
    for n, y in enumerate(sig[:-1]):

        if n in impacts:
            state = 'impact'
            tempi[5] = n
        
        elif state == 'impact':
            if blank_count < post_impact_blank:
                blank_count += 1
                continue
            
            else:
                if sigdf[n] > 0:
                    blank_count = 0
                    tempi[6] = n
                    tapi.append(np.array(tempi))
                    tempi = empty_timelist.copy()
                    state='lowRest'
                    

        elif state == 'lowRest':
            if np.logical_and(
                y > posThr,
                sigdf[n] > np.percentile(sigdf, 75)
            ):                
                state='upAcc1'
                tempi[0] = n  # START OF NEW TAP, FIRST INDEX
                
        elif state == 'upAcc1':
            if n in posPeaks:
                state='upAcc2'

        elif state == 'upAcc2':
            if y < 0:  # crossing zero-line, start of decelleration
                tempi[1] = n  # save n as FASTEST MOMENT UP
                state='upDec1'

        elif state=='upDec1':
            if n in posPeaks:  # later peak found -> back to up-accel
                state='upAcc2'
            elif n in negPeak:
                state='upDec2'

        elif state == 'upDec2':
            if np.logical_or(y > 0, sigdf[n] < 0):
                # if acc is pos, or goes into acceleration
                # phase of down movement
                state='highRest'  # end of UP-decell
                tempi[2]= n  # END OF UP !!!

        elif state == 'highRest':
            if np.logical_and(
                y < negThr,
                sigdf[n] < 0
            ):
                state='downAcc1'
                tempi[3] = n  # START OF LOWERING            

        elif state == 'downAcc1':
            if np.logical_and(
                y > 0,
                sigdf[n] > 0
            ):
                state='downDec1'
                tempi[4] = n  # fastest down movement

    
    tapi = tapi[1:]  # drop first tap due to starting time

    return tapi, impacts


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


    # # saving data array  -> DONT SAVE DATA PER RUN, COLLET ALL RUNS IN
    # # OVERALL FUNCTION AND SAVE ONE OVERALL FILES WITH DATA ROWS
    # # AND A CORRESPONDING ROWTIMES FILE
    # fname = f'{runname}_{neu_ch}_win{winlen}'
    # # np.save(os.path.join(
    # #     tempdir, 'restblocks', f'restBlocks_sub08_{fname}'), tempdat)
    # # save list of rowtimes
    # np.save(os.path.join(
    #     savedir, f'restTimes_sub08_{fname}'), rowTimes)


    # # return tempdat, rowTimes

    # ### Saving Rest Blocks
    # # determine neurophys axis before function input, function input
    # # only one timeseries
    # for neusource in ['lfp_left', 'ecog', 'lfp_right']:
    #     SelAndSave_Restblocks(
    #         neu_data = getattr(SUB08.runs[run], f'{neusource}_arr'),
    #         fs = getattr(SUB08.runs[run], f'{neusource}_Fs'),
    #         neu_names = getattr(SUB08.runs[run], f'{neusource}_names'),
    #         restTimes=restTimes,
    #         runname=run[-6:], neu_ch_incl=['ECOG_L_1', 'LFP_L_3_4']
    #     )