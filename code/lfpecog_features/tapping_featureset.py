'''
Functions to define tapping features
'''
# Import general packages and functions
import numpy as np

# Import own functions
from lfpecog_features.tapping_preprocess import find_main_axis


def signalvectormagn(acc_arr):
    """
    Input:
        - acc_arr (array): triaxial array
            with x-y-z axes (3 x n_samples)
    
    Returns:
        - svm (array): uniaxial array wih
            signal vector magn (1, n_samples)
    """
    if acc_arr.shape[0] != 3: acc_arr = acc_arr.T
    assert acc_arr.shape[0] == 3, ('Array must'
    'be tri-axial (x, y, z from accelerometer')
  
    svm = np.sqrt(
        acc_arr[0] ** 2 +
        acc_arr[1] ** 2 +
        acc_arr[2] ** 2
    )

    return svm


def calc_RMS(acc_signal):
    """
    Calculate RMS over preselected uniaxial
    acc-signal (can be uni-axis or svm)
    """
    S = np.square(acc_signal)
    MS = S.mean()
    RMS = np.sqrt(MS)

    return RMS


def tap_RMS(tapDict, triax_arr, ax, select: str='tap',
    impact_window: float=.25, fs: int=0):
    """
    Calculates RMS of full acc-signal per tap.

    Input:
        - tapDict (dict result of tap-detect - 
            continuous tap)
        - triax_arr (array)
        - select (str): if full -> return RMS per
            full tap; if impact -> return RMS
            around impact
        - impact_window (float): in seconds, total
            window around impact to calculate RMS
        - fs (int): sample frequency, default is
            zero to require input if impact is given
    
    Returns:
        - RMS_uniax (arr)
        - RMS_triax (arr)
    """
    select_options = ['run', 'tap', 'impact']
    assert select in select_options, ('select '
        'select variable incorrect for tap_RMS()')

    ax = triax_arr[ax]
    svm = signalvectormagn(triax_arr)

    if select == 'run':
        RMS_uniax = calc_RMS(ax)
        RMS_triax = calc_RMS(svm)

        return RMS_uniax, RMS_triax
    
    else:
        RMS_uniax = np.zeros(len(tapDict))
        RMS_triax = np.zeros(len(tapDict))

        for n, tap in enumerate(tapDict):
            tap = tap.astype(int)  # np.nan as int is -999999...

            if select == 'tap':
                sel1 = int(tap[0])
                sel2 = int(tap[-1])

            elif select == 'impact':
                sel1 = int(tap[-2] - int(fs * impact_window / 2))
                sel2 = int(tap[-2] + int(fs * impact_window / 2))

            if np.logical_or(sel1 == np.nan, sel2 == np.nan):
                print('tap skipped, missing indices')
                continue
            
            tap_ax = ax[sel1:sel2]
            tap_svm = svm[sel1:sel2]
            
            RMS_uniax[n] = calc_RMS(tap_ax)
            RMS_triax[n] = calc_RMS(tap_svm)
        
        return RMS_uniax, RMS_triax


def upTap_velocity(tapDict, triax_arr, ax):
    """
    Calculates velocity approximation via
    area under the curve of acc-signal within
    upwards part of a tap.

    Input:
        - tapDict
        - triax_arr
        - ax (int): main tap axis index
    
    Returns:
        - upVelo_uniax (arr)
        - upVelo_triax (arr)
    """
    ax = triax_arr[ax]
    svm = signalvectormagn(triax_arr)

    upVelo_uniax = velo_AUC_calc(tapDict, ax)
    upVelo_triax = velo_AUC_calc(tapDict, svm)
    
    return upVelo_uniax, upVelo_triax


import matplotlib.pyplot as plt


def velo_AUC_calc(tapDict, accSig,):
    """
    Calculates max velocity during finger-raising
    based on the AUC from the first big pos peak
    in one tap until the acceleration drops below 0

    Input:
        - tapDict (dict): containing lists with 6-
            timepoints per tap
        - accSig (array): uniax acc-array (one ax or svm)
    
    Returns:
        - out (array): one value or nan per tap in tapDict
    """
    out = []  #np.zeros(len(tapDict))

    for n, tap in enumerate(tapDict):

        if ~np.isnan(tap[1]):  # crossing 0 has to be known
            # take acc-signal [start : fastest point] of rise
            line = accSig[int(tap[0]):int(tap[1])]
            areas = []
            for s, y in enumerate(line[1:]):
                areas.append(np.mean([y, line[s]]))
            if sum(areas) == 0:
                print('\nSUM 0',n, line[:30], tap[0], tap[1])
            out.append(sum(areas))
    # print('out', out)
    
    return np.array(out)


def smallSlopeChanges(
    tempacc, resolution: str, n_hop: int=1,
    tapDict = []
):
    """
    Detects the number of small changes in
    direction of acceleration.
    Hypothesized is that best tappers, have
    the smoothest acceleration-trace and
    therefore lower numbers of small
    slope changes

    Inputs:
        - acc (array): tri-axial acceleration
            signal from e.g. 10-s tapping
        - n_hop (int): the number of samples used
            to determine the difference between
            two points
    
    Returns:
        - count (int): number of times the
            differential of all thee seperate
            axes changed in direction.
    """
    if resolution == 'run':

        count = 0
        for ax in [0, 1, 2]:

            diftemp = np.diff(tempacc[ax])
            for i in np.arange(diftemp.shape[0] - n_hop):
                if -1 < diftemp[i + n_hop] * diftemp[i] < 0:
                    count += 1

    elif resolution == 'taps':

        countlist = []

        for tap in tapDict:

            if np.logical_or(
                np.isnan(tap[0]),
                np.isnan(tap[-1])
            ):
                continue

            elif len(tap) == 0:
                continue
            
            else:
                tap_acc = tempacc[:, int(tap[0]):int(tap[-1])]
                count = 0

                for ax in [0, 1, 2]:
                    diftemp = np.diff(tap_acc[ax])

                    for i in np.arange(diftemp.shape[0] - n_hop):
                        if -1 < diftemp[i + n_hop] * diftemp[i] < 0:
                            count += 1
                
                countlist.append(count)

        count = np.array(countlist)

    return count



# ### Plot AUC-Method for velocity
# start=accTaps['40']['On'][6][0]
# stop=accTaps['40']['On'][6][1]
# plt.plot(accDat['40'].On[start:stop], label='Accelerating phase of finger raising')

# line = accDat['40'].On[start:stop]
# areas = []
# for n, y in enumerate(line[1:]):
#     areas.append(np.mean([y, line[n]]))
# plt.bar(
#     np.arange(.5, len(areas) + .5, 1), height=areas,
#     color='b', alpha=.2,
#     label='AUC extracted',
# )
# plt.ylabel('Acceleration (m/s/s)')
# plt.xlabel('Samples (250 Hz)')
# plt.legend(frameon=False)
# plt.savefig(
#     os.path.join(temp_save, 'ACC', 'fingerTap_AUC_method'),
#     dpi=150, facecolor='w',)
# plt.show()

## DEFINE FEATURE FUNCTIONS FROM MAHADEVAN 2020

def histogram(signal_x):
    '''
    Calculate histogram of sensor signal.
    :param signal_x: 1-D numpy array of sensor signal
    :return: Histogram bin values, descriptor
    '''
    descriptor = np.zeros(3)

    ncell = np.ceil(np.sqrt(len(signal_x)))

    max_val = np.nanmax(signal_x.values)
    min_val = np.nanmin(signal_x.values)

    delta = (max_val - min_val) / (len(signal_x) - 1)

    descriptor[0] = min_val - delta / 2
    descriptor[1] = max_val + delta / 2
    descriptor[2] = ncell

    h = np.histogram(signal_x, ncell.astype(int), range=(min_val, max_val))

    return h[0], descriptor


def signal_entropy(winDat):
    data_norm = winDat/np.std(winDat)
    h, d = histogram(data_norm)
    lowerbound = d[0]
    upperbound = d[1]
    ncell = int(d[2])

    estimate = 0
    sigma = 0
    count = 0

    for n in range(ncell):
        if h[n] != 0:
            logf = np.log(h[n])
        else:
            logf = 0
        count = count + h[n]
        estimate = estimate - h[n] * logf
        sigma = sigma + h[n] * logf ** 2

    nbias = -(float(ncell) - 1) / (2 * count)

    estimate = estimate / count
    estimate = estimate + np.log(count) + np.log((upperbound - lowerbound) / ncell) - nbias
    
    # Scale the entropy estimate to stretch the range
    estimate = np.exp(estimate ** 2) - np.exp(0) - 1
    
    return estimate


def jerkiness(winDat, fs):
    """
    jerk ratio/smoothness according to Mahadevan, npj Park Dis 2018
    uses rate of acc-changes (Hogan 2009). PM was aimed for 3-sec windows
    -> double check function with references
    """
    ampl = np.max(np.abs(winDat))
    jerk = winDat.diff(1) * fs
    jerkSqSum = np.sum(jerk ** 2)
    scale = 360 * ampl ** 2 / len(winDat) / fs
    meanSqJerk = jerkSqSum / fs / (len(winDat) / fs * 2)
    jerkRatio = meanSqJerk / scale
    
    return jerkRatio

