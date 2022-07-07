'''
Functions to define tapping features
'''
# Import general packages and functions
import numpy as np


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

def tapFt_duration(tapDict, fs):
    out = np.zeros(len(tapDict))
    for n, tap in enumerate(tapDict):
        l = (tap[-1] - tap[0]) / fs
        out[n] = l
    
    return out


def tapFt_dirChanges(tapDict, accSig,):
    """
    Calculates how often differential
    of acc-signal crossed the zero-line
    per tap.
    Uses a smoothened line of np.diff
    """
    out = np.zeros(len(tapDict))

    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size

    for n, tap in enumerate(tapDict):
        tapSig = accSig[tap[0]:tap[-1]]
        dfSm = np.convolve(
            np.diff(tapSig), kernel, mode='same')
        count = 0
        for i, df in enumerate(dfSm[1:]):
            if df * dfSm[i] < 0: count += 1

        out[n] = count
        
    return out


def tapFt_RMS(tapDict, accSig,):
    """
    Calculates RMS of acc-signal within
    each tap.
    """
    out = np.zeros(len(tapDict))

    for n, tap in enumerate(tapDict):
        tapSig = accSig[tap[0]:tap[-1]]
        S = np.square(tapSig)
        MS = S.mean()
        RMS = np.sqrt(MS)
        out[n] = RMS
        
    return out


def tapFt_maxVeloUpwards(tapDict, accSig, fs):
    """
    Calculates max velocity during finger-raising
    based on the AUC from the first big pos peak
    in one tap until the acceleration drops below 0

    Input:
        - tapDict (dict)
        - accSig (array)
        - fs (int)
    
    Returns:
        - out (array): one value or nan per tap in tapDict
    """
    out = np.zeros(len(tapDict))

    for n, tap in enumerate(tapDict):
        if tap[1] is not np.nan:  # crossing 0 has to be known
            # take acc-signal [start : fastest point] of rise
            line = accSig[tap[0]:tap[1]]
            areas = []
            for s, y in enumerate(line[1:]):
                areas.append(np.mean([y, line[s]]))
            out[n] = abs(sum(areas))

        else:  # if crossing-0 point unknown
            out[n] = np.nan
    
    return out



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

