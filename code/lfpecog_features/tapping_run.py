'''Run UPDRS FingerTap Detection and Assessment functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import resample
from itertools import compress
from typing import Any

# Import own functions
from lfpecog_features.tapping_time_detect import updrsTapDetector
from lfpecog_features.tapping_preprocess import run_preproc_acc, find_main_axis


def run_updrs_tapping(
    acc_arr, fs: int, already_preprocd: bool=True,
    orig_fs: Any=False,
):
    """
    Input:
        - acc_arr (array): tri-axial acc array
        - fs (int): sampling freq in Hz - in case of
            resampling: this is the wanted fs where the
            data is converted to.
        - already_preproc (bool): if True: preprocessing
            function is called
        - orig_fs: if preprocessing still has to be done,
            and the acc-signal has to be downsampled, the
            original sample frequency has to be given as
            an integer. if no integer is given, no
            resampling is performed.
    
    Returns:
        - tap_ind (list of lists): containing one tap per
            list, and per tap a list of the 6 timepoints
            of tapping detected.
        - impacts: indices of impact-moments (means: moment
            of finger-close on thumb)
        - acc_arr (array): preprocessed data array
    """
    if already_preprocd == False:
        # print('preprocessing raw ACC-data')
        if type(orig_fs) == int:
            # print('resample data array')
            acc_arr = resample(acc_arr,
                acc_arr.shape[0] // (orig_fs // fs))

        acc_arr, main_ax_i = run_preproc_acc(
            dat_arr=acc_arr,
            fs=fs,
            to_detrend=True,
            to_check_magnOrder=True,
            to_check_polarity=True,
            verbose=True
        )
    else:
        main_ax_i = find_main_axis(acc_arr)
        
    tap_ind, impacts = updrsTapDetector(
        acc_triax=acc_arr, fs=fs, main_ax_i=main_ax_i
    )

    return tap_ind, impacts, acc_arr



def runTapDetection(
    task: str, fs: int, 
    leftxyz=[], rightxyz=[],
):
    """
    ** USE FOR PAUSED TAPPING - MOVEMENT STATE DETECTION **


    Function for execution of (bilateral) tap-epoch
    detection. Detects for each side epochs of rest vs tap
    vs other movement. Then combines two sides to epochs
    representing bilateral rest, left-tap, right-tap,
    left-nontap-movement, right-non-tap-movement.
    
    TODO: Consider to differentiate between left-tap and
    right-other movement

    Input:
        - task (str): CHANGE TO ALWAYS PAUSED
                paused or continuous -> paused / cont
        - fs (int): sampling frequency
        - leftxyz/ rightxyz (list): list with x-y-z axes
        arrays; axes not present have to be empty lists

    Output:
        - restDict (dict): merged dictionary of bilateral
            rest. If only one side is given: unilat-rest
        - tapDict (dict): dict with two dict's (one per
            side) containing the timestamps of the tap-
            moments [start-UP, fastest-UP, end-UP,
            start-DOWN, fastest-DOWN, end-DOWN]
        - moveDict (dict): dict containing two dicts
            with non-tap-movement per side
    """
    sides = {}  # check which sides are given
    try:
        if leftxyz: sides['left'] = leftxyz
    except ValueError:
        if leftxyz.size > 0: sides['left'] = leftxyz
    try:
        if rightxyz: sides['right'] = rightxyz
    except ValueError:
        if rightxyz.size > 0: sides['right'] = rightxyz
    if list(sides.keys()) == []:
        return print('No Data inserted for both sides')
    for side in sides.keys():  # check if lists contain 3 axis
        assert len(sides[side]) == 3, (
            f'{side} array-list has to contain 3 array '
            '[x, y, z]; not present axis has to be [].'
        )
    # check whether task is correctly inserted
    assert task in ['paused', 'cont'], 'Unvalid task string'

    if task == 'paused': restDict, tapDict, moveDict = {}, {}, {}
    # if task == 'cont': tapInd, tapTimes, endPeaks = {}, {}, {}

    print('Sides included:', sides.keys())

    for s in sides.keys():
        print(f'\n\nSTART {s} side')
        axes = sides[s]
        if task == 'paused':
            tapDict[s], moveDict[s], restDict[s] = pausedTapDetector(
                fs=fs, x=axes[0], y=axes[1], z=axes[2],
                side=s,
            )
            if len(sides) == 2:
                restDict = mergeBilatDict(restDict, fs)
            
            return restDict, tapDict, moveDict



def mergeBilatDict(epochDict, fs):
    """
    Merges two dictionaries each with lists
    of start- and end-times of behavioral
    epochs.
    This function merges the two dicts and
    creates one dict which contains the epochs
    which are present in both original dicts

    Input:
        epochDict (dict): contains two dicts of
            lists with start and stop-times of
            epochs
        fs (int): sampleing freq in Hz
    
    Returns:
        mergedDict (dict): one dict containing
        lists with start and stop-times of
        mutual times.
    """
    # create one list with all epoch-times per side
    sideLists = {}
    for s in epochDict:
        sideLists[s] = []
        for e in epochDict[s]:
            sideLists[s].extend(np.arange(
                e[0], e[1], 1 / fs
            ))
    for s in sideLists.keys():
        sideLists[s] = np.around(sideLists[s], 3)
    # create list only with times present in both sides
    sel = [t in sideLists['left'] for t in sideLists['right']]
    epochList = list(compress(sideLists['right'], sel))
    # create new merged dict
    mergedDict = {}
    d, start = 0, 0
    for n, hop in enumerate(np.diff(epochList)):
        # loop every time, 'close' epoch when diff > sample freq
        if hop > (1 / fs) + 1e-4:  # add 1e-4 for small time-inaccur
            mergedDict[d] = [start, epochList[n]]
            d += 1
            start = epochList[n + 1]
    
    return mergedDict

    