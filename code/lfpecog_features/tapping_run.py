'''Run Finger- and HandTap Detection functions'''

# Import public packages and functions
import numpy as np
from itertools import compress

# Import own functions
from lfpecog_features.tapping_preprocess import run_preproc_acc, find_main_axis
from lfpecog_features.handTap_detect2 import pausedTapDetector, continTapDetector


def run_updrs_tapping(
    acc_arr, fs: int, already_preprocd: bool=True
):
    """
    Input:
        - acc_arr (array): tri-axial acc array
        - fs (int): sampling freq in Hz
    """
    if already_preprocd == False:
        axes, main_ax_i = run_preproc_acc(
            dat_arr=np.array(x, y, z),
            fs=fs,
            to_detrend=True,
            to_check_magnOrder=True,
            to_check_polarity=True,
        )
    else:
        main_ax_i = find_main_axis(acc_arr)
        
    tapInd, tapTimes, endPeaks = continTapDetector(
        acc_triax=acc_arr, fs=fs, main_ax_i=main_ax_i
    )

    return



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
        - task (str): paused or continuous -> paused / cont
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
        
        elif task == 'cont':
            print('start continuous tapping detection')
            tapInd, tapTimes, endPeaks = continTapDetector(
                fs=fs, x=axes[0, :], y=axes[1, :], z=axes[2, :],
                side=s,
            )

            return tapInd, tapTimes, endPeaks



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

    