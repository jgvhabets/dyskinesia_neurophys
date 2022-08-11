'''Run Finger- and HandTap Detection functions'''

# Import public packages and functions
import numpy as np
from itertools import compress

# Import own functions
import lfpecog_features.moveDetection_pausedTapFinder as tapFind


def runTapDetection(
    subdat
):
    """
    ** USE FOR PAUSED TAPPING - MOVEMENT STATE DETECTION **


    Function for execution of (bilateral) tap-epoch
    detection. Detects for each side epochs of rest vs tap
    vs other movement. Then combines two sides to epochs
    representing bilateral rest, left-tap, right-tap,
    left-nontap-movement, right-non-tap-movement.

    Input:
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
    
    tap_move_lists = tapFind.pausedTapDetector(subdat)

    # restDict, tapDict, moveDict = {}, {}, {}
    # tapDict, moveDict, restDict = pausedTapDetector(
    #     subdat
    # )
    # restDict = mergeBilatDict(restDict)
            
    
    # return restDict, tapDict, moveDict

    return tap_move_lists



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

    