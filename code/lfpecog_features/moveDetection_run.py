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
        - subdat: subData Class with data per patient

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


    return tap_move_lists


