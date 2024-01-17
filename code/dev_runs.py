"""
script to run and test ongoing scripts/work

winOS: cwd (repo/code): python dev_runs.py
"""

from lfpecog_analysis.psd_analysis_classes import (
    get_selectedEphys
)

if __name__ == '__main__':
    """
    Run extraction of state specific data selections
    and PSD calculations
    """
    print('run get_selectedEphys()')
    ALLSUBS = get_selectedEphys(STATE_SEL='freenomove_nolid',  # for creation, not that important, has to be allowed in function
                                RETURN_PSD_1sec=True,  # false: return mean PSDs, true return 1-sec windows
                                EXTRACT_FREE=True,  # FREE is stored in separate folders
                                ADD_FREE_MOVE_SELECTIONS=True,  # add FREE move labels (separately collected and stored)
                                LOAD_PICKLE=True,
                                USE_EXT_HD=True,  # small harddrive JH
                                PREVENT_NEW_CREATION=False,
                                FORCE_CALC_PSDs=False,
                                SKIP_NEW_CREATION=['REST', 'BASELINE',
                                                   'VOLUNTARY', 'INVOLUNTARY'],
                                # SKIP_NEW_CREATION=['FREE'],
                                verbose=True,)