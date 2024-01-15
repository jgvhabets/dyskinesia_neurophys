"""
script to run and test ongoing scripts/work

winOS: cwd (repo/code): python dev_runs.py
"""

from lfpecog_analysis.psd_analysis_classes import (
    get_selectedEphys
)

if __name__ == '__main__':

    print('run get_selectedEphys()')
    ALLSUBS = get_selectedEphys(STATE_SEL='freenomove_nolid',
                                RETURN_PSD_1sec=True,
                                LOAD_PICKLE=True,
                                USE_EXT_HD=True,
                                PREVENT_NEW_CREATION=False,
                                FORCE_CALC_PSDs=False,
                                verbose=True,
                                # SKIP_BASELINE=True,
                                # SKIP_DYSKMOVE=True,
                                # SKIP_REST=True,
                                # SKIP_TAP=True,
                                SKIP_FREE=False,)