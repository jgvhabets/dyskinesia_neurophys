"""
Command Line call for MVC-creation and -plotting
for multiple subjects
"""

# import public functions
import sys
# import time
# from os.path import join, exists
# from os import listdir, makedirs, getcwd
# import csv
# from numpy import (save, load, array,
#                    concatenate, argsort)
# from pandas import DataFrame, read_csv

from lfpecog_features.main_run_epochedConnFts import run_mvc_per_sub


if __name__ == '__main__':

    """
    Runs multivariate connectivity computation over
    time (windows, mne-epoching within windowa) per
    subject.

    Give subject-code as first argument,
    give mvc-method (mim or mic) as second arg
    
    Running on WIN (from repo_folder/code):
        (activate conda environment with custom mne_connectivity)
        python -m lfpecog_features.cmdRun_mvc_subs "010" "012" "013"
    """
    for sub in sys.argv[1:]:
        print(sub)
        run_mvc_per_sub(sub)

