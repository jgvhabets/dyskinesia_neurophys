"""
Functions to import tapping-traces from UPDRS-tapping
or equal tapping-tasks with (blocks of) continuous tapping.
"""

# Import external functions and packages
import os
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import resample
from itertools import compress

# Import own functions
import lfpecog_features.tapping_preprocess as preprocess


@dataclass(init=True, repr=True, )
class accData:
    """
    Store On/Off-Acc trace per patient in 1 class
    uses .txt files, one file per on- / off-state.
    """
    # give at initiation
    orig_fs: int
    wanted_fs: int
    OnFile: str = field(default_factory=str)
    OffFile: str = field(default_factory=str)
    to_detrend: bool = True
    to_resample: bool = False
    to_check_magnOrder: bool = True
    to_check_polarity: bool = True

    def __post_init__(self,):
        if self.orig_fs != self.wanted_fs:
            self.to_resample = True

        if self.OnFile:
            file = open(self.OnFile, 'rb')
            self.On = np.loadtxt(file, delimiter = ",")
        if self.OffFile:
            file = open(self.OffFile, 'rb')
            self.Off = np.loadtxt(file, delimiter = ",")

        if self.to_resample:
            for state in ['On', 'Off']:
                setattr(self, state, resample(
                    getattr(self, state),
                    (getattr(self, state).shape[0] // (
                        self.orig_fs // self.wanted_fs)),
                ))

        for state in ['On', 'Off']:
            processed_arr = preprocess.run_preproc_acc(
                dat_arr=getattr(self, state),
                fs=self.wanted_fs,
                to_detrend=self.to_detrend,
                to_check_magnOrder=self.to_check_magnOrder,
                to_check_polarity=self.to_check_polarity,
            )
            setattr(self, state, processed_arr)



def create_sub_side_lists(
    accFiles_dir,
):
    sub_folders = os.listdir(accFiles_dir)
    acc_sel = ['Sub' in f for f in sub_folders]
    subs = list(compress(sub_folders, acc_sel))
    sub_dirs = [os.path.join(
        accFiles_dir, s) for s in subs]
    subs = [s[:6] for s in subs]
    sub_sides = []
    sub_side_files = {}

    for sub, sub_dir in zip(subs, sub_dirs):
        sub_files = os.listdir(sub_dir)

        for S in ['L', 'R']:
            if f'{sub}_12mfu_M0_{S}Hand.txt' in sub_files:
                sub_sides.append(f'{sub}_{S}')
                sub_side_files[f'{sub}_{S}'] = {
                    'off': os.path.join(
                        sub_dir, f'{sub}_12mfu_M0_{S}Hand.txt'
                    ),
                    'on': os.path.join(
                        sub_dir, f'{sub}_12mfu_M1_{S}Hand.txt'
                    )
                }


    return sub_sides, sub_side_files


