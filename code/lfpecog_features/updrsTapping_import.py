"""
Functions to import tapping-traces from UPDRS-tapping
or equal tapping-tasks with (blocks of) continuous tapping.
"""

# Import external functions and packages
import os
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import resample
from scipy.io import loadmat
from itertools import compress
import h5py
from pandas import read_csv

# Import own functions
import lfpecog_features.tapping_preprocess as preprocess


@dataclass(init=True, repr=True, )
class accData:
    """
    Store On/Off-Acc trace per patient in 1 class
    uses .txt files, one file per on- / off-state.

    Input:
        - orig_fs (int): original sample freq (Hz)
        - wanted_fs (int): sample freq to resample to (Hz)
        - OnFile / OffFile (str): directory of .txt-file
            containing tri-axial acc-data
        - ...
    
    Returns:
        - class containing preprocessed tri-axial
            acc signal in attributes .On and .Off
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
    to_remove_outlier: bool = True
    verbose: bool = True

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
            processed_arr, main_ax_i = preprocess.run_preproc_acc(
                dat_arr=getattr(self, state),
                fs=self.wanted_fs,
                to_detrend=self.to_detrend,
                to_check_magnOrder=self.to_check_magnOrder,
                to_check_polarity=self.to_check_polarity,
                to_remove_outlier=self.to_remove_outlier,
                verbose=self.verbose
            )
            setattr(self, state, processed_arr)



def find_run_IDs(accFiles_dir):
    """
    Function to retrieve automatically available
    updrs tapping traces.
    PM: adjust in case of new data structure.
    """
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
        ### TODO: ADD POSSIBILITT OF 3 MONTHS FOLLOW UP
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


def matlab_import(filepath: str):
    """
    """
    
    try:  # try first (Matlab v until 7.3)
        acc = loadmat(filepath)
    except NotImplementedError:  # for Matlab > v7.3
        acc = h5py.File(filepath, 'a')
    
    return acc


def tap3x10_updrs_scores(
    file_dir:str, file_name:str,
    incl_stimRange: bool,
):
    """
    Imports and categories updrs subscores

    Inputs:
        - filedir: directory of file
        - filename: name of file
        - acc_runids: subjects-side-state
            included in accelerometer import
            and block extraction
        - incl_stimRange: set True if all stim-
            amplitudes in protocol are included
            in subscore file
    """
    scoreTable = read_csv(
        os.path.join(file_dir, file_name)
    )
    IDs = [sub[3:6] for sub in scoreTable['PerceptID']]
    IDs = list(set(IDs))

    if incl_stimRange:

        block_scores = find_stimRangeScores(
            IDs, scoreTable,
        )

        print('Sub-scores for full stim-Ampltiude range extracted')

        return block_scores

    block_scores = {}
    run_IDs = []

    for id in IDs:
        for side in ['L', 'R']:
            for med in ['Off', 'On']:
                run_IDs.append(f'{id}_{side}_{med}')

    for run_id in run_IDs:
        block_scores[run_id] = {'stimOn': [], 'stimOff': []}
        strings = run_id.split(sep='_')
        sub = strings[0]
        side = strings[1]
        med = strings[2]

        for row in range(scoreTable.shape[0]):

            if sub[1:3] in scoreTable['PerceptID'].iloc[row][:6]:
                if scoreTable['Hand'].iloc[row] == side:
                    if scoreTable['Block_N'].iloc[row] in [1, 2, 3]:
                        block_scores[run_id]['stimOn'].append(
                            scoreTable[f'Med{med}_StimOn'].iloc[row])
                        block_scores[run_id]['stimOff'].append(
                            scoreTable[f'Med{med}_StimOff'].iloc[row])
    
    return block_scores


def find_stimRangeScores(
    IDs, scoreTable,
):
    """
    
    """
    block_scores = {}
    run_IDs = []
    subs_incl = [
        '007', '013', '014', '015'
    ]
    stimAmps = [
        '0', '05', '1', '15', '2', '25', '3'
    ]

    for id in IDs:
        if id not in subs_incl: continue

        print(id)
        for side in ['L', 'R']:
            for med in ['Off', 'On']:
                run_IDs.append(f'{id}_{side}_{med}')
    
    for run_id in run_IDs:
        
        block_scores[run_id] = {}
        
        for amp in stimAmps:
            block_scores[run_id][f'{amp}mA'] = []
        
        strings = run_id.split(sep='_')
        sub = strings[0]
        side = strings[1]
        med = strings[2]
    
        for row in range(scoreTable.shape[0]):

            if sub not in scoreTable['PerceptID'].iloc[row][:6]:
                continue

            if scoreTable['Hand'].iloc[row] != side:
                continue
            
            if scoreTable['Block_N'].iloc[row] not in [1, 2, 3]:
                continue

            for amp in stimAmps:

                block_scores[run_id][f'{amp}mA'].append(
                    scoreTable[f'Med{med}_{amp}mA'].iloc[row]
                )

        for amp in stimAmps:

            if sum(np.isnan(
                block_scores[run_id][f'{amp}mA']
            )) == 3:

                del(block_scores[run_id][f'{amp}mA'])
    
    return block_scores
