'''
Functions to extract spectral from neurophysiology
data (LFP and ECOG) in ReTune's Dyskinesia Project

Creates features for moments classified as [Rest,
Movement, Tap] and annotates the time relative to
L-Dopa intake.
'''
# Import general packages and functions
from dataclasses import dataclass
import os
from re import sub
from typing import Any
import json
import numpy as np
from scipy.signal import welch, cwt, morlet2

# Import own functions
from lfpecog_features.feats_read_proc_data import subjectData
import lfpecog_features.feats_spectral_baseline as baseLine

"""
Include in feature class:
    per window (1 s?, 0.5 s?, .25 sec with >> sample freq):
        - state (nomove, tap (after tap/taprest), move)
        - beta peak
        - gamma peak
        - beta peak bandw
        - peak heigth (Swann '16)
"""

@dataclass(init=True, repr=True,)
class dopaTimed_ftSpace:
    """
    Extract feature-set alligned to dopa-times.

    Input:
        - sub (str): subject-code
        - incl_baseline (bool): set True to include
            baseline values
    """
    sub: str
    data_version: str
    project_path: str
    incl_baseline: bool = False
    
    def __post_init__(self,):
        subData = subjectData(
            sub=self.sub,
            data_version=self.data_version,
            project_path=self.project_path  
        )

        if self.incl_baseline:

            blClass = baseLine.createBaseline(subData)
            setattr(self, 'baseline', blClass)







class EphyBaseData:
    """Create data per ecog/ lfp-L / lfp-R"""
    def __init__(self, runClass, runname, dtype):
        self.dtype = dtype
        base_ind_f = os.path.join(  # make sure projectpath is cwd
            'data/analysis_derivatives/'
            'base_spectral_run_indices.json'
        )
        with open(base_ind_f) as jsonfile:
            base_ind = json.load(jsonfile)
        sub = runClass.sub
        ses = runClass.ses
        base_ind = base_ind[sub][ses][dtype]

        for row, level in enumerate(
            getattr(runClass, f'{dtype}_names')
        ):
            # iterate all levels, skip first time-row
            if np.logical_and(row == 0, level == 'time'):
                continue
            setattr(
                self,
                level,  # key of attr
                EphyBaseLevel(
                    runClass,
                    dtype,
                    level,
                    row,
                    base_ind
                )
            )
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__} Class '
            f'for {self.dtype}')


class EphyBase():
    '''Baseline creation for spectral analyses'''
    def __init__(self, runClass, runname: str):

        self.runClass = runClass.runs[runname]
        self.runname = runname
        self.ecog = EphyBaseData(
            runClass=self.runClass,
            runname=self.runname,
            dtype='ecog',
        )
        self.lfp_left = EphyBaseData(
            runClass=self.runClass,
            runname=self.runname,
            dtype='lfp_left',
        )
        self.lfp_right = EphyBaseData(
            runClass=self.runClass,
            runname=self.runname,
            dtype='lfp_right',
        )
    
    def __repr__(self):
        return (f'{self.__class__.__name__}: '
            f'Main EphysBaseline Class')

