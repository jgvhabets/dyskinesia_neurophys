'''
Functions to extract spectral from neurophysiology
data (LFP and ECOG) in ReTune's Dyskinesia Project

'''
# Import general packages and functions
from dataclasses import dataclass
from mimetypes import init
from multiprocessing.dummy import Array
import os
from typing import Any
import json
import numpy as np
from scipy.signal import welch, cwt, morlet2

@dataclass(init=True, repr=True,)
class extractSpectralFts:
    """ Create base data per level 
    
    CREATE FUNCTION TO EPOCH BASED ON TAPS/MOVES, INCL
        REST EPOCHS (3 S PRE AND POST NON TAP/MOV)
    
    CREATE REUSABLE! FUNCTION FOR SPECTRAL BANDWIDTH

    START WITH EASY STR.FORWARD BANDPOWERS

    COMPARE DESCRIPTIVELY AND MAKE Z-SCORED PLOTS?
        Z-SCORE AGAINST BASELINE SPECTRAL EPOCHS
    """
    runClass: Any=None
    ft_params: list=#default factory
    lowbeta: str=#default factory
    
    def __post_init__(self,):
        sub = self.runClass.sub
        self.lowbeta += 'lowbeta'
    

    print(f'end of class, {lowbeta}')
    



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

