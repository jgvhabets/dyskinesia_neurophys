'''
Functions to define spectral baselines for neuro-
physiology data (LFP and ECOG) in ReTune's Dyskinesia Project

Based on the run that is analyzed, a class is createsd
with baseline raw signals, PSDs (Welch) and wavelet
decomposition based on predefined minutes from the Rest
recording of the corresponding Session. Minutes from Rest
recording are selected to contain no movement.
'''
# Import general packages and functions
import os
from typing import Any
import json
import numpy as np
from scipy.signal import welch, cwt, morlet2

class EphyBaseLevel:
    """ Create base data per level """
    def __init__(
        self,
        runClass,
        dtype,
        level,
        row,
        base_ind
    ):
        self.level = level
        self.rawsig = getattr(runClass, f'{dtype}_arr')[
            row, base_ind[0]:base_ind[1]
        ]
        fs = getattr(runClass, f'{dtype}_Fs')
        self.psd_512 = welch(
            self.rawsig,
            fs=fs,
            nperseg=512, noverlap=256,
        )
        self.psd_256 = welch(
            self.rawsig,
            fs=fs,
            nperseg=256, noverlap=128,
        )
        w = 8  # depth/spaces?
        base_f = np.linspace(1, fs / 2, 100)
        widths = (fs * w) / (2 * base_f * np.pi)
        time = np.arange(len(self.rawsig))
        scp_cfs = cwt(
            self.rawsig, morlet2,
            widths=widths, w=w, dtype='complex128'
        )
        self.wav = {
            'time': time,
            'freq': base_f,
            'psd': np.abs(scp_cfs)
        }
        self.wavlog = {
            'time': time,
            'freq': base_f,
            'psd': np.log10(np.abs(scp_cfs))
        }
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__} Class '
            f'for {self.level}')


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

