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
from typing import Any
import numpy as np
from dataclasses import dataclass, field

# Import own custom functions
import lfpecog_features.tapping_featureset as tap_feats
from lfpecog_features.tapping_preprocess import find_main_axis

@dataclass(init=True, repr=True, )
class tapFeatures:
    """
    Extract features from detected taps in acc-tapping trace

    Input:
        - triax_arr: 2d-array with tri-axial acc-signal
        - fs (int): sample freq in Hz
        - impacts (array): array containing indices of
            impact (closing finger) moments
        - tapDict: list of taps with lists of 6 taptimes
            resulting from continuous tapping detect function
        - updrsSubScore: UPDRS III Fingertapping subscore
            corresponding to acc signal, default False
        

    """
    triax_arr: Any
    fs: int
    impacts: Any
    tapDict: dict = field(default_factory=dict)
    updrsSubScore: Any = False
    # nTaps: int = field(default_factory=int)
    # tap_durations: Any = None
    # freq1: float = field(default_factory=float)
    # freq2: float = field(default_factory=float)
    # RMS_ax: Any = None
    # RMS_svm: Any = None
    # upVelo_ax: Any = None
    # upVelo_svm: Any = None

    
    # nGoodTaps: int = field(default_factory=int)
    
    # maxVelUp: Any = None  #float = field(default_factory=float)
    # dirChange: Any = None
    
    
    def __post_init__(self,):
        ax = find_main_axis(self.triax_arr)
        self.nTaps = len(self.impacts)
        self.freq = self.nTaps / (
            self.triax_arr.shape[1] / self.fs)
        self.tap_durations = np.diff(self.impacts) / self.fs
        self.freq2 = round(1 / self.tap_durations.mean(), 1)


        self.runRMS_ax, self.runRMS_svm = tap_feats.tap_RMS(
            self.tapDict, self.triax_arr, select='run', ax=ax,
        )
        # ADD RUN VARIATION
        
        self.tapRMS_ax, self.tapRMS_svm = tap_feats.tap_RMS(
            self.tapDict, self.triax_arr, select='tap', ax=ax,
        )
        self.upVelo_ax, self.upVelo_svm = tap_feats.upTap_velocity(
            self.tapDict, self.triax_arr, ax=ax,
        )

        self.impactRMS_ax, self.impactRMS_svm = tap_feats.tap_RMS(
            self.tapDict, self.triax_arr, select='impact', ax=ax,
            impact_window=.25, fs=self.fs,
        )

        if type(self.updrsSubScore) == str or np.str_:
            self.updrsSubScore = float(self.updrsSubScore)

        # # in progress
        # self.nGoodTaps = sum(
        #     [np.nan not in t for t in self.tapDict])
        
        # hesitations
        # # self.dirChange = tap_feats.tapFt_dirChanges(
        # #     self.tapDict, self.triax_arr,)
        
        
        # clear up space
        self.triax_arr = 'cleaned up'
        
