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

@dataclass(init=True, repr=True, )
class tapFeatures:
    """
    Extract features from detected taps in acc-tapping trace

    tapDict: dict with taptimes
    accSig: 1d-array with most relevant acc-axis
    """
    accSig: Any
    fs: int
    tapDict: dict = field(default_factory=dict)
    nTaps: int = field(default_factory=int)
    nGoodTaps: int = field(default_factory=int)
    duration: Any = None
    freq: float = field(default_factory=float)
    maxVelUp: Any = None  #float = field(default_factory=float)
    dirChange: Any = None
    RMS: Any = None
    
    def __post_init__(self,):
        try:
            if len(self.tapDict) > 0:
                self.nTaps = len(self.tapDict)
                self.nGoodTaps = sum(
                    [np.nan not in t for t in self.tapDict])
                self.duration = tap_feats.tapFt_duration(
                    self.tapDict, self.fs)
                self.freq = round(1 / self.duration.mean(), 1)
                self.maxVelUp = tap_feats.tapFt_maxVeloUpwards(
                    self.tapDict, self.accSig, self.fs)
                self.dirChange = tap_feats.tapFt_dirChanges(
                    self.tapDict, self.accSig,)
                self.RMS = tap_feats.tapFt_RMS(
                    self.tapDict, self.accSig,)
        
        except TypeError:  # without pre-detected taps
