'''
Connectivity-Feature Extraction
based on two ephys-channels
'''

# Import public packages and functions
from typing import Any
import numpy as np
import os
from scipy.signal import welch
from array import array
from dataclasses import dataclass, field
from itertools import product

# import own functions
from utils import utils_windowing as utils_win
import lfpecog_features.feats_main as ftsMain
import lfpecog_features.feats_spectral_features as specFeats

@dataclass(init=True, repr=True, )
class prepare_segmConnectFts:
    """
    Prepare the Extraction of Connectivity
    features per segments
    """
    sub: str
    sub_df: Any
    fs: int
    winLen_sec: float
    segLen_sec: float
    part_overlap: float = 0.
    channels_incl: list = field(default_factory=list)
    channels_excl: list = field(default_factory=list)
    
    
    def __post_init__(self,):
        # divide full dataframe in present windows
        data_arr, data_keys, dataWinTimes = ftsMain.get_windows(
            self.sub_df,
            fs=self.fs,  
            winLen_sec=self.winLen_sec
        )
        # make class out of window results
        windows = utils_win.windowedData(
            data_arr, data_keys, dataWinTimes
        )

        # select all ECOG-channels as targets
        targets = [ch for ch in windows.keys if 'ECOG' in ch]
        # select all ipsilateral STN-channels as seeds
        ecogside = targets[0][5]
        seeds = [
            ch for ch in windows.keys if
            f'LFP_{ecogside}' in ch
        ]

        # create class with 3d-segmented-data and times, per channel
        chSegments = utils_win.segmArrays_multipleChannels(
            windows=windows,
            channels_incl=targets+seeds,
            fs=self.fs,
        )
        setattr(self, 'chSegments', chSegments)
        # segments here are still py float's, not np.float64

        # get list with tuples of all target x seed combis
        self.allCombis = list(product(seeds, targets))


@dataclass(init=True, repr=True, )
class extract_segmConnectFts:
    """
    Extract of Connectivity
    features per segments
    """
    chSegments: Any  # attr of prepare_segmConnectFts()
    allCombi: Any   # attr of prepare_segmConnectFts()
    fts_to_extract: list

    def __post_init__(self,):
        for seedTarget in self.allCombis[:1]:

            print(
                f'SEED: {seedTarget[0]} x '
                f'TARGET: {seedTarget[1]}'
            )
            # define channel classes containing segmented 3d-data and times
            seed = getattr(self.chSegments, seedTarget[0])
            target = getattr(self.chSegments, seedTarget[1])
            # reshape 3d segments to 2d, and internally check whether
            # number of segments and timestamps are equal
            seed2d = get_clean2d(seed.data, seed.times)
            target2d = get_clean2d(target.data, target.times)

            # IMPORTANT TO TRANSFORM FLOAT's INTO NP.FLOAT64's

            # loop over times and find matching indices
            for i_seed, t in enumerate(seed.times):

                if t in target.times:

                    i_target = np.where(target.times == t)[0][0]

                    # COHERENCE EXTRACTION
                    f, icoh, coh = specFeats.calc_coherence(
                        sig1=seed2d[i_seed, :],
                        sig2=target2d[i_target],
                        fs=self.fs,
                    )






def get_clean2d(data3d, times=None):

    data2d = data3d.reshape((
        data3d.shape[0] * data3d.shape[1],
        data3d.shape[-1]
    ))

    sel = [
        ~np.isnan(list(data2d[i])).any() for
        i in range(data2d.shape[0])
    ]
    if type(times) == array: 
        assert sum(sel) == times.shape, print(
            'new 2d data is not equal with times'
        )
    # CONVERT TO NP.FLOAT64 FOR NORMAL SPECTRAL-DECOMPOSITION RESULTS
    data2d = data2d[sel].astype(np.float64)
    
    return data2d



# @dataclass(init=True, repr=True, )
# class segmentFts_perContact:
#     """
#     Store segment-features and timings per column/contact
#     """
#     segmPsds: array
#     psdFreqs: array
#     segmTimes: array
#     winStartTimes: list
#     winStartIndices: list
#     # segmentTimes: list
#     contactName: str
#     sub: str