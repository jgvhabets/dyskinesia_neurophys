'''
Connectivity-Feature Extraction
based on two ephys-channels
'''

# Import public packages and functions
from socket import AI_NUMERICHOST
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
class run_segmConnectFts:
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
        data_arr, data_keys, dataWinTimes = utils_win.get_windows(
            self.sub_df,
            fs=self.fs,  
            winLen_sec=self.winLen_sec
        )
        print('got windows')
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
            winLen_sec=self.winLen_sec,
        )
        setattr(self, 'chSegments', chSegments)  # store for notebook use
        # segments here are still py float's, not np.float64
        print('got segments')

        # get list with tuples of all target x seed combis
        self.allCombis = list(product(seeds, targets))

        setattr(
            self,
            'features',
            calculate_segmConnectFts(
                chSegments=self.chSegments,
                fs=self.fs,
                allCombis=self.allCombis,
                fts_to_extract=['COH', 'ICOH', 'absICOH', 'sqCOH']
            )
        )


@dataclass(init=True, repr=True, )
class calculate_segmConnectFts:
    """
    Extract of Connectivity
    features per segments

    """
    chSegments: Any  # attr of prepare_segmConnectFts()
    fs: int
    allCombis: Any   # attr of prepare_segmConnectFts()
    fts_to_extract: list

    def __post_init__(self,):

        coh_fts_incl = [
            ft for ft in self.fts_to_extract
            if 'coh' in ft.lower()
        ]
        if len(coh_fts_incl) > 1:
            extract_COH = True  # create boolean indicator for coherences     
            coh_lists = {}  # create empty lists to store COH's while calculating
            for ft in coh_fts_incl: coh_lists[f'{ft}_list'] = []

        print(f'Coherence fts to include: {coh_fts_incl}')

        for seedTarget in self.allCombis:

            print(
                '\tstarting feature extraction: '
                f'SEED: {seedTarget[0]} x '
                f'TARGET: {seedTarget[1]}'
            )
            # define channel classes containing segmented 3d-data and times
            seed = getattr(self.chSegments, seedTarget[0])
            target = getattr(self.chSegments, seedTarget[1])
            # reshape 3d segments to 2d, and internally check whether
            # number of segments and timestamps are equal
            seed2d = get_clean2d(seed.data, seed.segmTimes)
            target2d = get_clean2d(target.data, target.segmTimes)
            # IMPORTANT: 2d arrays returned as NP.FLOAT64's, not PY FLOAT's
            print(seed.data.shape, target.data.shape)
            print(seed2d.shape, target2d.shape)
            # reset storing lists
            time_list = []
            if extract_COH:
                for l in coh_lists: coh_lists[l] = []

            # loop over times and find matching indices
            for i_seed, t in enumerate(seed.segmTimes):

                if t in target.segmTimes:

                    i_target = np.where(target.segmTimes == t)[0][0]

                    time_list.append(t)

                    # COHERENCE EXTRACTION
                    if len(coh_fts_incl) > 0:
                        f_coh, icoh, icoh_abs, coh, sq_coh = specFeats.calc_coherence(
                            sig1=seed2d[i_seed, :],
                            sig2=target2d[i_target],
                            fs=self.fs,
                            nperseg=seed2d.shape[1],
                        )
                        coh_lists['absICOH_list'].append(icoh_abs)
                        coh_lists['ICOH_list'].append(icoh)
                        coh_lists['COH_list'].append(coh)
                        coh_lists['sqCOH_list'].append(sq_coh)
                        
            
            # TODO: CONSIDER POST-HOC STANDARDISATION, such as ICOH-detectability


            # STORE RESULTING FEATURES IN CLASSES
            setattr(
                self,
                f'{seedTarget[0]}_{seedTarget[1]}',
                storeSegmFeats(
                    channelName=f'{seedTarget[0]}_{seedTarget[1]}',
                    times_list=time_list,
                    ft_lists=coh_lists,
                    freqs=f_coh,
                    winStartTimes=target.winTimes,
                )
            )

@dataclass(init=True, repr=True,)
class storeSegmFeats:
    channelName: str
    times_list: list
    ft_lists: dict
    freqs: array
    winStartTimes: Any

    def __post_init__(self,):

        # create feature attrbiutes
        for key in self.ft_lists.keys():

            ft = key.split('_')[0]
            setattr(
                self,
                ft,
                np.array(self.ft_lists[f'{ft}_list'])
            )
        
        # create time attributes
        self.segmTimes = np.array(self.times_list)  # all timestamps for segmentfts
        # find corr indices to window moments
        self.winStartIndices = [
            np.argmin(abs(self.segmTimes - t)) for t in self.winStartTimes
        ]



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