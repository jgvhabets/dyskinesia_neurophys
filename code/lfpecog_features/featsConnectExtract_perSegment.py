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
class run_segmConnectFts:
    """
    Prepare the Extraction of Connectivity
    features per segments

    TODO: MAKE THIS MAIN FEATURE EXTRACTION FUNCTION
        - INCLUDE UNI-SIGNAL, NOT-CONNECTIVITY FEATURES IN HERE
        - CREATE OPTIONAL attributes for features and conn_features
    TODO: ADD PROPER FEATURE STORING FUNCTION

    Default settings translate to a feature resolution
    of 2 Hz (e.g. 60 sec windows, with 50% overlap
    lead to a coherency-value every 30 seconds).
    """
    sub: str
    sub_df: Any
    fs: int
    winLen_sec: float = 60
    part_winOverlap: float = .5
    segLen_sec: float = .5
    part_segmOverlap: float = .5
    movement_part_acceptance: float = 1
    channels_incl: list = field(default_factory=list)
    channels_excl: list = field(default_factory=list)
    
    
    def __post_init__(self,):
        # divides full dataframe in present windows
        windows = utils_win.get_windows(
            self.sub_df,
            fs=self.fs,  
            winLen_sec=self.winLen_sec,
            part_winOverlap=self.part_winOverlap,
            return_as_class = True,  # returns data/keys/times as class
            movement_part_acceptance=self.movement_part_acceptance,
        )
        print('got windows')
        

        # select all ECOG-channels as targets
        targets = [ch for ch in windows.keys if 'ECOG' in ch]
        # select all ipsilateral STN-channels as seeds
        ecogside = targets[0][5]
        seeds = [
            ch for ch in windows.keys if
            f'LFP_{ecogside}' in ch
        ]

        # create class with 3d-segmented-data and times, per channel
        epochedChannels = utils_win.epochedData_multipleChannels(
            windows=windows,
            channels_incl=targets+seeds,
            fs=self.fs,
            winLen_sec=self.winLen_sec,
            segLen_sec=self.segLen_sec,
            part_segmOverlap=self.part_segmOverlap,
        )
        # store for notebook use
        setattr(self, 'channels', epochedChannels)

        print('got epoched data')
        # TODO: FUTURE SAVE EPOCHED CHANNEL DATA

        # get list with tuples of all target x seed combis
        self.allCombis = list(product(seeds, targets))

        setattr(
            self,
            'features',
            calculate_segmConnectFts(
                epochChannels=self.channels,
                fs=self.fs,
                allCombis=self.allCombis,
                fts_to_extract=[
                    'COH', 'ICOH', 'absICOH', 'sqCOH',
                ],
            )
        )


@dataclass(init=True, repr=True, )
class calculate_segmConnectFts:
    """
    Extract of Connectivity
    features per segments

    """
    epochChannels: Any  # attr of prepare_segmConnectFts()
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
            # get epoched 2d/3d-data and times for channels of interest
            seed = getattr(self.epochChannels, seedTarget[0])
            target = getattr(self.epochChannels, seedTarget[1])
            # reshape 3d segments to 2d, and internally check whether
            # number of segments and timestamps are equal
            if len(seed.data.shape) == 3:
                setattr(
                    seed,
                    'data',
                    get_clean2d(seed.data, seed.segmTimes)
                )
            if len(target.data.shape) == 3:
                setattr(
                    target,
                    'data',
                    get_clean2d(target.data, target.segmTimes)
                )

            # reset storing lists
            time_list = []
            if extract_COH:
                for l in coh_lists: coh_lists[l] = []

            # find time-matching target-indices for seed-indices
            for i_seed, t in enumerate(seed.winTimes):

                if t in target.winTimes:

                    i_target = np.where(target.winTimes == t)[0][0]

                    time_list.append(t)

                    # COHERENCE EXTRACTION
                    if len(coh_fts_incl) > 0:

                        f_coh, icoh, icoh_abs, coh, sq_coh = specFeats.calc_coherence(
                            sig1=seed.data[i_seed, :],
                            sig2=target.data[i_target, :],
                            fs=self.fs,
                            nperseg=int(
                                self.epochChannels.segLen_sec * self.fs
                            ),
                        )
                        coh_lists['absICOH_list'].append(icoh_abs)
                        coh_lists['ICOH_list'].append(icoh)
                        coh_lists['COH_list'].append(coh)
                        coh_lists['sqCOH_list'].append(sq_coh)


            # STORE RESULTING FEATURES IN CLASSES
            setattr(
                self,
                f'{seedTarget[0]}_{seedTarget[1]}',
                storeSegmFeats(
                    channelName=f'{seedTarget[0]}_{seedTarget[1]}',
                    epoch_times=np.array(time_list),
                    ft_lists=coh_lists,
                    freqs=f_coh,
                    winStartTimes=target.winTimes,
                )
            )

@dataclass(init=True, repr=True,)
class storeSegmFeats:
    channelName: str
    epoch_times: array
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


