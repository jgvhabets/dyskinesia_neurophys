'''
Connectivity-Feature Extraction
based on two ephys-channels
'''

# Import public packages and functions
import json
from typing import Any
import numpy as np
from os.path import join
from os import listdir
from scipy.signal import welch
from array import array
from dataclasses import dataclass, field
from itertools import product, compress
import matplotlib.pyplot as plt


# import own functions
from utils.utils_fileManagement import get_project_path, get_onedrive_path, load_class_pickle
from utils.utils_windowing import get_windows
import lfpecog_features.feats_spectral_features as specFeats
from lfpecog_features import feats_SSD as ssd

@dataclass(init=True, repr=True, )
class extract_multivar_features:
    """
    MAIN FEATURE EXTRACTION FUNCTION

    Run from run_ftExtr_multivar: python -m lfpecog_features.run_ftExtr_multivar 'ftExtr_spectral_v1.json'
    """
    sub: str
    settings: dict = field(default_factory=lambda:{})
    feat_setting_filename: str = None
    ephys_sources: list = field(
        default_factory=lambda: ['lfp_left', 'lfp_right',
                                 'ecog_left', 'ecog_right']
    )
    
    def __post_init__(self,):

        assert np.logical_or(
            isinstance(self.settings, dict),
            isinstance(self.feat_setting_filename, str)
        ), 'define settings-dict or setting-json-filename'

        ### load settings from json
        if not isinstance(self.settings, dict):
            json_path = join(get_onedrive_path('data'),
                            'featureExtraction_jsons',
                            self.feat_setting_filename)
            with open(json_path, 'r') as json_data:
                SETTINGS =  json.load(json_data)
        else:
            SETTINGS = self.settings

        ### Load Data
        mergedData_path = join(get_project_path('data'),
                               'merged_sub_data',
                                SETTINGS['DATA_VERSION'],
                                f'sub-{self.sub}')
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            dat_fname = (f'{self.sub}_mergedData_{SETTINGS["DATA_VERSION"]}'
                        f'_{dType}.P')
            # check existence of file in folder
            if dat_fname not in listdir(mergedData_path):
                print(f'{dat_fname} NOT AVAILABLE')
                continue
            # load data (as mergedData class)
            data = load_class_pickle(join(mergedData_path, dat_fname))
            print(f'{dat_fname} loaded')
              
            # divides full dataframe in present windows
            windows = get_windows(
                data=data.data,
                fs=int(data.fs),
                col_names=data.colnames,
                winLen_sec=SETTINGS['WIN_LEN_sec'],
                part_winOverlap=SETTINGS['WIN_OVERLAP_part'],
                min_winPart_present=.5,
                remove_nan_timerows=False,
                return_as_class=True,
                only_ipsiECoG_STN=False,
            )
            # filter out none-ephys signals
            sel_chs = [c.startswith('LFP') or c.startswith('ECOG')
                       for c in windows.keys]
            print(f'GOT WINDOWS {dType}, shape: {windows.data.shape}, '
                    f'colnames: {windows.keys}')
            setattr(windows, 'data', windows.data[:, :, sel_chs])
            setattr(windows, 'keys', list(compress(windows.keys, sel_chs)))
            print(f'\tWINDOWS {dType} ONLY EPHYS, shape: {windows.data.shape}, '
                    f'colnames: {windows.keys}')


            # loop over windows
            for i_w, win_dat in enumerate(windows.data[:5]):
                
                # select only rows without missing
                nan_rows = [pd.isna(win_dat.data[:, i]).any()
                            for i in range(win_dat.data.shape[-1])]
                win_dat = win_dat[:, ~nan_rows]
                win_chnames = list(compress(windows.keys, ~nan_rows))
                win_time = windows.win_starttimes[i_w]
                
                # loop over defined frequency bands
                for bw in SETTINGS['SPECTRAL_BANDS']:
                    f_range = SETTINGS['SPECTRAL_BANDS'][bw]
                    # check whether to perform SSD
                    if SETTINGS['FEATS_INCL']['SSD']:
                        (ssd_filt_data,
                            ssd_pattern,
                            ssd_eigvals
                        ) = ssd.get_SSD_component(
                            data_2d=epoch_dat,
                            fband_interest=f_range,
                            s_rate=windows.fs,
                            use_freqBand_filtered=True,
                            return_comp_n=0,
                        )
                        f, psd = welch(ssd_filt_data, axis=-1,
                                              nperseg=data.fs, fs=data.fs)
                        plt.plot(f, psd, label=bw)
                plt.xlim(0, 100)
                plt.title(f'WINDOW # {i_w} - {dType.upper()}')
                plt.legend()
                plt.show()

        



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


