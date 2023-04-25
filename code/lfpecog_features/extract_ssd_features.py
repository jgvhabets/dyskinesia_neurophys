'''
Connectivity-Feature Extraction
based on two ephys-channels
'''

# Import public packages and functions
import json
from typing import Any
import numpy as np
from os.path import join, exists
from os import listdir, makedirs
from pandas import isna, DataFrame
from scipy.signal import welch
from scipy.stats import variation
from array import array
from dataclasses import dataclass, field
from itertools import product, compress
import matplotlib.pyplot as plt


# import own functions
from utils.utils_fileManagement import (
    get_project_path, get_onedrive_path,
    load_class_pickle, save_class_pickle,
    load_ft_ext_cfg
)
from lfpecog_preproc.preproc_import_scores_annotations import get_ecog_side

from utils.utils_windowing import get_windows, windowedData
import lfpecog_features.feats_spectral_features as specFeats




@dataclass(init=True, repr=True, )
class extract_local_SSD_powers():
    """
    Extract local power spectral features from
    SSD-d data
    """
    sub: str
    sub_SSD: str
    settings_dict: dict
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.settings_dict    
        # define ephys_sources
        self.ephys_sources = []
        if self.incl_ecog:
            ecog_side = get_ecog_side(self.sub)
            self.ephys_sources.append(f'ecog_{ecog_side}')
        if self.incl_stn:
            self.ephys_sources.extend(['lfp_left', 'lfp_right'])

        ### Define paths
        feat_path = join(get_project_path('results'),
                         'features',
                         'SSD_powers_TEST',
                         f'windows_{SETTINGS["WIN_LEN_sec"]}s_'
                         f'{SETTINGS["WIN_OVERLAP_part"]}overlap')
        if not exists(feat_path): makedirs(feat_path)
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD powers')
            data = getattr(self.sub_SSD, dType)  # assign datatype data
            # check if features already exist
            feat_fname = f'SSD_spectralFeatures_{self.sub}_{dType}.csv'

            if exists(join(feat_path, feat_fname)) and not self.overwrite_features:
                print(f'...\tfeatures already exist for {dType}'
                      ' and are NOT overwritten, skip!')

            ### Create storing of features
            fts_incl = SETTINGS['FEATS_INCL']  # get features to include
            feats_out = []  # list to store lists per window
            feat_names = []  # feature names
            for band in SETTINGS['SPECTRAL_BANDS'].keys():
                # NOTE: order has to be identical of order of adding features
                n_spec_feats = 0
                for ft_name in fts_incl:
                    # to feats_out (via temporarily list feats_win)
                    if fts_incl[ft_name] and ft_name.startswith('SSD_'):
                        feat_names.append(f'{band}_{ft_name[4:]}')
                        n_spec_feats += 1
            
            # loop over windows
            for i_w, t in enumerate(data.times):
                # temporary list to store features of current window
                feats_win = []
                
                # loop over defined frequency bands
                for bw in SETTINGS['SPECTRAL_BANDS']:
                    f_range = SETTINGS['SPECTRAL_BANDS'][bw]
                    
                    # check for unconverted nan array
                    if np.isnan(list(getattr(data, bw)[i_w])).all():
                        # fill with nan values for feature and go to next band
                        feats_win.extend([np.nan] * n_spec_feats)
                        continue

                    # Convert SSD'd signal into Power Spectrum
                    f, psd = welch(getattr(data, bw)[i_w], axis=-1,
                                    nperseg=data.fs, fs=data.fs)
                    f_sel = [f_range[0] < freq < f_range[1] for freq in f]  # select psd in freq of interest

                    # CALCULATE SPECTRAL PEAK FEATURES
                    # loop over ft-names and ft-funcs ensures correct ft-order
                    for ft_name in fts_incl:
                        if fts_incl[ft_name] and ft_name.startswith('SSD_'):
                            # get value from function corresponding to ft-name
                            ft_func = getattr(specFeats.Spectralfunctions(
                                psd=psd, ssd_sig=getattr(data, bw)[i_w],
                                f=f, f_sel=f_sel
                            ), f'get_{ft_name}')
                            feats_win.append(ft_func())
                       
                # END OF WINDOW -> STORE list with window features to total list
                feats_out.append(feats_win)

            # AFTER ALL WINDOWS OF DATA TYPE ARE DONE -> STORE FEATURE DATAFRAME
            feats_out = np.array(feats_out, dtype='object')
            feats_out = DataFrame(data=feats_out, columns=feat_names, index=data.times,)
            feats_out.to_csv(join(feat_path, feat_fname),
                             index=True, header=True,)
            print(f'Saved FEATURES for sub-{self.sub} {dType} as '
                  f'{feat_fname} in {feat_path}')


from lfpecog_features.feats_phases import calculate_PAC_matrix

@dataclass(init=True, repr=True,)
class extract_local_PACs:
    """
    Extracting local PAC based on SSD'd freq-bands
    stored per window

    Called from run_ssd_ftExtr
    """
    sub: str
    sub_SSD: str
    settings_dict: dict
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.settings_dict    
        # define ephys_sources
        self.ephys_sources = []
        if self.incl_ecog:
            ecog_side = get_ecog_side(self.sub)
            self.ephys_sources.append(f'ecog_{ecog_side}')
        if self.incl_stn:
            self.ephys_sources.extend(['lfp_left', 'lfp_right'])

        ### Define paths
        feat_path = join(get_project_path('results'),
                         'features',
                         'SSD_powers_TEST',
                         f'windows_{SETTINGS["WIN_LEN_sec"]}s_'
                         f'{SETTINGS["WIN_OVERLAP_part"]}overlap')
        if not exists(feat_path): makedirs(feat_path)
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD PACs')
            data = getattr(self.sub_SSD, dType)  # assign datatype data
            # check if features already exist
            feat_fname = f'SSD_localPAC_{self.sub}_{dType}.csv'

            if exists(join(feat_path, feat_fname)) and not self.overwrite_features:
                print(f'...\tfeatures already exist for {dType}'
                      ' and are NOT overwritten, skip!')
                continue

            ### Create storing of features
            pac_bands = SETTINGS['FEATS_INCL']['local_PAC']  # get features to include
            feats_out = []  # list to store lists per window
            feat_names = []  # feature names

            # load SSD'd data
            for [pha, ampl] in pac_bands:
                print(pha, ampl)
                # use tensorpac functions in notebook phase_feats py
                # calculate_PAC_matrix()





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


