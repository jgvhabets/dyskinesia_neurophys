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
from lfpecog_features import feats_ssd as ssd



@dataclass(init=True, repr=True, )
class create_SSDs():
    """
    MAIN FEATURE EXTRACTION FUNCTION

    is ran from run_ftExtr_multivar:
    """
    sub: str
    settings: dict = field(default_factory=lambda:{})
    ft_setting_fname: str = None
    incl_ecog: bool = True
    incl_stn: bool = True
    use_stored_windows: bool = True
    save_ssd_windows: bool = True
    
    def __post_init__(self,):

        assert np.logical_or(
            isinstance(self.settings, dict),
            isinstance(self.ft_setting_fname, str)
        ), 'define settings-dict or setting-json-filename'

        ### load settings from json
        if self.settings == {}:
            SETTINGS = load_ft_ext_cfg(self.ft_setting_fname)
        else:
            SETTINGS = self.settings
        
        # define ephys_sources
        self.ephys_sources = []
        if self.incl_ecog:
            ecog_side = get_ecog_side(self.sub)
            self.ephys_sources.append(f'ecog_{ecog_side}')
        if self.incl_stn:
            self.ephys_sources.append(['lfp_left', 'lfp_right'])

        ### Define paths
        mergedData_path = join(get_project_path('data'),
                               'merged_sub_data',
                                SETTINGS['DATA_VERSION'],
                                f'sub-{self.sub}')
        windows_path = join(get_project_path('data'),
                            'windowed_data_classes_'
                            f'{SETTINGS["WIN_LEN_sec"]}s_'
                            f'{SETTINGS["WIN_OVERLAP_part"]}overlap',
                            SETTINGS['DATA_VERSION'],
                            f'sub-{self.sub}')
        # feat_path = join(get_project_path('results'),
        #                  'features',
        #                  'SSD_powers_TEST',
        #                  f'windows_{SETTINGS["WIN_LEN_sec"]}s_'
        #                  f'{SETTINGS["WIN_OVERLAP_part"]}overlap')
        if not exists(windows_path): makedirs(windows_path)
        # if not exists(feat_path): makedirs(feat_path)
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart {dType}')
            # print('ft path listdir', listdir(feat_path))
            # check if features already exist
            # feat_filename = f'SSD_spectralFeatures_{self.sub}_{dType}.csv'
            ssd_windows_name = f'SSD_windowedBands_{self.sub}_{dType}'

            if (
                # if both features and ssd windowed data present
                SETTINGS['OVERWRITE_FEATURES'] == False and
                exists(join(windows_path, ssd_windows_name+'.json')) and
                exists(join(windows_path, ssd_windows_name+'.npy'))
            ):  
                # CREATE ATTRIBUTE CONTAINING WINDOWED SSD BANDS
                setattr(self,
                        dType,
                        ssd.SSD_bands_windowed(self.sub, dType, SETTINGS))
                print(f'\n\texisting windowed ssd-data loaded')
                continue

            print('create windowed ssd data...')

            # define path for windows of dType
            dType_fname = (f'sub-{self.sub}_windows_'
                           f'{SETTINGS["WIN_LEN_sec"]}s_'
                           f'{SETTINGS["DATA_VERSION"]}_{dType}.P')
            dType_win_path = join(windows_path, dType_fname)
            
            # check if windows are already available
            if np.logical_and(self.use_stored_windows,
                              exists(dType_win_path)):
                print(f'load data from {windows_path}....')
                windows = load_class_pickle(dType_win_path)
                print(f'\tWINDOWS LOADED from {dType_fname} in {windows_path}')


            else:
                print('create data ....')
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
                save_class_pickle(windows, windows_path, dType_fname)
                print(f'\tWINDOWS SAVED as {dType_fname} in {windows_path}')


            # filter out none-ephys signals
            sel_chs = [c.startswith('LFP') or c.startswith('ECOG')
                       for c in windows.keys]
            setattr(windows, 'data', windows.data[:, :, sel_chs])
            setattr(windows, 'keys', list(compress(windows.keys, sel_chs)))
            
            ### empty list to store all SSD-bands per window
            if self.save_ssd_windows: ssd_arr_3d = []

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
            for i_w, win_dat in enumerate(windows.data):

                feats_win = []  # temporary list to store features of current window
                win_dat = win_dat.astype(np.float64)
                # select only rows without missings in current window
                nan_rows = np.array([isna(win_dat[:, i]).any()
                            for i in range(win_dat.shape[-1])])
                win_dat = win_dat[:, ~nan_rows]

                # empty list to store ssd-signals for this window
                if self.save_ssd_windows: ssd_arr_2d = []
                
                ### CALCULATE UNI-SOURCE FEATURES ###

                # loop over defined frequency bands
                for bw in SETTINGS['SPECTRAL_BANDS']:
                    f_range = SETTINGS['SPECTRAL_BANDS'][bw]
                    # check whether to perform SSD
                    if not (fts_incl['SSD_max_psd'] or fts_incl['SSD_mean_psd']
                        or fts_incl['SSD_variation']): continue

                    # Perform SSD
                    try:
                        (ssd_filt_data,
                            ssd_pattern,
                            ssd_eigvals
                        ) = ssd.get_SSD_component(
                            data_2d=win_dat.T,
                            fband_interest=f_range,
                            s_rate=windows.fs,
                            use_freqBand_filtered=True,
                            return_comp_n=0,
                        )
                    except ValueError:
                        print(f'{dType}: window # {i_w} failed on {bw}')
                        feats_win.extend([np.nan] * n_spec_feats)
                        if self.save_ssd_windows:
                            ssd_arr_2d.append([np.nan] * win_dat.shape[0])
                        continue

                    if self.save_ssd_windows:
                        ssd_arr_2d.append(ssd_filt_data)

                    # Convert SSD'd signal into Power Spectrum
                    f, psd = welch(ssd_filt_data, axis=-1,
                                    nperseg=windows.fs, fs=windows.fs)
                    f_sel = [f_range[0] < freq < f_range[1] for freq in f]  # select psd in freq of interest

                    # CALCULATE SPECTRAL PEAK FEATURES
                    # loop over ft-names and ft-funcs ensures correct ft-order
                    for ft_name in fts_incl:
                        if fts_incl[ft_name] and ft_name.startswith('SSD_'):
                            # get value from function corresponding to ft-name
                            func = getattr(specFeats.Spectralfunctions(
                                psd=psd, ssd_sig=ssd_filt_data,
                                f=f, f_sel=f_sel
                            ), f'get_{ft_name}')
                            feats_win.append(func())
                       
                # END OF WINDOW -> STORE list with window features to total list
                feats_out.append(feats_win)

                # add 2d of window to overall list
                if self.save_ssd_windows:
                    ssd_arr_2d = np.array(ssd_arr_2d, dtype='object')  # convert list to 2d array
                    ssd_arr_3d.append(ssd_arr_2d)
            
            # AFTER ALL WINDOWS OF DATA TYPE ARE DONE -> STORE FEATURE DATAFRAME
            feats_out = np.array(feats_out, dytpe='object')
            feats_out = DataFrame(data=feats_out, columns=feat_names, index=windows.win_starttimes,)
            feats_out.to_csv(join(feat_path, feat_filename),
                             index=True, header=True,)
            print(f'Saved FEATURES for sub-{self.sub} {dType} as '
                  f'{feat_filename} in {feat_path}')

            if self.save_ssd_windows:
                # save SSD-bands per window as .npy and meta-data as .json
                ssd_arr_3d = np.array(ssd_arr_3d, dtype='object')  # convert array-list to 3d array
                with open(join(windows_path, ssd_windows_name+'.npy'), mode='wb') as f:
                    np.save(f, ssd_arr_3d)
                # store meta info
                assert np.logical_and(
                    ssd_arr_3d.shape[0] == len(windows.win_starttimes),
                    ssd_arr_3d.shape[1] == len(SETTINGS['SPECTRAL_BANDS'])
                ), (f'ssd_arr_3d shape ({ssd_arr_3d.shape}) does not match '
                    f'n-window-times ({len(windows.win_starttimes)}) or '
                    f"n-freqbands ({len(SETTINGS['SPECTRAL_BANDS'])})")
                
                meta = {'npy_filename': ssd_windows_name,
                        'bandwidths': SETTINGS['SPECTRAL_BANDS'],
                        'timestamps': windows.win_starttimes}
                with open(join(windows_path, ssd_windows_name+'.json'), 'w') as f:
                    json.dump(meta, f)
                print(f'Saved SSD windows for sub-{self.sub} {dType} as '
                  f'{feat_filename} in {feat_path}')
            
            # store created windowed ssd-timeseries as attr
            setattr(self, dType, )

@dataclass(init=True, repr=True,)
class extract_local_connectivitiy_fts:
    """
    Extracting local PAC based on SSD'd freq-bands
    stored per window

    Run from run_ftExtr_multivar
    """
    sub: str
    settings: dict = field(default_factory=lambda:{})
    feat_setting_filename: str = None
    
    
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
        
        # define ephys_sources
        ecog_side = get_ecog_side(self.sub)
        self.ephys_sources = [f'ecog_{ecog_side}', 'lfp_left', 'lfp_right']

        ### Define paths
        data_path = join(get_project_path('data'),
                            'windowed_data_classes_'
                            f'{SETTINGS["WIN_LEN_sec"]}s_'
                            f'{SETTINGS["WIN_OVERLAP_part"]}overlap',
                            SETTINGS['DATA_VERSION'],
                            f'sub-{self.sub}')
        feat_path = join(get_project_path('results'),
                         'features',
                         'SSD_powers_TEST',
                         f'windows_{SETTINGS["WIN_LEN_sec"]}s_'
                         f'{SETTINGS["WIN_OVERLAP_part"]}overlap')
        if not exists(feat_path): makedirs(feat_path)
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart {dType}')
            # check is windowed SSDs bands exist
            ssd_data_fname = f'SSD_windowedBands_{self.sub}_{dType}.npy'
            assert exists(join(data_path, ssd_data_fname)), (
                f"SSD'd bands missing for {ssd_data_fname} in {data_path}"
            )
            # check if features already exist
            feat_fname = f'SSD_localPAC_{self.sub}_{dType}.npy'
            if np.logical_and(feat_fname in listdir(feat_path),
                              SETTINGS['OVERWRITE_FEATURES'] == False):
                print(f'\n\tFEATURES ALREADY EXIST and are not overwritten'
                      f' ({feat_fname} in {feat_path})')
                continue
            # get defined local PAC combinations (1st is phase, 2nd is amplitude)
            local_pacs = SETTINGS['local_PAC']
            # load SSD'd data
            for [pha, ampl] in local_pacs:
                print(pha, ampl)
                #### TODO. CONTINUE
                # load
                # use tensorpac functions in notebook -> py script
            





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


