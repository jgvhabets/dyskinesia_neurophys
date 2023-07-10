"""
Function to perform SSD
(Spatio-Spectral-Decomposition) of
timeseries

Based on:
- https://github.com/neurophysics/meet
- Waterstraat G, Curio G, Nikulin VV.
    On optimal spatial filtering for the detection of phase coupling in multivariate neural recordings.
    Neuroimage. 2017 Jun 13;157:331-340. doi: 10.1016/j.neuroimage.2017.06.025.
"""

# import packages
import numpy as np
from dataclasses import dataclass, field
from os.path import join, exists
from os import listdir, makedirs
import json
from itertools import compress
from pandas import DataFrame, isna
from scipy.signal import welch

import meet.meet as meet  # https://github.com/neurophysics/meet

from utils.utils_windowing import get_windows
from lfpecog_preproc.preproc_import_scores_annotations import get_ecog_side
from utils.utils_fileManagement import (
    get_onedrive_path, get_project_path,
    load_class_pickle, save_class_pickle,
    load_ft_ext_cfg
)
from lfpecog_features.feats_helper_funcs import (
    check_matrix_properties, regularize_matrix
)

def get_SSD_component(
    data_2d, fband_interest, s_rate,
    flank_distance=5,
    use_freqBand_filtered=True, return_comp_n=0,
):
    """
    Function to freq-specifically filter 2d-data
    with SSD (using meet toolbox).

    Inputs:
        - data_2d (ndarray): 2d array n-channels x n-samples
        - fband_interest: tuple or list with lower
            and upper freq-band of interest border
        - fs (int): sampling rate
        - flank_distance (int/float): distance distal of freq-band
            of interest to define flank bands,
            defaults to +/- 5 Hz
        - use_freqBand_filtered (bool): use the freq-band
            filtered signal to multiply with the
            SSD-filter and to return
        - return_comp_n (int): the nth SSD-component to
            return (0 is the first component)
    
    Returns:
        - SSD_filtered_data
        - SSD_pattern
        - SSD_eigvals

    """
    assert type(return_comp_n) in [int, np.ndarray, list], (
        'return_comp_n should be integer or list/array,'
        f' instead {type(return_comp_n)} was given'
    )

    if isinstance(return_comp_n, int):
        assert return_comp_n < data_2d.shape[1], (
            'return_comp_n cannot be larger than n-channels'
        )
    else:
        for c in return_comp_n:
            assert c < data_2d.shape[1], (
                'return_comp_n cannot be larger than n-channels'
            )

    # set bandrange of interest    
    freqOfInt_range = np.r_[fband_interest]
    # set bandrange of flanks (SSD values of interest are normalized versus the full flank and bw-of-int)
    if flank_distance == 'broadband':
        flank_range = np.r_[4, 198]
    else:
        flank_range = np.r_[freqOfInt_range[0] - flank_distance,
                            freqOfInt_range[1] + flank_distance]
    # correct negative freq-values to 1 Hz
    flank_range[flank_range < 1] = 1

    # filter signals for SSD
    interestBand_filt = meet.iir.butterworth(data_2d, fp=freqOfInt_range,
                                    fs=np.r_[0.8,1.2]*freqOfInt_range, s_rate=s_rate)

    wide_filt= meet.iir.butterworth(data_2d, fp=flank_range,
                                    fs=np.r_[0.8,1.2]*flank_range, s_rate=s_rate)
    flank_filt = meet.iir.butterworth(wide_filt, fs=freqOfInt_range,
                                    fp=np.r_[0.8,1.2]*freqOfInt_range, s_rate=s_rate)

    # Perform SSD
    SSD_filter, SSD_eigvals = meet.spatfilt.CSP(interestBand_filt, flank_filt)
    SSD_pattern = meet.spatfilt.pattern_from_filter(SSD_filter, interestBand_filt)
    #normalize the pattern to have a maximum absolute value of 1
    SSD_pattern /= np.abs(SSD_pattern).max(-1)[:,np.newaxis]

    if use_freqBand_filtered:
        # Apply freq-spec SSD filter on beta-filtered signals
        SSD_filtered_data = SSD_filter.T @ interestBand_filt

    else:
        # Apply freq-spec SSD filter on origin data containing all frequencies
        SSD_filtered_data = SSD_filter.T @ data_2d

    # if only one component is asked
    if isinstance(return_comp_n, int):
        return SSD_filtered_data[return_comp_n], SSD_pattern, SSD_eigvals
    
    # if multiple components are asked
    elif np.logical_or(
        isinstance(return_comp_n, list),
        isinstance(return_comp_n, np.ndarray)
    ):
        return_array = np.zeros((len(return_comp_n), data_2d.shape[1]))
        for i, c in enumerate(return_comp_n):
            return_array[i, :] = SSD_filtered_data[c]
        return return_array, SSD_pattern, SSD_eigvals
        

@dataclass(init=True, repr=True,)
class SSD_bands_per_window:
    """
    Store SSD filtered timeseries per freq-band,
    is used per window

    - data: 2d array n-channels x n-samples
    - s_rate: sampling rate
    - freq_bands_incl: dict like {'alpha': [4, 8],
        'lo_beta': [12, 20], etc}
    - ssd_flanks: either the width in Hz, or 'broadband'
        which defines the flanks always between 4 and 198 Hz
    """
    data: np.ndarray
    s_rate: int
    freq_bands_incl: dict
    ssd_flanks = 5

    def __post_init__(self,):
        for f_band in self.freq_bands_incl.keys():
            ssd_ts, _, _ = get_SSD_component(
                data_2d=self.data,
                fband_interest=self.freq_bands_incl[f_band],
                s_rate=self.s_rate,
                flank_distance=self.ssd_flanks,
            )
            setattr(self, f_band, ssd_ts)


@dataclass(init=True, repr=True,)
class SSD_bands_windowed:
    """
    Get object containing attribute with windowed
    SSDs timeseries per freq band, and timestamps

    - sub: sub string code
    - data_source: string 'lfp_side' , 'ecog_side
    - settings dict

    Results:
        - data_source: 2d-array [n-windows, n-samples]
        - times: list with timestamps of windows
    """
    sub: str
    datasource: str
    settings: dict
    incl_ecog: bool = True
    incl_stn: bool = True
    
    def __post_init__(self,):
        # use stored data
        try:
            data, meta = load_windowed_ssds(
                sub=self.sub,
                dType=self.datasource,
                settings=self.settings)
            print('\t...loaded SSD windowed-data and meta-info'
                  f' for {self.datasource} of sub-{self.sub}')
            for i_f, fband in enumerate(meta['bandwidths']):
                setattr(self, fband, data[:, i_f, :])  # [n-windows x n-samples]
            
            if 'ssd_flanks' in list(meta.keys()): self.flanks = meta['ssd_flanks']
        
        # create data if not found
        except FileNotFoundError:
            print('\t...windowed SSD data doesnot exist yet'
                  f' for {self.datasource} of sub-{self.sub}')
            create_SSDs(sub=self.sub, settings=self.settings,
                        incl_ecog=self.incl_ecog,
                        incl_stn=self.incl_stn)
            print('\t...created SSD windowed-data and meta-info'
                  f' for {self.datasource} of sub-{self.sub}')
            # use newly created data
            data, meta = load_windowed_ssds(sub=self.sub,
                                            dType=self.datasource,
                                            settings=self.settings)
        
        for i_f, fband in enumerate(meta['bandwidths']):
            setattr(self, fband, data[:, i_f, :])  # [n-windows x n-samples]

        self.times = meta['timestamps']
        # correct for incorrect timestamps in v4 subject 103 (Wrong day in json)
        if self.sub == '103' and max(self.times) > 2e6:
            setattr(self,
                    'times',
                    np.array(self.times) - np.float64(27 * 24 * 60 * 60))
            print('corrected 103 SSD timings for', self.datasource)
        self.fs = meta['fs']



def load_windowed_ssds(sub, dType, settings: dict):
    """
    loads SSD bands saved per datatype as 3d arrays
    [n-windows, n-bands, n-samples]

    craeted in feats_extract_multivar.py

    Returns:
        - data_ssd: 3d-array containing n-windows,
            n-freqbands, n-samples-per-window
        - meta_ssd: dict containing 'bandwidths' and
            'timestamps' corresponding to data.
            'bandwidths' contain freq-limits.
    """
    # define folder and file to load from
    win_path = join(get_project_path('data'),
                    'windowed_data_classes_'
                    f'{settings["WIN_LEN_sec"]}s_'
                    f'{settings["WIN_OVERLAP_part"]}overlap',
                    settings['DATA_VERSION'],
                    f'sub-{sub}')
    ssd_win_fname = f'SSD_windowedBands_{sub}_{dType}'

    try:
        if settings['SSD_flanks'] == 'broadband':
            ssd_win_fname = f'broad{ssd_win_fname}'
    except:
        ssd_win_fname=ssd_win_fname

    meta_f = join(win_path, ssd_win_fname + '.json')
    data_f = join(win_path, ssd_win_fname + '.npy')
    
    # # assure existence of files
    # assert np.logical_and(exists(meta_f), exists(data_f)), (
    #     f'inserted SSD data files {ssd_win_fname} (.npy, .json)'
    #     f'do not exist in {win_path}'
    # )

    # load files
    with open(meta_f, 'r') as meta_f:
        meta_ssd = json.load(meta_f)
    data_ssd = np.load(data_f, allow_pickle=True,).astype(np.float64)

    
    return data_ssd, meta_ssd



@dataclass(init=True, repr=True)
class get_subject_SSDs:
    sub: str
    settings: dict = None
    ft_setting_fname: str = None
    incl_ecog: bool = True
    incl_stn: bool = True

    def __post_init__(self,):
        if not self.settings:
            # load settings dict
            self.settings = load_ft_ext_cfg(self.ft_setting_fname)
        # define ephys data sources
        self.ephys_sources = []
        if self.incl_ecog:
            ecog_side = get_ecog_side(self.sub)
            if ecog_side:
                self.ephys_sources.append(f'ecog_{ecog_side}')
            else:
                print(f'\n\tsubject {self.sub} has STN ONLY recordings')
        if self.incl_stn:
            self.ephys_sources.extend(['lfp_left', 'lfp_right'])

        # for data sources, get all SSD windows
        for dType in self.ephys_sources:
            setattr(self,
                    dType,
                    SSD_bands_windowed(self.sub, dType,
                                       self.settings,
                                       incl_ecog=self.incl_ecog,
                                       incl_stn=self.incl_stn)
            )
            

import matplotlib.pyplot as plt

@dataclass(init=True, repr=True, )
class create_SSDs():
    """
    Create windowed SSD timeseries per defined freq-
    bandwidths
    """
    sub: str
    settings: dict = field(default_factory=lambda:{})
    ft_setting_fname: str = None
    incl_ecog: bool = True
    incl_stn: bool = True
    use_stored_windows: bool = True
    save_ssd_windows: bool = True
    check_matrix: bool = False
    MATRIX_REGULARIZATION: bool = False
    
    def __post_init__(self,):
        ### load settings from json
        if self.settings == {}:
            SETTINGS = load_ft_ext_cfg(self.ft_setting_fname)
        else:
            SETTINGS = self.settings
        # define SSD flank setting
        try:
            SSD_FLANKS = SETTINGS['SSD_flanks']
        except:
            SSD_FLANKS = 5
        
        # define ephys_sources
        self.ephys_sources = []
        if self.incl_ecog:
            ecog_side = get_ecog_side(self.sub)
            self.ephys_sources.append(f'ecog_{ecog_side}')
        if self.incl_stn:
            self.ephys_sources.extend(['lfp_left', 'lfp_right'])

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
        if not exists(windows_path): makedirs(windows_path)
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart create SSDs: {dType}')
            ssd_windows_name = f'SSD_windowedBands_{self.sub}_{dType}'
            # add broadband flanks to SSD-naming
            if SSD_FLANKS == 'broadband': ssd_windows_name = f'broad{ssd_windows_name}'
            # check existence of ssd windowed data
            if (SETTINGS['OVERWRITE_DATA'] == False and
                exists(join(windows_path, ssd_windows_name+'.json')) and
                exists(join(windows_path, ssd_windows_name+'.npy'))
            ):  
                # CREATE ATTRIBUTE CONTAINING WINDOWED SSD BANDS
                setattr(self,
                        dType,
                        SSD_bands_windowed(self.sub, dType, SETTINGS))
                print(f'\n\texisting windowed ssd-data loaded {ssd_windows_name}')
                continue
            
            # Create windowed SSDd data 
            print('\tcreate windowed ssd data...')

            # define path for windowed preprocessed data of dType
            dType_fname = (f'sub-{self.sub}_windows_'
                           f'{SETTINGS["WIN_LEN_sec"]}s_'
                           f'{SETTINGS["DATA_VERSION"]}_{dType}.P')
            dType_win_path = join(windows_path, dType_fname)
            
            # check if windows are already available
            if np.logical_and(self.use_stored_windows,
                              exists(dType_win_path)):
                print(f'\tload data from {windows_path}....')
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
            ssd_arr_3d = []

            NaN_accept_part = .33
            n_samples_win = windows.fs * SETTINGS["WIN_LEN_sec"]
            # loop over windows
            for i_w, win_dat in enumerate(windows.data):
                
                win_dat = win_dat.astype(np.float64)
                # print(f'start win # {i_w}, win_dat shape: {win_dat.shape}')

                if NaN_accept_part == 0:
                    # select only rows without missings in current window
                    nan_rows = np.array([isna(win_dat[:, i]).any()
                                for i in range(win_dat.shape[-1])])
                    win_dat = win_dat[:, ~nan_rows]
                    # if sum(nan_rows) > 0: print(f'{sum(nan_rows)} rows deleted')
                else:
                    # select rows with less than threshold NaNs
                    nan_rows_parts = np.array([sum(isna(win_dat[:, i])) / len(win_dat)
                                for i in range(win_dat.shape[-1])])
                    nan_rows = nan_rows_parts > NaN_accept_part
                    win_dat = win_dat[:, ~nan_rows]
                    # select out nan-timings
                    i_nan = isna(win_dat).any(axis=1)
                    win_dat = win_dat[~i_nan, :]
                    # if sum(i_nan) > 0: print(f'{sum(i_nan)} indices deleted')

                
                # check data shape after nan-removal
                # dont perform SSD on empty array
                if win_dat.shape[1] == 0 or win_dat.shape[0] < NaN_accept_part:
                    print(f'{dType}: window # {i_w} no NON-NAN data')
                    i_nan = isna(windows.data[i_w]).any(axis=1)
                    
                    ssd_arr_2d = []
                    # add nan-list for every bandwidth
                    for i_temp in np.arange(len(SETTINGS['SPECTRAL_BANDS'])):
                        ssd_arr_2d.append([np.nan] * n_samples_win)  # add nans if not converted
                    # add all lists of window to 3d array
                    ssd_arr_2d = np.array(ssd_arr_2d, dtype='object')  # convert list to 2d array
                    ssd_arr_3d.append(ssd_arr_2d)

                    # # visualise window before nan-removal
                    # for i_row in np.arange(windows.data.shape[2]):
                    #     plt.plot(windows.data[i_w, :, i_row],
                    #              label=windows.keys[i_row])
                    # plt.plot(i_nan.astype(int) * 2e-4,
                    #          label='NAN')

                    # plt.legend()
                    # plt.title(f'window # {i_w}, removed due to NaNs')
                    # plt.show()
                    # raise ValueError('DEBUG: after plot error')
                    continue

                # empty list to store ssd-signals for this window
                ssd_arr_2d = []
                
                ### CALCULATE SSD timeseries per band
                
                if self.check_matrix:
                    check_matrix_properties(M=win_dat, verbose=True)
                    # if self.MATRIX_REGULARIZATION:
                    #     win_dat = regularize_matrix(M=win_dat, lasso_alpha=.1,)
                    #     check_matrix_properties(M=win_dat, verbose=True)
    
                        
                
                # loop over defined frequency bands
                for bw in SETTINGS['SPECTRAL_BANDS']:
                    f_range = SETTINGS['SPECTRAL_BANDS'][bw]
                    
                    try:
                        # print(f'\tSSD for {bw}')
                        (ssd_filt_data,
                            ssd_pattern,
                            ssd_eigvals
                        ) = get_SSD_component(
                            data_2d=win_dat.T,
                            fband_interest=f_range,
                            s_rate=windows.fs,
                            flank_distance=SSD_FLANKS,
                            use_freqBand_filtered=True,
                            return_comp_n=0,
                        )
                        if len(ssd_filt_data) < n_samples_win:
                            n_pad = n_samples_win - len(ssd_filt_data)
                            ssd_filt_data = np.concatenate([
                                ssd_filt_data, [np.nan] * n_pad
                            ])
                            # print(f'padded n={n_pad}')
                            # TODO: remove NaNs in feature selection
                        
                        ssd_arr_2d.append(ssd_filt_data)  # add band to current window

                    except ValueError:
                        print(f'{dType}: SSD in window # {i_w} failed on {bw}')
                        ssd_arr_2d.append([np.nan] * win_dat.shape[0])  # add nans if not converted

                        # DEBUG: why SSD failed
                        print(f'shape of win_dat: {win_dat.shape}')
                        print(f'win_dat nans or inf: {np.isnan(win_dat)}'
                              f', max: {np.max(win_dat, axis=0)}'
                              f', nan-max: {np.nanmax(win_dat, axis=0)}'
                              )

                        continue

                # End of window: add 2d-data of window to 3d overall list
                ssd_arr_2d = np.array(ssd_arr_2d, dtype='object')  # convert list to 2d array
                ssd_arr_3d.append(ssd_arr_2d)
            
            # End of ALL windows within dataType: STORE FEATURE DATAFRAME
            ssd_arr_3d = np.array(ssd_arr_3d, dtype='object')  # convert array-list to 3d array
            setattr(self, dType, ssd_arr_3d)
            
            if self.save_ssd_windows:
                # save SSD-bands per window as .npy and meta-data as .json
                with open(join(windows_path, ssd_windows_name+'.npy'), mode='wb') as f:
                    np.save(f, ssd_arr_3d)

                # check compatability data and meta info
                assert np.logical_and(
                    ssd_arr_3d.shape[0] == len(windows.win_starttimes),
                    ssd_arr_3d.shape[1] == len(SETTINGS['SPECTRAL_BANDS'])
                ), (f'ssd_arr_3d shape ({ssd_arr_3d.shape}) does not match '
                    f'n-window-times ({len(windows.win_starttimes)}) or '
                    f"n-freqbands ({len(SETTINGS['SPECTRAL_BANDS'])})")
                
                meta = {'npy_filename': ssd_windows_name,
                        'fs': windows.fs,
                        'bandwidths': SETTINGS['SPECTRAL_BANDS'],
                        'ssd_flanks': SSD_FLANKS,
                        'timestamps': windows.win_starttimes}
                
                with open(join(windows_path, ssd_windows_name+'.json'), 'w') as f:
                    json.dump(meta, f)
                
                print(f'Saved SSD windowed data (SSD-flanks: {SSD_FLANKS})'
                      f' and meta for sub-{self.sub}: {dType}'
                      f' as {ssd_windows_name} in {windows_path}')
            