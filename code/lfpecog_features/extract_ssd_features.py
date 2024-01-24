'''
Feature Extraction using SSD'd time-series

extraciton of both local features, as well as
(connetivity) features between two ephys-data sources
(STN-STN, or STN-ECoG)
'''

# Import public packages and functions
import json
from typing import Any
import numpy as np
from os.path import join, exists, splitext
from os import listdir, makedirs
from pandas import isna, DataFrame
from scipy.signal import welch, hilbert
from scipy.stats import variation
from array import array
from dataclasses import dataclass, field
from itertools import product, compress
import matplotlib.pyplot as plt


# import own functions
from utils.utils_fileManagement import (
    make_object_jsonable, get_project_path,
    get_avail_ssd_subs
)
import lfpecog_features.feats_spectral_features as specFeats
from lfpecog_features.feats_helper_funcs import smoothing
from lfpecog_features.bursts_funcs import (
    get_burst_indices, calc_burst_profiles
)
from lfpecog_analysis.ft_processing_helpers import (
    categorical_CDRS
)
from lfpecog_predict.prepare_predict_arrays import (
    get_move_selected_env_arrays
)


def get_moveSpec_predArrays(
    MOV_SEL: str, FEAT_WIN_sec: int = 5,
    FT_VERSION: str = 'v6', DATA_VERSION: str = 'v4.0',
    LOAD_SOURCES=['STN', 'ECOG', 'CONN'],
    INCL_ECOG: bool = False,
):
    """
    Arguments:
        - MOV_SEL: define movement dependency to extract
    """
    # try to LOAD
    (
        X_arr, y_arr, sub_arr, feat_names
    ) = saveLoad_mov_spec_predArrays(
        SAVELOAD='LOAD', MOV_SEL=MOV_SEL,
        FEAT_WIN_sec=FEAT_WIN_sec,
        LOAD_SOURCES=LOAD_SOURCES,
    )

    if isinstance(X_arr, bool):
        # create if load not successful
        print(f'CREATING {MOV_SEL} PRED ARRAYS ({FEAT_WIN_sec} sec)')

        # get move-selected env arrays
        DATA = {}
        SUBS = get_avail_ssd_subs(DATA_VERSION=DATA_VERSION,
                                  FT_VERSION=FT_VERSION,)

        for sub in SUBS:
            DATA[sub] = get_move_selected_env_arrays(
                sub=sub, LOAD_SAVE=True
            )
        # create group arrays
        (
            X_arr, y_arr, sub_arr, feat_names
        ) = create_pred_arrays_from_movSpec_envs(
            env_dict=DATA, incl_ECOG=INCL_ECOG,
            MOV_SEL=MOV_SEL, FEAT_WIN_sec=FEAT_WIN_sec,
        )
        # save afterwards
        saveLoad_mov_spec_predArrays(
            X_arr=X_arr, y_arr=y_arr,
            sub_arr=sub_arr, feat_names=feat_names,
            SAVELOAD='SAVE', MOV_SEL=MOV_SEL,
            FEAT_WIN_sec=FEAT_WIN_sec,
        )

    else:
        print(f'LOADED {MOV_SEL} PRED ARRAYS ({FEAT_WIN_sec} sec)')

    return X_arr, y_arr, sub_arr, feat_names


def create_pred_arrays_from_movSpec_envs(
    env_dict: dict, MOV_SEL: str,
    incl_ECOG: bool = False,
    FEAT_WIN_sec: float = 1.0,
    sfreq: int = 2048,
    INCL_POWER: bool = True,
    INCL_SPECVAR: bool = True,
    INCL_COH_biSTN: bool = False,
    INCL_COH_STNECOG: bool = False,
    CATEG_CDRS: bool = True,
    CDRS_cuts = [3.5, 7.5],
):
    """
    Arguments:
        - env_dict: dict with envelops per subject and source



    Returns:
        - X_arr
        - y_arr: contains CDRS scores (default bilat incl axial)
        - sub_arr: contains 
    """
    allowed_moves = ['DEPEND', 'INDEP', 'all']
    assert MOV_SEL in allowed_moves, (f'MOV_SEL {MOV_SEL} not in'
                                      f'{allowed_moves}')
    

    
    win_samples = int(sfreq * FEAT_WIN_sec)

    if ~ incl_ECOG: INCL_COH_STNECOG = False

    bands = ['theta', 'alpha', 'lowbeta', 'highbeta', 'gamma']
    ft_names = {'STN': [], 'ECOG': [], 'CONN': []}

    if INCL_POWER:
        ft_names['STN'].extend([f'STN_power_{b}' for b in bands])
        if incl_ECOG: ft_names['ECOG'].extend([f'ECOG_{b}_power' for b in bands])

    if INCL_SPECVAR:
        ft_names['STN'].extend([f'STN_var_{b}' for b in bands])
        if incl_ECOG: ft_names['ECOG'].extend([f'ECOG_{b}_var' for b in bands])
        
    if INCL_COH_biSTN:
        ft_names['CONN'].extend([f'biSTN_coh_{b}' for b in bands])

    if INCL_COH_STNECOG:
        ft_names['CONN'].extend([f'STNECOG_coh_{b}' for b in bands])
    
    feat_funcs = {
        'power': calc_power,
        'var': calc_var
    }

    temp_feat_lists = {'STN': [], 'ECOG': [], 'CONN': []}
    temp_cdrs_lists = {'STN': [], 'ECOG': [], 'CONN': []}
    temp_sub_lists = {'STN': [], 'ECOG': [], 'CONN': []}

    for sub in env_dict:
        print(f'\n- start sub-{sub} move-specific ssd-feature extraction')

        for src in env_dict[sub].keys():
            # check for ecog incl and extract SRC for later feat naming
            if 'ecog' in src and not incl_ECOG: continue
            if 'ecog' in src: SRC = 'ECOG'
            else: SRC = 'STN'
            
            # extract meta info
            cdrs = env_dict[sub][src][-4, :]
            time = env_dict[sub][src][-3, :]
            move = env_dict[sub][src][-1, :]

            # select on movement dependency
            if MOV_SEL == 'INDEP': mov_sel = move == 0
            if MOV_SEL == 'DEPEND': mov_sel = move == 1
            if MOV_SEL == 'all': mov_sel = np.ones_like(move).astype(bool)
            sig_arr = env_dict[sub][src][:len(bands), mov_sel]
            cdrs = cdrs[mov_sel]

            # categorize CDRS
            cdrs_present = np.unique(cdrs)
            if CATEG_CDRS:
                cdrs = categorical_CDRS(cdrs, cutoff_mildModerate=CDRS_cuts[0],
                                        cutoff_moderateSevere=CDRS_cuts[1],)
                cdrs_present = np.unique(cdrs)
            
            for CAT in cdrs_present:
                temp_sigs = sig_arr[:, cdrs == CAT]
                full_wins = int(temp_sigs.shape[1] // (win_samples / 2))
                samples_incl = int(full_wins * (win_samples / 2))
                temp_sigs = temp_sigs[:, :samples_incl]  # cut to full 1/2-window length

                if samples_incl < win_samples:
                    print(f'...NOT ENOUGH DATA FOR FEATURE. {sub, src, CAT}')
                    continue

                # loop over available feature windows, and add features per window
                for win_i0 in np.arange(0, temp_sigs.shape[1], win_samples*.5)[:-1]:
                    win_sig = temp_sigs[:, int(win_i0):int(win_i0 + win_samples)]
                    temp_win_fts = []
                    # loop over src-features (always cmatching order)
                    for ft in ft_names[SRC]:
                        band = ft.split('_')[-1]
                        i_band = np.where([b == band for b in bands])[0][0]
                        ft_type = ft.split('_')[1]
                        ft_func = feat_funcs[ft_type]
                        temp_win_fts.append(ft_func(win_sig[i_band]))
                    # ADDING FEATURES, SCORES, SUB-IDS
                    temp_feat_lists[SRC].append(temp_win_fts)
                    temp_cdrs_lists[SRC].append(CAT)
                    temp_sub_lists[SRC].append(sub)


    # when all subjects are added, convert to arrays
    for d in [temp_feat_lists, temp_cdrs_lists, temp_sub_lists]:
        for k in d.keys():
            d[k] = np.array(d[k])
    
    return temp_feat_lists, temp_cdrs_lists, temp_sub_lists, ft_names


def calc_power(sig_win):
    
    pow = np.nanmean(sig_win)

    return pow

def calc_var(sig_win):

    var = variation(sig_win)

    return var


def saveLoad_mov_spec_predArrays(
    MOV_SEL, FEAT_WIN_sec,
    X_arr=None, y_arr=None, sub_arr=None,
    feat_names=None, SAVELOAD: str = 'SAVE',
    LOAD_SOURCES=['STN', 'ECOG', 'CONN'],
):
    ft_path = join(
        get_project_path('results'),
        'features', 'SSD_feats_broad_v6',
        'v4.0', 'movement_specific'
    )

    if SAVELOAD == 'SAVE':
    
        for src in X_arr:
            # general naming
            name_base = (f'predArr_{src}_mov{MOV_SEL}_'
                        f'{FEAT_WIN_sec}sWindows_')
            
            # save X_arr
            np.save(join(ft_path, name_base + 'X.npy'),
                    X_arr[src], allow_pickle=True)
            # save y_arr
            np.save(join(ft_path, name_base + 'y.npy'),
                    y_arr[src], allow_pickle=True)
            # save sub_arr
            np.save(join(ft_path, name_base + 'subs.npy'),
                    sub_arr[src], allow_pickle=True)
            # save feat_names
            with open(
                join(ft_path, name_base + 'ftnames.json'),
                'w'
            ) as f:
                json.dump(feat_names[src], f)
        
        return print('SAVED ALL ARRAYS and JSONs')


    elif SAVELOAD == 'LOAD':
        X_arr, y_arr, sub_arr, feat_names = {}, {}, {}, {}

        for src in LOAD_SOURCES:
            name_base = (f'predArr_{src}_mov{MOV_SEL}_'
                         f'{FEAT_WIN_sec}sWindows_')
            try:
                # load arrays and names
                X_arr[src] = np.load(
                    join(ft_path, name_base + 'X.npy'),
                    allow_pickle=True
                )
                # save y_arr
                y_arr[src] = np.load(
                    join(ft_path, name_base + 'y.npy'),
                    allow_pickle=True
                )
                # save sub_arr
                sub_arr[src] = np.load(
                    join(ft_path, name_base + 'subs.npy'),
                    allow_pickle=True)
                # save feat_names
                with open(join(ft_path, name_base +
                                    'ftnames.json'), 'r') as f:
                    feat_names[src] = json.load(f)
            except FileNotFoundError:
                print(f'files not found ({name_base})')
                return [False] * 4

        return X_arr, y_arr, sub_arr, feat_names
    


@dataclass(init=True, repr=True, )
class extract_local_SSD_powers():
    """
    Extract local power spectral features from
    SSD-d data
    """
    sub_SSD: str
    feat_path: str
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.sub_SSD.settings
        self.overwrite_features = SETTINGS['OVERWRITE_FEATURES']

        # loop over possible datatypes
        for dType in self.sub_SSD.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD powers')
            # check if features already exist
            feat_fname = f'SSDfeats_{self.sub_SSD.sub}_{dType}_local_spectralFeatures.csv'

            if exists(join(self.feat_path, feat_fname)) and not self.overwrite_features:
                print(f'...\tfeatures already exist for {dType}'
                      ' and are NOT overwritten, skip!')
                continue

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
            
            # get data and loop over windows
            data = getattr(self.sub_SSD, dType)  # assign datatype data

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

                    ssd_signal = getattr(data, bw)[i_w].copy()
                    # remove NaNs padded to SSD timeseries in create_SSDs()
                    ssd_signal = ssd_signal[~isna(ssd_signal)]

                    # Convert SSD'd signal into Power Spectrum
                    f_arr, psd = welch(ssd_signal, axis=-1,
                                       nperseg=data.fs, fs=data.fs)
                    f_sel = [f_range[0] < freq < f_range[1] for freq in f_arr]  # select psd in freq of interest

                    # CALCULATE SPECTRAL PEAK FEATURES
                    # loop over ft-names and ft-funcs ensures correct ft-order
                    for ft_name in fts_incl:
                        if fts_incl[ft_name] and ft_name.startswith('SSD_'):
                            # get value from function corresponding to ft-name
                            ft_func = getattr(
                                specFeats.Spectralfunctions(
                                    psd=psd, ssd_sig=ssd_signal,
                                    f=f_arr, f_sel=f_sel,
                                    SSD_FILTERED=SETTINGS['use_filtered_SSD'],
                                    f_range=f_range, sfreq=data.fs,
                                ), f'get_{ft_name}'
                            )
                            feats_win.append(ft_func())
                       
                # END OF WINDOW -> STORE list with window features to total list
                feats_out.append(feats_win)

            # AFTER ALL WINDOWS OF DATA TYPE ARE DONE -> STORE FEATURE DATAFRAME
            feats_out = np.array(feats_out, dtype='object')
            feats_out = DataFrame(data=feats_out, columns=feat_names, index=data.times,)
            feats_out.to_csv(join(self.feat_path, feat_fname),
                             index=True, header=True,)
            print(f'Saved FEATURES for sub-{self.sub_SSD.sub} {dType} as '
                  f'{feat_fname} in {self.feat_path}')



def get_ssd_bursts(
    sub_SSD,
    feat_path: str,
    SMOOTH_MILLISEC = {'beta': 250, 'gamma': 100, 'lowfreq': 250},
    bands_include = ['lo_beta', 'hi_beta', 'gamma'],
    OVERWRITE=True,
):
    """
    Extracts bursts for every window, using a threshold based
    on the windows itself
    Gamma burst differ between dopaminergic states in burst rate,
    not in amplitude or duration (Lofredi, ELife 2018).

    Arguments:
        - sub: e.g. '008'
        - SMOOTH_MILLISEC: milliseconds window for envelop smoothing
        - THRESH_ORIGIN: off or on or combi
        - TO_SAVE_FIG: to save or not
        - FIG_DIR: needed if figure should be saved
        - sub_SSD_class: result from ssd.get_subject_SSDs(),
            if given, this will decrease computational time
        - LOAD_STORED_RESULTS: make use previsouly created brust-values
    """
    print(f'start {sub_SSD.sub}, burst-extraction')
    
    filename = f'SSDfeats_{sub_SSD.sub}_BURSTS.json'

    values_store = {}  # to store final values

    for source, bw in product(sub_SSD.ephys_sources, bands_include):
        print(f'start getting {bw} bursts from {source}')
        # define burst-settings for bw and source
        win_len_sec = sub_SSD.settings['WIN_LEN_sec']
        Fs = getattr(sub_SSD, source).fs
        if 'beta' in bw:
            SMOOTH = SMOOTH_MILLISEC['beta']
            MIN_BURST_SEC = 1 / 12  # samples for cycle-length
            metric = 'duration'
        elif 'gamma' in bw:
            SMOOTH = SMOOTH_MILLISEC['gamma']
            MIN_BURST_SEC = 1 / 60  # samples for cycle-length
            metric = 'rate'
        else:
            SMOOTH = SMOOTH_MILLISEC['lowfreq']
            MIN_BURST_SEC = 1 / 4  # samples for cycle-length
        # print(f'BURSTS {bw} {source}: smooth: {SMOOTH} ms, min-sec {MIN_BURST_SEC}, metric: {metric}')
        # select SSD'd data to extract from
        tempdat = getattr(sub_SSD, source)
        fs = tempdat.fs
        temp_times = tempdat.times.copy()
        tempdat = getattr(tempdat, bw)  # 2d array of shape n-windows, n-samples (per window) 
        # temp_values={'times': [],
        #               'mean_duration_sec': [],
        #               'burst_prob': []}  # store timestamps corresponding to n-windows
        
        temp_values = {'times': [], f'{bw}_{metric}': []}

        for sig, time in zip(tempdat, temp_times):  # loops over all windows
            # clean signal from NaNs
            sig = sig[~np.isnan(sig)]
            if len(sig) < (.5 * 5 * Fs): continue  # skip windows with too few samples
            
            # get envelop and apply smoothing
            env = abs(hilbert(sig))  # env from SSD'd timeseries of one window
            if SMOOTH > 0: env = smoothing(sig=env, fs=fs, win_ms=SMOOTH)
            
            # get burst-lengths and exclude too short bursts
            burst_value = calc_burst_profiles(env=env, bw=bw, Fs=Fs,
                                              min_burst_sec=MIN_BURST_SEC,
                                              window_length_sec=win_len_sec)
            temp_values['times'].append(time)
            temp_values[f'{bw}_{metric}'].append(burst_value)
            

            # start_idx, end_idx = get_burst_indices(envelop=env, burst_thr=THRESH)
            # burst_lengths_sec = (end_idx - start_idx) / Fs
            # burst_lengths_sec = burst_lengths_sec[burst_lengths_sec >= MIN_BURST_SEC]
            # temp_values['mean_duration_sec'].append(np.nanmean(burst_lengths_sec))
            # temp_values['burst_prob'].append(sum(burst_lengths_sec) / (len(sig) / Fs))
            
            
        # save values correct in json
        for key in temp_values.keys():
            temp_values[key] = make_object_jsonable(temp_values[key])
            
        # add important settings variables to save
        temp_values['win_len_sec'] = float(sub_SSD.settings['WIN_LEN_sec'])
        temp_values['smooth_millisec'] = int(SMOOTH)  # convert to json writable int
        temp_values['min_burst_sec'] = float(MIN_BURST_SEC)
        temp_values['data_version'] = sub_SSD.settings['DATA_VERSION']  # string is json writable

    
        values_store[f'{source}_{bw}'] = temp_values
        # print(f'save dict as "{source}_{bw}" with keys: {temp_values.keys()}')
    with open(join(feat_path, filename), 'w') as f:
        json.dump(values_store, f)


from lfpecog_features.feats_phase_amp_coupling import calculate_PAC_matrix

@dataclass(init=True, repr=True,)
class extract_local_SSD_PACs:
    """
    Extracting local PAC based on SSD'd freq-bands
    stored per window

    Called from run_ssd_ftExtr
    """
    sub_SSD: str
    feat_path: int
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.sub_SSD.settings    
        
        # loop over possible datatypes
        for dType in self.sub_SSD.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD PACs')
            data = getattr(self.sub_SSD, dType)  # assign datatype data
            # data is here class with per bandwidth a 2d array (n-windows, n-samples)

            # define PAC-frequencies to extract
            pac_bands = SETTINGS['FEATS_INCL']['local_PAC_freqs']  # get features to include

            # load SSD'd data
            for [pha, ampl] in pac_bands:
                print(f'\n\t...START PAC {pha} {ampl}')
                # check if features already exist
                feat_fname = (f'SSDfeats_{self.sub_SSD.sub}_{dType}_localPAC'
                              f'_{pha}_{ampl}.npy')

                if exists(join(self.feat_path, feat_fname)) and not self.overwrite_features:
                    print(f'...\tlocal PAC already exist for {dType} '
                          f'{pha}_{ampl} and are NOT overwritten, skip!')
                    continue
                
                # calculate 3d array with pac values (n-windows, n-ampl, n-phase bins)
                pac_values, pac_times = calculate_PAC_matrix(
                    sig_pha=getattr(data, pha),
                    sig_amp=getattr(data, ampl),
                    fs=data.fs,
                    freq_range_pha=SETTINGS['SPECTRAL_BANDS'][pha],
                    freq_range_amp=SETTINGS['SPECTRAL_BANDS'][ampl],
                    window_times=data.times
                )

                # store feature values and window times
                np.save(file=join(self.feat_path, feat_fname),
                        arr=pac_values,
                        allow_pickle=True)
                ft_time_fname = splitext(feat_fname)[0] + '_times.csv'
                np.savetxt(fname=join(self.feat_path, ft_time_fname),
                           X=pac_times, delimiter=',',)


from lfpecog_features.feats_phases import get_phase_synchrony_index

@dataclass(init=True, repr=True, )
class extract_SSD_connectivity:
    """
    Extract of Connectivity features per segments

    Called from run_ssd_ftExtr
    """
    sub_SSD: str
    sources: str
    connectivity_metric: str
    feat_path: int
    incl_stn_ecog: bool = True
    incl_stn_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        assert self.sources.upper() in ['STN_ECOG', 'STN_STN']

        SETTINGS = self.sub_SSD.settings    

        bands_to_extract = SETTINGS['SPECTRAL_BANDS']

        if self.sources == 'STN_STN':
            seed_data = self.sub_SSD.lfp_left
            target_data = self.sub_SSD.lfp_right
        
        elif self.sources == 'STN_ECOG':
            if self.sub_SSD.sub.startswith('1'):
                return print(f'subject-{self.sub_SSD.sub} has no ECOG')

            ecog_source = [s for s in self.sub_SSD.ephys_sources if 'ecog' in s][0]
            stn_ecog_side = f"lfp_{ecog_source.split('_')[1]}"
            seed_data = getattr(self.sub_SSD, stn_ecog_side)
            target_data = getattr(self.sub_SSD, ecog_source)

        # define time-samples not present in seed and target
        if len(seed_data.times) != len(target_data.times):
            time_sel_seed = [t in target_data.times for t in seed_data.times]
            time_sel_target = [t in seed_data.times for t in target_data.times]
        else:
            time_sel_seed, time_sel_target = False, False
        
        for bw in bands_to_extract:
            
            if self.connectivity_metric == 'PSI':
                if bw in ['delta', 'alpha']: continue  # no PSI for low frequencies

            values = calculate_connectivity_per_band(
                seed_data=seed_data, target_data=target_data,
                time_sel_seed=time_sel_seed,
                time_sel_target=time_sel_target,
                connectivity_metric=self.connectivity_metric,
                band_name=bw, band_range=bands_to_extract[bw],
                incl_sq_coh=SETTINGS['FEATS_INCL']['sq_coh'],
                incl_imag_coh=SETTINGS['FEATS_INCL']['imag_coh'],
                coh_segment_sec=SETTINGS['FEATS_INCL']['coherence_segment_sec'],
            )

            if self.connectivity_metric == 'COH':
                for COH in ['sq_coh', 'imag_coh']:
                    if SETTINGS['FEATS_INCL'][COH]:
                        fname = f'SSDfeats_{self.sub_SSD.sub}_{self.sources}_{bw}_{COH}.csv'
                        coh_df = DataFrame(index=values['times'],
                                           columns=values['freqs'],
                                           data=values[COH])
                        coh_df.to_csv(join(self.feat_path, fname))
                        print(f'\tCOH saved: {fname} (n-windows, n-freq-bins: {coh_df.shape})')
            
            elif self.connectivity_metric == 'PSI':
                fname = f'SSDfeats_{self.sub_SSD.sub}_{self.sources}_{bw}_PSI.csv'
                psi_df = DataFrame(index=values['times'],
                                columns=['PSI'],
                                data=values['PSI'])
                psi_df.to_csv(join(self.feat_path, fname))
                print(f'\tPSI saved: {fname}')



from mne.filter import resample            
        

def calculate_connectivity_per_band(
    seed_data, target_data,
    connectivity_metric: str,
    band_name: str, band_range: list,
    time_sel_seed = False,
    time_sel_target = False,
    incl_sq_coh: bool = True,
    incl_imag_coh: bool = True,
    coh_segment_sec: float = 0.250,
    DOWN_SAMPLE: bool = False
):
    """
    calculate coherence features between two SSD'd
    timeseries of different locations.

    Results in 2d array per freq-band, per coherence-
    metric with n-windows, n-freq-bins within the defined
    bandwidth, and one array
    with corresponding window timestamps

    Inputs:
        - seed_data, taget_data: SSDs data per source
        - band_name: str with name of freq band
        - band_range: list (or tuple) with lower and 
            upper border of freq band
    
    Returns:
        - coh_values (dict): containing 'times' (list),
            'freqs' (list), 
            'sq_coh' and 'imag_coh' if defined (both 2d array)
    """
    assert seed_data.fs == target_data.fs, (
        'sampling rates should be equal between seed and target'
    )
    fs = seed_data.fs

    seed_windows = getattr(seed_data, band_name).astype(np.float64)
    seed_times = seed_data.times
    target_windows = getattr(target_data, band_name).astype(np.float64)
    target_times = target_data.times

    if time_sel_target: target_windows = target_windows[time_sel_target, :]
    if time_sel_seed: seed_windows = seed_windows[time_sel_seed, :]

    if DOWN_SAMPLE:
        new_fs = 800
        print(f'\t...downsample to {new_fs} Hz')
        seed_windows = resample(seed_windows, up=1, down=fs/new_fs, axis=1)
        target_windows = resample(target_windows, up=1, down=fs/new_fs, axis=1)
        fs = new_fs

    # print('DEBUG: SEED TIMES', len(seed_times), type(seed_times), seed_times)
    # print('DEBUG: target TIMES', len(target_times), type(target_times), target_times)
        
    try:
        assert target_times == seed_times, (
            'times arrays should be equal between seed and target'
        )
    except ValueError:  # not allowing comparison equal length arrays (not understood)
        if (not all([t in target_times for t in seed_times]) or
            not all([t in seed_times for t in target_times])):
            raise ValueError('target_times and seed_times not equal')
    
    # create dict to store values
    values = {'times': []}
    if connectivity_metric == 'COH':
        if incl_sq_coh: values['sq_coh'] = []
        if incl_imag_coh: values['imag_coh'] = []
    elif connectivity_metric == 'PSI':
        values['PSI'] = []

    freq_sel = None
    
    for i_win, (seed_win, target_win) in enumerate(
        zip(seed_windows, target_windows)
    ):
        # check and correct for NaNs
        if np.isnan(seed_win).all() or np.isnan(target_win).all():
            # skip window bcs of NaNs in data
            continue

        elif np.isnan(seed_win).any() or np.isnan(target_win).any():
            idx_nan = np.isnan(seed_win) + np.isnan(target_win)  # bool-array for seed or target is false
            if (sum(idx_nan) / len(idx_nan)) > .66:
                # skip window if too many nan-points in both arrays
                continue

            else:
                seed_win = seed_win[~idx_nan]
                target_win = target_win[~idx_nan]

        # if enough/all data present, and values will be calculated
        values['times'].append(seed_times[i_win])
        
        # COHERENCE EXTRACTION
        if connectivity_metric == 'COH':
            f, _, icoh_abs, _, sq_coh = specFeats.calc_coherence(
                sig1=seed_win,
                sig2=target_win,
                fs=fs,
                nperseg=int(fs * coh_segment_sec),
            )

            if not isinstance(freq_sel, np.ndarray):
                # if freq_sel is still None, create boolean for freq-bin selection
                freq_sel = np.logical_and(f >= band_range[0],
                                        f <= band_range[1])
                values['freqs'] = list(f[freq_sel])

            if incl_imag_coh:
                icoh_abs = icoh_abs[freq_sel]
                values['imag_coh'].append(list(icoh_abs))
            if incl_sq_coh:
                sq_coh = sq_coh[freq_sel]
                values['sq_coh'].append(list(sq_coh))

        elif connectivity_metric == 'PSI':
            psi = get_phase_synchrony_index(sig1=seed_win, sig2=target_win)
            values['PSI'].append(psi)

    
    if connectivity_metric == 'COH':
        values['sq_coh'] = np.array(values['sq_coh'], dtype='object')
        values['imag_coh'] = np.array(values['imag_coh'], dtype='object')
    

    return values


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


