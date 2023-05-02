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
    ephys_sources: list
    feat_path: str
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.settings_dict    
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD powers')
            data = getattr(self.sub_SSD, dType)  # assign datatype data
            # check if features already exist
            feat_fname = f'SSDfeats_{self.sub}_{dType}_local_spectralFeatures.csv'

            if exists(join(self.feat_path, feat_fname)) and not self.overwrite_features:
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
            feats_out.to_csv(join(self.feat_path, feat_fname),
                             index=True, header=True,)
            print(f'Saved FEATURES for sub-{self.sub} {dType} as '
                  f'{feat_fname} in {self.feat_path}')


from lfpecog_features.feats_phases import calculate_PAC_matrix

@dataclass(init=True, repr=True,)
class extract_local_SSD_PACs:
    """
    Extracting local PAC based on SSD'd freq-bands
    stored per window

    Called from run_ssd_ftExtr
    """
    sub: str
    sub_SSD: str
    settings_dict: dict
    ephys_sources: list
    feat_path: int
    incl_ecog: bool = True
    incl_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        SETTINGS = self.settings_dict    
        
        # loop over possible datatypes
        for dType in self.ephys_sources:
            print(f'\n\tstart ({dType}) extracting local SSD PACs')
            data = getattr(self.sub_SSD, dType)  # assign datatype data
            # data is here class with per bandwidth a 2d array (n-windows, n-samples)

            # define PAC-frequencies to extract
            pac_bands = SETTINGS['FEATS_INCL']['local_PAC_freqs']  # get features to include

            # load SSD'd data
            for [pha, ampl] in pac_bands:
                print(f'\n\t...START PAC {pha} {ampl}')
                # check if features already exist
                feat_fname = (f'SSDfeats_{self.sub}_{dType}_localPAC'
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




@dataclass(init=True, repr=True, )
class extract_SSD_coherences:
    """
    Extract of Connectivity features per segments

    Called from run_ssd_ftExtr
    """
    sub: str
    sub_SSD: str
    sources: str
    settings_dict: dict
    ephys_sources: list
    feat_path: int
    incl_stn_ecog: bool = True
    incl_stn_stn: bool = True
    overwrite_features: bool = False
    
    def __post_init__(self,):
        assert self.sources.upper() in ['STN_ECOG', 'STN_STN']

        SETTINGS = self.settings_dict    

        bands_to_extract = SETTINGS['SPECTRAL_BANDS']

        if self.sources == 'STN_STN':
            seed_data = self.sub_SSD.lfp_left
            target_data = self.sub_SSD.lfp_right
        
        elif self.sources == 'STN_ECOG':
            ecog_source = [s for s in self.ephys_sources if 'ecog' in s][0]
            stn_ecog_side = f"lfp_{ecog_source.split('_')[1]}"
            seed_data = getattr(self.sub_SSD, stn_ecog_side)
            target_data = getattr(self.sub_SSD, ecog_source)

        for bw in bands_to_extract:
                
            coh_values = calculate_coherence_per_band(
                seed_data=seed_data, target_data=target_data,
                band_name=bw, band_range=bands_to_extract[bw],
                incl_sq_coh=SETTINGS['FEATS_INCL']['sq_coh'],
                incl_imag_coh=SETTINGS['FEATS_INCL']['imag_coh'],
                coh_segment_sec=SETTINGS['FEATS_INCL']['coherence_segment_sec'],
            )
            
            for COH in ['sq_coh', 'imag_coh']:

                if SETTINGS['FEATS_INCL'][COH]:
                    fname = f'SSDfeats_{self.sub}_{self.sources}_{bw}_{COH}.csv'
                    coh_df = DataFrame(index=coh_values['times'],
                                       columns=coh_values['freqs'],
                                       data=coh_values[COH])
                    coh_df.to_csv(join(self.feat_path, fname))
                    print(f'\tCOH saved: {fname} (n-windows, n-freq-bins: {coh_df.shape})')

            
from mne.filter import resample            
        

def calculate_coherence_per_band(
    seed_data, target_data,
    band_name: str, band_range: list,
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

    if DOWN_SAMPLE:
        new_fs = 800
        print(f'\t...downsample to {new_fs} Hz')
        seed_windows = resample(seed_windows, up=1, down=fs/new_fs, axis=1)
        target_windows = resample(target_windows, up=1, down=fs/new_fs, axis=1)
        fs = new_fs

    assert target_times == seed_times, (
        'times arrays should be equal between seed and target'
    )
    
    coh_values = {'times': []}
    if incl_sq_coh: coh_values['sq_coh'] = []
    if incl_imag_coh: coh_values['imag_coh'] = []

    freq_sel = None
    
    for i_win, (seed_win, target_win) in enumerate(
        zip(seed_windows, target_windows)
    ):

        if np.isnan(seed_win).any() or np.isnan(target_win).any():
            # skip window bcs of NaNs in data
            continue

        # if all data present
        coh_values['times'].append(seed_times[i_win])
        
        # COHERENCE EXTRACTION
        # print(f'fs: {fs}, seg: {coh_segment_sec}; nperseg IN: {int(fs * coh_segment_sec)}')
        f, _, icoh_abs, _, sq_coh = specFeats.calc_coherence(
            sig1=seed_win,
            sig2=target_win,
            fs=fs,
            nperseg=int(fs * coh_segment_sec),
        )

        if not isinstance(freq_sel, np.ndarray):
            # if freq_sel is still None, create boolean for freq-bin selection
            freq_sel = np.logical_and(f > band_range[0],
                                      f < band_range[1])
            coh_values['freqs'] = list(f[freq_sel])

        if incl_imag_coh:
            icoh_abs = icoh_abs[freq_sel]
            coh_values['imag_coh'].append(list(icoh_abs))
        if incl_sq_coh:
            sq_coh = sq_coh[freq_sel]
            coh_values['sq_coh'].append(list(sq_coh))
    
    coh_values['sq_coh'] = np.array(coh_values['sq_coh'], dtype='object')
    coh_values['imag_coh'] = np.array(coh_values['imag_coh'], dtype='object')

    return coh_values


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


