"""
Function to load SSD'd features for
analysis
"""

# import public packages
import os
from os.path import join, exists
from os import listdir, makedirs
import numpy as np
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass, field

# import custom functions
from utils.utils_fileManagement import (
    load_ft_ext_cfg, get_project_path,
)
from lfpecog_preproc.preproc_import_scores_annotations import (
    read_clinical_scores
)


@dataclass(init=True, repr=True)
class ssdFeatures:
    """
    Main class to load SSD'd Features

    Arguments:
        - settings_json: filename of json file,
            location should be defined in utils_fileManagement
        - sub_list: if defined, these subjects are
            included; otherwise all available subjects are included
        - data_Version: defaults to 'v3.0' 
        - win_len_sec: window length of features in seconds
        - win_overlap_part: part of ft-window overlap
        - incl_powers: include SSD power spectrum values
        - incl_localPAC: include SSD PAC values
        - incl_coherence: include SSD Coherences values
    """
    settings_json: str = 'ftExtr_spectral_v1.json'
    sub_list: list = field(default_factory=lambda: [])
    data_version: str = 'v3.0'
    win_len_sec: int or float = 10
    win_overlap_part: float = 0.0
    incl_powers: bool = True
    incl_localPAC: bool = True
    incl_coherence: bool = True

    def __post_init__(self,):
        # load feature extraction settings
        extract_settings = load_ft_ext_cfg(cfg_fname=self.settings_json)

        # define feature path and check existence
        self.feat_path = join(get_project_path('results'),
                    'features',
                    'SSD_feats',
                    self.data_version,
                    f'windows_{self.win_len_sec}s_'
                    f'{self.win_overlap_part}overlap')
        assert exists(self.feat_path), f'feat_path ([{self.feat_path}]) does not exist'
        # take all available subjects from feature path if sub_list is not defined
        if self.sub_list == []:
            self.sub_list = list(set([name.split('_')[1]
                                      for name in os.listdir(self.feat_path)]))

        keywords = vars(self)

        for sub in self.sub_list:
            print(f'\nload SSDd features for sub-{sub}')
            setattr(self,
                    f'sub{sub}',
                    ssdFeats_perSubject(sub=sub, feat_path=self.feat_path,
                                        settings=keywords,
                                        extract_settings=extract_settings),)
            


@dataclass(init=True, repr=True)
class ssdFeats_perSubject:
    sub: str = 'default'  # default given to prevent inheritance error
    feat_path: str = 'default'
    settings: dict = field(default_factory=lambda: {})
    extract_settings: dict = field(default_factory=lambda: {})
    verbose: bool = False

    def __post_init__(self,):
        # import CDRS scores and times
        scores = read_clinical_scores(sub=self.sub, rater='Patricia')
        # add scores as namedtuple (times, left, right, total)
        self.scores = CDRS_scores(
            scores['dopa_time'].values,
            scores['CDRS_total_left'].values,
            scores['CDRS_total_right'].values,
            scores['CDRS_total'].values
        )
        
        if self.settings['incl_powers']:
            if self.verbose: print(f'load POWERS - {self.sub}')
            self.powers = load_ssd_powers(self.sub, feat_path=self.feat_path)
        
        if self.settings['incl_localPAC']:
            if self.verbose: print(f'load local PAC - {self.sub}')
            pac_freqs = self.extract_settings['FEATS_INCL']['local_PAC_freqs']
            self.localPAC = load_ssd_localPAC(self.sub, feat_path=self.feat_path,
                                              pac_freqs=pac_freqs,
                                              extr_settings=self.extract_settings)
        
        if self.settings['incl_coherence']:
            if self.verbose: print(f'TODO: load COHERENCES - {self.sub}')
            self.coherences = load_ssd_coherences(
                self.sub,
                feat_path=self.feat_path,
                bandwidths=self.extract_settings['SPECTRAL_BANDS'].keys()
            )

        

CDRS_scores = namedtuple('CDRS_scores', 'times left right total')

localPAC = namedtuple('localPAC', 'values times pha_bins amp_bins')


def load_ssd_powers(sub, feat_path):

    sub_ft_files = [f for f in os.listdir(feat_path) if
                    sub in f and 'spectralFeatures' in f]
    df_out = None
    for f in sub_ft_files:
        for dType in ['lfp_left', 'lfp_right',
                      'ecog_left', 'ecog_right']:
            if dType not in f: continue

            temp = pd.read_csv(join(feat_path, f), header=0, index_col=0)
            
            temp = temp.rename(columns={k: f'{dType}_{k}'
                                        for k in temp.keys()})

            if not isinstance(df_out, pd.DataFrame):
                df_out = temp
                continue
            # add to existing df_out
            df_out = pd.concat([df_out, temp], ignore_index=False, axis=1,)


    return df_out


def load_ssd_localPAC(
    sub, feat_path, pac_freqs, extr_settings
):
    """
    
    Returns:
        - dict with PAC results per SOURCE_PHA_AMP
            within dict is a namedtuple containing:
                - values: 3d ndarray, (n-amp-bins,
                    n-pha-bins, n-windows)
                - times: list w/ start times from windows
                - pha_bins: list w/ tuples of phase bin ranges
                - amp_bins: list w/ tuples of ampl bin ranges
    """

    sub_ft_files = [f for f in os.listdir(feat_path) if
                    sub in f and 'localPAC' in f]
    dict_out = {}

    for dType in ['lfp_left', 'lfp_right',
                    'ecog_left', 'ecog_right']:
        
        dtype_files = [f for f in sub_ft_files if dType in f]
        if len(dtype_files) == 0: continue  # skip for i.e. not-existing ecog side 
        
        for pha_f, amp_f in pac_freqs:
        
            for f in dtype_files:
                if f'{pha_f}_{amp_f}' not in f: continue
        
                if 'times' in f: 
                    times = np.loadtxt(join(feat_path, f), delimiter=',')
                else:
                    dat = np.load(join(feat_path, f), allow_pickle=True)
                # get pac bins for phase and ampl
                pha_bins = get_pac_bins(
                    freq_range=extr_settings['SPECTRAL_BANDS'][pha_f],
                    binwidth=extr_settings['FEATS_INCL']['PAC_binwidths']['phase']
                )
                amp_bins = get_pac_bins(
                    freq_range=extr_settings['SPECTRAL_BANDS'][amp_f],
                    binwidth=extr_settings['FEATS_INCL']['PAC_binwidths']['ampl']
                )
            assert len(times) == dat.shape[-1], (
                f'loaded PACs times ({len(times)}) and data ({dat.shape})'
                f' dont match for {dType}_{pha_f}_{amp_f}'
            )
            
            dict_out[f'{dType}_{pha_f}_{amp_f}'] = localPAC(
                values=dat,
                times=times,
                pha_bins=pha_bins,
                amp_bins=amp_bins
            )


            del(times, dat)

    return dict_out


from lfpecog_features.feats_phases import get_pac_bins
from itertools import product

Coh_sources = namedtuple('Coh_sources', 'STN_STN STN_ECOG')

def load_ssd_coherences(
    sub,
    feat_path,
    incl_imag_coh: bool = True,
    incl_sq_coh: bool = True,
    bandwidths: list = ['alpha', 'lo_beta',
                        'hi_beta', 'narrow_gamma',
                        'broad_gamma'],
):
    scores_to_incl = []
    if incl_sq_coh: scores_to_incl.append('sq_coh')
    if incl_imag_coh: scores_to_incl.append('imag_coh')
    if len(scores_to_incl) == 0: return None

    Coherences = namedtuple('Coherences', scores_to_incl)
    Bandwidths = namedtuple('Bandwidths', bandwidths)

    COHs_per_source = {}
    for source in ['STN_STN', 'STN_ECOG']:
        COHs_per_bw = {}
        for bw in bandwidths:

            sel_files = [f for f in os.listdir(feat_path) if
                        sub in f and source in f and bw in f]
            
            coh_dfs = {}
            for score, f in product(scores_to_incl, sel_files):
                # check if file contains sq_coh or imag_coh
                if score not in f: continue  # skip incorrect file
                # if score in file, load csv to df
                coh_dfs[score] = pd.read_csv(
                    join(feat_path, f),
                    header=0, index_col=0)

            # coh_dfs contains now all scores to incl
            # ._make needed to insert a list in the namedtuple
            COHs_per_bw[bw] = Coherences._make([coh_dfs[s] for s
                                                in scores_to_incl])  # ensures the correct order    
        # all bandwidths included
        COHs_per_source[source] = Bandwidths._make([COHs_per_bw[bw] for bw
                                                    in bandwidths])
    
    COHs_sub = Coh_sources(COHs_per_source['STN_STN'],
                           COHs_per_source['STN_ECOG'])
    
    return COHs_sub