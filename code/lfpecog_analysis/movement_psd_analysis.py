"""
Analyze movement voluntary effect
"""
# import functions and packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch

# import own functions
import lfpecog_analysis.load_SSD_features as load_ssd_fts
import lfpecog_features.get_ssd_data as ssd
import lfpecog_analysis.get_SSD_timefreqs as ssd_TimeFreq
import utils.utils_fileManagement as utilsFiles
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle,
                                        load_ft_ext_cfg)




def load_movement_psds(data_version='v4.0', ft_version='v4',):

    # load ft settings
    SETTINGS = load_ft_ext_cfg(f'ftExtr_spectral_{ft_version}.json')
    freqs = SETTINGS['SPECTRAL_BANDS']
    winlen = SETTINGS['WIN_LEN_sec']
    winoverlap = SETTINGS['WIN_OVERLAP_part']
    # define main directory with stored merged data
    results_path = os.path.join(get_project_path('results'), 'features',
                                  'SSD_feats_broad', data_version,
                                  f'windows_{winlen}s_{winoverlap}overlap_tapRest')
    
    # get available SUBS
    files = os.listdir(results_path)
    SUBS = np.unique([f.split('_')[0] for f in files])

    print(f'available subjects: {SUBS}')

    for sub in SUBS:
        # define lateralities
        for lfp_side, acc_side in zip(['left', 'right',],
                                    ['right', 'left']):
        
            # get lfp data
            rest_lfp = np.load(os.path.join(results_path,
                                       f'{sub}_lfp_{lfp_side}_restSigs.npy'))
            tap_lfp = np.load(os.path.join(results_path,
                                       f'{sub}_lfp_{lfp_side}_tapSigs.npy'))
            print(f'{sub}, lfp: {lfp_side}',
                  rest_lfp.shape, tap_lfp.shape)

            # create 2d array with bands
            for i_bw, bw in enumerate(['delta', 'alpha', 'lo_beta',
                                    'hi_beta', 'gamma']):
                continue

                # TODO:
                # create mmerged PSDs from all 5 freq-band
                # take baseline %-changes vs first 5 rest minutes
                # plot rest vs tap, mean + std-error
