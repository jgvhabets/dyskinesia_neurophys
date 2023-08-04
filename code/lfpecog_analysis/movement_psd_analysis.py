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
    Fs = 2048

    # get available SUBS
    files = os.listdir(results_path)
    SUBS = np.unique([f.split('_')[0] for f in files])

    print(f'available subjects: {SUBS}')

    freq_arr = np.arange(4, 91)

    total_psds = {'rest': [], 'tap': []}

    for sub in SUBS:
        # define lateralities
        for lfp_side, acc_side in zip(['left', 'right',],
                                    ['right', 'left']):
            # empty psd array to fill
            sub_lfp_psds = {'rest': np.array([np.nan] * len(freq_arr)),
                            'tap': np.array([np.nan] * len(freq_arr))}

            # get lfp data
            rest_lfp = np.load(os.path.join(results_path,
                                       f'{sub}_lfp_{lfp_side}_restSigs.npy'))
            tap_lfp = np.load(os.path.join(results_path,
                                       f'{sub}_lfp_{lfp_side}_tapSigs.npy'))

            # calculate psds per band
            for i_bw, bw in enumerate(['delta', 'alpha', 'lo_beta',
                                        'hi_beta', 'gamma']):
                # take first 3 rest-minutes as baseline
                f, ps_base = welch(rest_lfp[i_bw, :Fs * 60 * 3], fs=Fs, nperseg=Fs)
                f, ps_rest = welch(rest_lfp[i_bw], fs=Fs, nperseg=Fs)
                f, ps_tap =  welch(tap_lfp[i_bw], fs=Fs, nperseg=Fs)

                ps_rest = (ps_rest - ps_base) / ps_base * 100
                ps_tap = (ps_tap - ps_base) / ps_base * 100
                
                psx_f_sel = np.logical_and(f > SETTINGS['SPECTRAL_BANDS'][bw][0],
                                           f < SETTINGS['SPECTRAL_BANDS'][bw][1])
                arr_f_sel = np.logical_and(freq_arr > SETTINGS['SPECTRAL_BANDS'][bw][0],
                                           freq_arr < SETTINGS['SPECTRAL_BANDS'][bw][1])
                sub_lfp_psds['rest'][arr_f_sel] = ps_rest[psx_f_sel]
                sub_lfp_psds['tap'][arr_f_sel] = ps_tap[psx_f_sel]
            
            for task in ['rest', 'tap']: total_psds[task].append(sub_lfp_psds[task])
            
            # for key in sub_lfp_psds.keys():
            #     plt.plot(freq_arr, sub_lfp_psds[key], label=key)
            
            # plt.legend(loc='upper left')
            # plt.title(f'Sub-{sub}, LFP-{lfp_side}')
            # plt.show()
    
    PSD_REST = np.array(total_psds['rest'])
    PSD_TAP = np.array(total_psds['tap'])

    # print(PSD_REST.shape, PSD_TAP.shape)

    # plt.plot(freq_arr, PSD_REST.mean(axis=0), label='REST')
    # plt.plot(freq_arr, PSD_TAP.mean(axis=0), label='TAP')

    # plt.show()

    return PSD_REST, PSD_TAP, freq_arr
                
                
                # TODO:
                # create mmerged PSDs from all 5 freq-band
                # take baseline %-changes vs first 5 rest minutes
                # plot rest vs tap, mean + std-error
