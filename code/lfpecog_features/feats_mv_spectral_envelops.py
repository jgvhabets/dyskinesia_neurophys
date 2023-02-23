'''
Functions to calculate freq-bandwidth specific
spectral power and bursts

Uses either PCA of analytic signals or anatomically
predefined electrode levels to include
'''
# Import general packages and functions
import os
import numpy as np
from scipy import signal
from scipy.stats import variation


# import own functions
import lfpecog_features.feats_helper_funcs as ftHelpers
import lfpecog_features.bursts_funcs as bursts_funcs


def get_multivariate_spectral_features(
    data_array, fs, ch_names,
    source_of_interest,
    n_top_comps=3,
    baseline_array=None,
    metrics_of_interest=['power', 'burst'],
    bw_of_interest=['alpha', 'lo_beta', 'hi_beta', 'midgamma'],
    bw_ranges = {   'alpha': [8, 12],
                    'lo_beta': [12, 20],
                    'hi_beta': [20, 35],
                    'midgamma': [60, 90]},
):
    result_dict = {}
    
    for bw in bw_of_interest:
        result_dict[bw] = {}
        
        # get baseline envelop mean and std dev
        if baseline_array:
            zscore_env = True
            bl_m, bl_sd = get_baseline_env_mean_std(
                baseline_array, fs=fs, ch_names=ch_names,
                base_source=source_of_interest.split('_')[0], base_bw=bw)
        else:
            zscore_env = False
            bl_m, bl_sd = None, None
       
        for i, sig in (enumerate):
            # get explained variance and pca env-components
            exp_var, sig_pca = bursts_funcs.get_multichannel_analyticSignal(
                dat=sig, fs=fs, bw_sel=bw, ch_names=ch_names,
                source_sel=source_of_interest, return_n_comp=None,
                zscore_env=zscore_env, zscore_m=bl_m, zscore_sd=bl_sd,
            )
            # print(f'expl variance, {bw}: ({sum(exp_var)}) {exp_var}')
            if 'power' in metrics_of_interest:
                comp_rms = np.around(np.sqrt(np.mean(sig_pca ** 2, axis=1)), 6)  # RMS per component
                rms_cvar_all = variation(comp_rms)  # coef of variation of RMS of all components
                rms_mean_all = np.mean(comp_rms)  # mean of RMS of all components
                rms_mean_top = np.mean(comp_rms[:n_top_comps])  # mean of RMS of top components

            if 'burst' in metrics_of_interest:
                # compute bursts over 
                burst_pca = abs(sig_pca)  # take absolute value
            
            



    return



def get_baseline_env_mean_std(
    base_array, fs, ch_names, base_source, base_bw,
):
    """
    Input:
        - base_array: n-channels x n-samples
        - fs
        - ch_names
        - base_source: LFP or ECOG to include
        - base_bw: alpha, lo_beta, hi_beta, midgamma,
            or tuple with lower and higher cut off
    
    Returns:
        - mean and std
    """
    assert base_source.upper() in ['LFP', 'ECOG'], (
        'base_source should be LFP or ECOG'
    )
    env_means, env_sds = [], []
    for i_ch, ch in enumerate(ch_names):
        if not ch.startswith(base_source.upper()): continue

        env = bursts_funcs.get_envelop(base_array[i_ch, :],  fs=fs,
                                       bandpass_freqs=base_bw,)
        env_means.append(np.mean(env))
        env_sds.append(np.std(env))

    return np.mean(env_means), np.mean(env_sds)

    