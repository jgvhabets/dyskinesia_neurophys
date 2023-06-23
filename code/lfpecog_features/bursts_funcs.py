"""
Function to calculate neurophysiological
burst-dynamics
"""

# import public packages and functions
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d
from array import array
from sklearn.decomposition import PCA
# import own functions
from lfpecog_features.feats_spectral_features import bandpass
from lfpecog_features.feats_helper_funcs import (
    baseline_zscore, smoothing
)


def get_burst_features(
    sig, fs, burst_freqs, cutoff_perc,
    min_shortBurst_sec, min_longBurst_sec,
    envelop_smoothing=False,
    smooth_factor_ms=125,
):
    """
    Function to extract burst-dynamics from specified
    ephys signal.
    
    Input:
        - sig (arr): one-dimensional ephys-array
        - fs (int): sample freq of sig
        - burst_freqs (list): lower and upper cutoff
            of freq-band to consider for bursts
        - cutoff_perc (int): percentage cutoff for
            burst detection, default 75
        - threshold_meth: method to define thresholds,
            'full-window' means threshold-definition
            over every full considered window again;
            'extremes-on-off' means threshold definition
            on extreme windows at start and end of recording
        - min_short(long)Burst_sec: n seconds a short/
            long burst should be
        - envelop-smoothing (bool): moving average over
            envelop. default in function fs / 8 acc to
            175 ms in Lofredi, Neurobiol of Dis 2019
        - smooth_factor_ms: window of smoothing in milisec
    """
    env = get_envelop(
        sig=sig, fs=fs, bandpass_freqs=burst_freqs,
        in_blocks=True
    )

    if envelop_smoothing:
        env = smoothing(sig, win_ms=smooth_factor_ms, fs=fs)
        # smooth_samples = int(fs / 1000 * smooth_factor_ms)  # smoothing-samples in defined ms-window
        # env = uniform_filter1d(env, smooth_samples)
    
    burst_thr = get_burst_threshold(cutoff_perc, env)


    nShort, nLong, rateShort, rateLong = calc_bursts_from_env(
        envelop=env, burst_thr=burst_thr, fs=fs,
        min_shortBurst_sec=min_shortBurst_sec,
        min_longBurst_sec=min_longBurst_sec,
    )
    

    return nShort, nLong, rateShort, rateLong


def get_burst_threshold(cutoff_perc, envelop,):
    """
    for now: use xx-percentile of 3-minute window

        try-out: xx-percentile based on rest data
        in max med-Off and med-On
    """
    burst_thr = np.nanpercentile(
        list(envelop), cutoff_perc
    )

    return burst_thr


def find_data_blocks(sig):

    nans_bool = np.isnan(list(sig)).astype(int)
    nans_change = np.diff(nans_bool)

    nanStarts = np.where(nans_change == 1)[0]  # add one to correct for diff-index
    nanStops = np.where(nans_change == -1)[0]  # add one to correct for diff-index

    if not np.isnan(sig[0]):

        startBlocks = np.array([0])
        startBlocks = np.concatenate([startBlocks, nanStops])

        stopBlocks = nanStarts
    
    elif np.isnan(sig[0]):

        startBlocks = nanStops
        stopBlocks = nanStarts
    
    if not np.isnan(sig[-1]):
        # if ongoing data at end: add last index to nanStops
        stopBlocks = np.concatenate(
            [stopBlocks, np.array([len(sig) - 1])]  # -1 for indexing
        )
        
    return startBlocks, stopBlocks


def get_envelop(
    sig, fs, bandpass_freqs,
    return_none_abs_Hilb=False,
    in_blocks=False,
    min_sec_block = 10,
    bw_ranges = {
        'alpha': [8, 12],
        'lo_beta': [12, 20],
        'hi_beta': [20, 35],
        'midgamma': [60, 90]
    },
):
    """
    """
    if type(bandpass_freqs) == str:
        bandpass_freqs = bw_ranges[bandpass_freqs]

    # discard nans for hilbert transform
    if not in_blocks:
    # simply removing NaNs and pasting signal toegether
        sig = sig[~np.isnan(list(sig))]
        filt_sig = bandpass(
            sig, bandpass_freqs, fs
        )
        hilb_sig = hilbert(filt_sig)

        if return_none_abs_Hilb: return hilb_sig

        env = abs(hilb_sig)  # rectify -> analytical signal / envelop

        return env

    elif in_blocks:
        # calculating seperate available data blocks
        
        blockStarts, blockEnds = find_data_blocks(sig)

        env = []
        list_of_envs = []
        
        for i1, i2 in zip(blockStarts, blockEnds):

            filt_sig = bandpass(
                sig[i1 + fs:i2 - fs], bandpass_freqs, fs
            )  # discard neighbouring seconds to NaN

            if len(filt_sig) < (min_sec_block * fs):
                # skip too short blocks of data
                continue

            hilb_sig = hilbert(filt_sig)
            
            env.extend(abs(hilb_sig))  # rectify -> analytical signal / envelop
            list_of_envs.append(abs(hilb_sig))
        
        env = np.array(env)

    return env  # return list_of_envs to calculate data-parts seperately


def get_burst_indices(envelop, burst_thr,):
    """
    calculate burst start and end indeices based 
    on envelop and threshold

    Returns:
        - startBursts (array of indices)
        - endBursts (array of indices)
    """
    exceedThr = envelop > burst_thr

    if sum(exceedThr) == 0: return None, None

    changeThr = np.diff(exceedThr.astype(int))
    startBursts = np.where(changeThr == 1)[0] + 1  # add one to correct for diff-index
    endBursts = np.where(changeThr == -1)[0] + 1
    
    # correct for burst at beginning or end
    if exceedThr[0]:
        startBursts = np.concatenate([np.array([0]), startBursts])
    
    if exceedThr[-1]:
        endBursts = np.concatenate([endBursts, np.array([len(exceedThr) - 1])])
    
    return startBursts, endBursts



def calc_bursts_from_env(
    envelop: array,
    fs: int,
    burst_thr: float,
    min_shortBurst_sec: float,
    min_longBurst_sec: float,
):
    """
    Consider to add burst-amplitudes

    
    """
    # determine resulting burst-settings
    min_shortBurst_samples = min_shortBurst_sec * fs
    min_longBurst_samples = min_longBurst_sec * fs

    # define burst-starts and - ends
    if isinstance(envelop, list):
        burstIndices = []
        # create list with tuples
        for env in envelop:
            burstIndices.append(get_burst_indices(
                env, burst_thr,
            ))

    else:
        # if one envelop -> create tuple with start- and end-indices
        burst_starts, burst_ends = get_burst_indices(envelop, burst_thr)

    if not isinstance(burst_ends, np.ndarray):
        return 0, 0, 0, 0, []
    
    # calculate length all bursts (in samples)
    burst_lengths = burst_ends - burst_starts

    # count number of short and long bursts as defined
    n_short = sum(np.logical_and(
        min_shortBurst_samples < burst_lengths,
        burst_lengths < min_longBurst_samples
    ))
    n_long = sum(burst_lengths > min_longBurst_samples)

    rateShort = n_short / (len(envelop) / fs)
    rateLong = n_long / (len(envelop) / fs)

    return n_short, n_long, rateShort, rateLong, burst_lengths


def get_burst_duration_profile():
    """
    Calculate average burst duration profiles, as found
    to be a predictive marker for medication-state,
    INDEPENDENT OF ARBITRARY BURST-THRESHOLDS, by
    Duchet et al, PLOS Comp Biol 2021
    (https://doi.org/10.1371/journal.pcbi.1009116).

    """