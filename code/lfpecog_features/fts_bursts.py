"""
Function to calculate neurophysiological
burst-dynamics
"""

# import public packages and functions
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# import own functions
from lfpecog_features.feats_spectral_features import bandpass


def get_burst_features(
    sig, fs, burst_freqs, cutoff_perc,
    threshold_meth,
    min_shortBurst_sec, min_longBurst_sec,
):
    """
    PM: consider envelop-smoothing (in function)
        (e.g. kernel of +/- 175 ms (Lofredi 2019))
    """
    env = get_envelop(
        sig=sig, fs=fs, bandpass_freqs=burst_freqs,
        in_blocks=True
    )
    
    burst_thr = get_burst_threshold(
        threshold_meth, cutoff_perc, env
    )

    nShort, nLong, rateShort, rateLong = calc_bursts_from_env(
        envelop=env, burst_thr=burst_thr, fs=fs,
        min_shortBurst_sec=min_shortBurst_sec,
        min_longBurst_sec=min_longBurst_sec,
    )
    

    return nShort, nLong, rateShort, rateLong


def get_burst_threshold(
    threshold_meth, cutoff_perc,
    envelop,
):
    """
    for now: use xx-percentile of 3-minute window

        try-out: xx-percentile based on rest data
        in max med-Off and med-On
    """
    if threshold_meth == 'full-window':

        burst_thr = np.nanpercentile(
            list(envelop), cutoff_perc
        )

    return burst_thr


def find_data_blocks(sig):

    nans_bool = np.isnan(list(sig)).astype(int)
    nans_change = np.diff(nans_bool)

    nanStarts = np.where(nans_change == 1)[0]  # add one to correct for diff-index
    nanStops = np.where(nans_change == -1)[0]  # add one to correct for diff-index

    if np.isnan(sig[-1]): nanStarts = nanStarts[:-1]  # leave last nan-start out if it didnot end in data

    if not np.isnan(sig[0]):

        startBlocks = np.array([0])
        startBlocks = np.concatenate([startBlocks, nanStops])

        stopBlocks = nanStarts

        if not np.isnan(sig[-1]):
            
            stopBlocks = np.concatenate([stopBlocks, np.array([len(sig - 1)])])
    
    elif np.isnan(sig[0]):

        startBlocks = nanStops
        stopBlocks = nanStarts
    
    return startBlocks, stopBlocks


def get_envelop(
    sig, fs, bandpass_freqs,
    in_blocks=False,
):
    """
    """
    # remove nan's
    print(
        'NaN percentage: ',
        sum(np.isnan(list(sig))) / len(sig),
        ' %'
    )
    if not in_blocks:
    # simply removing NaNs and pasting signal toegether
        sig = sig[~np.isnan(list(sig))]
        filt_sig = bandpass(
            sig, bandpass_freqs, fs
        )
        hilb_sig = hilbert(filt_sig)
        env = abs(hilb_sig)  # rectify -> analytical signal / envelop

    elif in_blocks:
        # calculating seperate available data blocks
        blockStarts, blockEnds = find_data_blocks(sig)
        env = []
        
        for i1, i2 in zip(blockStarts, blockEnds):

            filt_sig = bandpass(
                sig[i1:i2], bandpass_freqs, fs
            )
            hilb_sig = hilbert(filt_sig)
            env.extend(abs(hilb_sig))  # rectify -> analytical signal / envelop
        
        env = np.array(env)

    return env

def calc_bursts_from_env(
    envelop, fs, burst_thr,
    min_shortBurst_sec, min_longBurst_sec,
    
):
    """
    Consider to add burst-amplitudes

    
    """
    # determine resulting burst-settings
    min_shortBurst_samples = min_shortBurst_sec * fs
    min_longBurst_samples = min_longBurst_sec * fs

    overThr = envelop > burst_thr
    
    # define burst-lengths
    changeThr = np.diff(overThr.astype(int))
    startBursts = np.where(changeThr == 1)[0] + 1  # add one to correct for diff-index
    endBursts = np.where(changeThr == -1)[0] + 1
    # correct for ongoing bursts at beginning or end of signal
    if startBursts[0] > endBursts[0]: endBursts = endBursts[1:]
    if len(startBursts) > len(endBursts): startBursts = startBursts[:-1]

    burstSampleLengths = endBursts - startBursts
    # count number of short and long bursts as defined
    nShort = sum(np.logical_and(
        min_shortBurst_samples < burstSampleLengths,
        burstSampleLengths < min_longBurst_samples
    ))
    nLong = sum(burstSampleLengths > min_longBurst_samples)

    rateShort = nShort / (len(envelop) / fs)
    rateLong = nLong / (len(envelop) / fs)

    return nShort, nLong, rateShort, rateLong