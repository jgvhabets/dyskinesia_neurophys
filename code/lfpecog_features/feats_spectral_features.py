'''
Functions to calculate bandwidth-based
spectral features from neurophysiology
data (LFP and ECOG) in ReTune's B04 Dyskinesia Project

Containing:
- bandpass filter
- coherence
- Phase Amplitude Coupling (tensorpac)

PM: consider seperate .py-files per feature
'''
# Import general packages and functions
import os
import numpy as np
from scipy import signal
# import own functions
import lfpecog_features.feats_spectral_helpers as specHelpers


def bandpass(sig, freqs, fs, order=3,):
    """
    Bandpass filtering using FIR window (based on scipy
    firwin bandpas filter documentation)
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html?highlight=bandpass%20firwin
    
    Input:
        - sig: neurophys-signal to filter
        - freqs: list or tuple with lower and upper freq
        - fs: sampling freq
        - order (int): n-order of filter
        mentation -> equal to filtfilt, no direct difference
        in test data
        
        Returns:
        - filtsig (arr): filtered signal
    """
    # butter_bandpass
    nyq = 0.5 * fs
    low = freqs[0] / nyq
    high = freqs[1] / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtsig = signal.lfilter(b, a, sig)

    return filtsig


def calc_coherence(
    sig1,
    sig2,
    fs: int,
    use_rel_powers: bool = False,
    nperseg=None,
):
    """
    Coherence, calculated per bandwidth frequencies based on
    Qasim et al Neurobiol Dis 2016/ De Hemptinne 2015.
    Imaginary coherence ignores signals with zero-time lag,
    potentially caused by volume conduction theory (the coherent
    signal is coming from an identical physiological source
    instead of a parallel coherent-oscillating source).
    Using the imag-coherence reduces the probability of high
    coherence values due to non-physiological correlations
    (based on Nolte ea, Clin Neurophy 2004; Hohlefeld ea,
    Neuroscience 2015; Rolston & Chang, Cereb Cortex 2018).

    PM1: Coherence function works with large windows (+/- 1 min),
    despite of the defined shorter segments within the different
    parts of the formula, the formula doesnt produce reasonable
    values if its performed on windows of 0.5 sec e.g.
    Averaging the COH values afterwards, does NOT give comparable
    values to inputting larger windows, which are averaged
    within the Welch and CSD computations.

    PM2: Standardisation (detectability) are calculated outside
    of this function.

    PM3: imag-coherency assumption no phase shift is proven
    for freq's below 100 Hz (Stinstra & Peters, 1998; referred
    to in Nolte 2004).

    Input:
        - sig1-2: two arrays with signals to include
        - fs: sample frequency (should be same for sig1-2)
        - nseg= length of welch window

    Returns:
        - f: array of frequencies corr to coherence-values
        - icoh: array of imaginary coherence values
        - icoh: array of abs-values of imaginary coherence
        - coh: array of coherence values
        - sq_coh: array of squared-coherence values
    """
    if nperseg == None: nperseg = fs // 2
    # check if signals are in np.float64 dtype, if not psd output raises errors 
    if np.logical_or(
        type(sig1[0]) != np.float64, type(sig2[0]) != np.float64 
    ):
        raise ValueError(
            'Not both signals contain np.float64:'
            f'sig1 is {type(sig1[0])}, sig2 is {type(sig2[0])}'
        )
    
    # calculate power spectra (these power spectra are not stored)
    f, S_xx = signal.welch(sig1, fs=fs, nperseg=nperseg,)
    _, S_yy = signal.welch(sig2, fs=fs, nperseg=nperseg,)
    _, S_xy = signal.csd(sig1, sig2, fs=fs, nperseg=nperseg,)

    if use_rel_powers:  # TODO test rel powers
        S_xx = specHelpers.relative_power(S_xx)
        S_yy = specHelpers.relative_power(S_yy)
        S_xy = specHelpers.relative_power(S_xy)

    # calculate coherencies (Nolte ea 2004)
    coherency = S_xy / np.sqrt(S_xx * S_yy)

    coh = coherency.real  # take real part for coherence
    sq_coh = S_xy.real**2 / (S_xx * S_yy)  # squared coherence, used by Gilron 2021
    icoh = np.imag(coherency)  # take imaginary (pos and neg)
    abs_icoh = abs(icoh)  # take absolute value

    # get rid of 3rd dimensionality
    if len(coh.shape) == 3: coh = coh[:, 0, :]
    if len(icoh.shape) == 3: icoh = icoh[:, 0, :]
    if len(abs_icoh.shape) == 3: abs_icoh = abs_icoh[:, 0, :]
    if len(sq_coh.shape) == 3: sq_coh = sq_coh[:, 0, :]

    """
    PM: implement ICOH detectable according to
    https://link.springer.com/article/10.1007/s10548-018-0640-0
    """

    return f, icoh, abs_icoh, coh, sq_coh


def get_fooof_peaks_freqs_and_sizes(
    f, pxx, range=[4, 98], knee_or_fix: str = 'knee',
):
    """
    get relevant fooof-parameters

    content orignal foof-params:
    - fm.get_results().error  # error of the fit
    - fm.get_results().r_squared  # r^2 (goodness) of fit
    - fm.get_results().aperiodic_params  # offset, knee, exponent OR offset, exponent
    - fm.get_results().peak_params  # 2d array with pro row one peak's [mid-f, power, bandwidth]
    - fm.get_results().gaussian_params  # 2d array with Gaussian fits of peaks [mid-f, height, sd]

    Returns:
        - ap_off
        - ap_exp
        - peak_cf: center freqs
        - peak_size: height gaussian * bandwidth
    """
    assert knee_or_fix in ['knee', 'fix'], (
        'variable knee_or_fix (fooof) incorrect'
    )
    
    from fooof import FOOOF
    
    fm = FOOOF(
        peak_width_limits=(.5, 5),
        peak_threshold=.5,
        aperiodic_mode=knee_or_fix,
        verbose=False,
    )     
    fm.fit(f, pxx, range)

    ap_off = fm.get_results().aperiodic_params[0]
    ap_exp = fm.get_results().aperiodic_params[-1]
    temp_peaks = fm.get_results().peak_params
    temp_gaus = fm.get_results().gaussian_params
    peak_freqs = temp_peaks[:, 0]
    peak_sizes = temp_peaks[:, 2] * temp_gaus[:, 1]  # height * bandwidth
    # peak_pows = temp_peaks[:, 1]

    return ap_off, ap_exp, peak_freqs, peak_sizes