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
from pandas import DataFrame
from dataclasses import dataclass
from scipy import signal
from scipy.stats import variation
import matplotlib.pyplot as plt
# import own functions
import lfpecog_features.feats_spectral_helpers as specHelpers
from lfpecog_features.bursts_funcs import get_envelop
from lfpecog_features.feats_helper_funcs import smoothing

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


@dataclass
class Spectralfunctions():
    """
    Input:
        - f (freqs corr to psd)
        - psd: spectral pwoers from welch
        - ssd_sig: time series, possibly unfiltered depending
            on feature version
        - f_sel: bool array which frequencies fall in current bw-range
        - f_range: freq range start and stop
        - s_freq: sampling freq
        - SSD_FILTERED: bool whether ssd signal is bandpass filtered
    """
    f: np.ndarray = np.array([])
    psd: np.ndarray = np.array([])
    ssd_sig: np.ndarray = np.array([])
    f_sel: np.ndarray = np.array([])
    f_range: np.ndarray = np.array([])
    sfreq: int = 0
    SSD_FILTERED: bool = True

    def get_SSD_max_psd(self):
        max_peak = np.max(self.psd[self.f_sel])
        return max_peak

    def get_SSD_peak_freq(self):
        i_peak = np.argmax(self.psd[self.f_sel])
        f_peak = self.f[self.f_sel][i_peak]
        return f_peak

    def get_SSD_mean_psd(self):
        mean_peak = np.mean(self.psd[self.f_sel])
        return mean_peak

    def get_SSD_variation(self):
        if not self.SSD_FILTERED:
            filt_sig = bandpass(self.ssd_sig,
                                freqs=self.f_range,
                                fs=self.sfreq)
        else: filt_sig = self.ssd_sig
        env = abs(signal.hilbert(filt_sig))
        cv_signal = variation(env)
        return cv_signal
    

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
    try:
        if np.logical_or(
            type(sig1[0]) != np.float64, type(sig2[0]) != np.float64 
        ):
            sig1 = sig1.astype(np.float64)
            sig2 = sig2.astype(np.float64)

    except IndexError:
        print(type(sig1), sig1)
        print(type(sig2), sig2)
    
    except:
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

    # coh = coherency.copy().real  # take real part for coherence
    coh = None
    sq_coh = np.abs(S_xy)**2 / (S_xx * S_yy)  # squared coherence, used by Gilron 2021
    icoh = np.imag(coherency)  # take imaginary (pos and neg)
    abs_icoh = np.abs(icoh)  # take absolute value
    # abs_icoh[abs_icoh < 0] = abs_icoh[abs_icoh < 0] * -1

    # get rid of 3rd dimensionality
    # if len(coh.shape) == 3: coh = coh[:, 0, :]
    if len(icoh.shape) == 3: icoh = icoh[:, 0, :]
    if len(abs_icoh.shape) == 3: abs_icoh = abs_icoh[:, 0, :]
    if len(sq_coh.shape) == 3: sq_coh = sq_coh[:, 0, :]

    """
    PM: implement ICOH detectable according to
    https://link.springer.com/article/10.1007/s10548-018-0640-0
    """

    return f, icoh, abs_icoh, coh, sq_coh



def get_theta_from_betaGamma(
    beta_sig: np.ndarray,
    gamma_sig: np.ndarray, fs: int,
    beta_freqs=[12, 20],
    gamma_freqs=[70, 80],
    theta_freqs=[4, 8],
    ZSCORE_envs: bool = True,  # result of env z-scoring looks most oscillatory
    ZSCORE_sigs: bool = False,
):
    if isinstance(beta_sig, list): beta_sig = np.array(beta_sig)
    if isinstance(gamma_sig, list): gamma_sig = np.array(gamma_sig)

    assert beta_sig.shape == gamma_sig.shape, 'beta and gamma not same shape'
    
    if ZSCORE_sigs:
        beta_sig = (beta_sig - np.nanmean(beta_sig)) / np.nanstd(beta_sig)
        gamma_sig = (gamma_sig - np.nanmean(gamma_sig)) / np.nanstd(gamma_sig)
    # calc beta env 
    beta = get_envelop(
        beta_sig, fs=fs, bandpass_freqs=beta_freqs
    )
    beta = smoothing(beta, fs=fs, win_ms=250,)

    # calc gamma env
    gamma = get_envelop(
        gamma_sig, fs=fs, bandpass_freqs=gamma_freqs
    )
    gamma = smoothing(gamma, fs=fs, win_ms=250,)

    if ZSCORE_envs:
        beta = (beta - np.nanmean(beta)) / np.nanstd(beta)
        gamma = (gamma - np.nanmean(gamma)) / np.nanstd(gamma)
    # calc ongoing difference between gamma and beta
    alt = gamma - beta
    theta = get_envelop(
        alt, fs=fs, bandpass_freqs=theta_freqs,
    )
    return theta


def get_aperiodic(data, fs, method='fooof',
                  method_params=None,):
    """
    Arguments:
        - data: currently handles 2d array
            (n-channels x n-samples)
        - fs
    
    Returns:
        - offset
        - exponent
        - R2 of fit
    """
    if data.shape[0] > data.shape[1]: data = data.T
    # select and transform data from one channel, in one epoch
    f, pxx = signal.welch(data, fs=fs, nperseg=fs, axis=1)
    # correct psd shape for linenoise corrected throughs
    pxx = specHelpers.correct_notch_throughs(
        f, pxx, np.arange(50, f[-1], 50))

    from fooof import FOOOF
    
    fm = FOOOF(peak_width_limits=(.5, 5),
               peak_threshold=.5,
               aperiodic_mode=method_params['knee'],
               max_n_peaks=method_params['max_n_peaks'],
               verbose=False,)     
    fm.fit(f, pxx, freq_range=method_params['f_range'])

    ap_off = fm.get_results().aperiodic_params[0]
    ap_exp = fm.get_results().aperiodic_params[-1]
    ap_fit_r2 = fm.get_results().r_squared

    return ap_off, ap_exp, ap_fit_r2


def get_fooof_peaks_freqs_and_sizes(
    f, pxx, range=[4, 98], knee_or_fix: str = 'knee',
    max_n_peaks = np.inf, fooof_fig_dir=None,
    plot=False, i_e=None, chname=None,
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
        max_n_peaks=max_n_peaks,
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
    fit_r2 = fm.get_results().r_squared
    
    # Swann / De Hemptinne
    pk_idx = [np.argmin(abs(np.array(f) - cf)) for cf in peak_freqs]
    pk_hgts = [np.log(pxx[i]) for i in pk_idx]
    lo_cut_idx = [np.argmin(abs(np.array(f) - cf - 5)) for cf in peak_freqs]
    hi_cut_idx = [np.argmin(abs(np.array(f) - cf + 5)) for cf in peak_freqs]
    base_hgts = [np.mean(np.log([pxx[i_lo], pxx[i_hi]]))
                for i_lo, i_hi in zip(lo_cut_idx, hi_cut_idx)]
    peak_logHeights = [(pk - base) for pk, base in zip(pk_hgts, base_hgts)]

    # random plot for check
    if plot:
        plt.figure()
        fm.plot()
        plt.title(f'{chname}  -  Goodness, R2: {fit_r2}')
        plt.savefig(
            os.path.join(fooof_fig_dir, f'FOOOF_fit_win5epoch{i_e}_{chname}'),
            dpi=150, facecolor='w',
        )
        plt.close()


    return ap_off, ap_exp, peak_freqs, peak_sizes, peak_logHeights, fit_r2


def get_fooof_fts_per_epoch(
    epoch_dat, fs, nperseg, ch_names,
    sources_incl=['ECOG', 'LFP_L', 'LFP_R'],
    fooof_range = [4, 98],  # wider window -> more accurate, more comp-time
    max_n_fooof_peaks = 15,
    bw_ranges = {
        'alpha': [8, 12],
        'lo_beta': [12, 20],
        'hi_beta': [20, 35],
        'midgamma': [60, 90]
    },
    i_e=None, fooof_fig_dir=None, ch_fig_dir=None,
    plot_examples=False,
):
    """
    extracts aperiodic and spectral peak features
    over 2d array (channels, timepoints)
    """
    # lists to store temp-single values and featuress (means, cv, sd)
    ft_out = specHelpers.get_empty_spectral_ft_dict(lists_or_means='means')
    ft_temp = specHelpers.get_empty_spectral_ft_dict(lists_or_means='lists')
    goodness_fits = []
    
    for i_ch in np.arange(epoch_dat.shape[0]): # loop over all channels
        check_plot = False

        chname = ch_names[i_ch]
        skip = True
        # define matching source
        for src in sources_incl:
            if chname.startswith(src):
                source = src
                skip = False
        if skip:
            print(f'skipped {chname}')
            continue  # skip channel if not in sources_incl
        # select and transform data from one channel, in one epoch
        f, pxx = signal.welch(
            epoch_dat[i_ch, :], fs=fs, nperseg=nperseg,
        )
        # correct psd shape for linenoise corrected throughs
        pxx = specHelpers.correct_notch_throughs(f, pxx, np.arange(50, 1201, 50))

        # for random plotting
        if plot_examples:
            if i_e in [0, 20, 40, 60, 80]:
                if i_ch in [0, 10, 20]:
                    check_plot = True
                    ch_fig_dir=os.path.join(fooof_fig_dir, chname)
                    if not os.path.exists(ch_fig_dir): os.makedirs(ch_fig_dir)

        # get aperiodic and periodic features
        (
            ap_off, ap_exp, pk_cf, pk_pw, log_hgts, fit_r2
        ) = get_fooof_peaks_freqs_and_sizes(
            f, pxx, range=fooof_range, knee_or_fix='knee',
            max_n_peaks=max_n_fooof_peaks,
            i_e=i_e, chname=chname, plot=check_plot,
            fooof_fig_dir=ch_fig_dir,
        )

        # distribute peaks to feat_arrays based on bandwidths
        if np.isnan(ap_off): continue

        goodness_fits.append(fit_r2)
        # add aperiodic feats
        ft_temp[source]['ap_off'].append(ap_off)
        ft_temp[source]['ap_exp'].append(ap_exp)
        # add peak-feats per bandwidth
        for bw, rng in zip(bw_ranges.keys(), bw_ranges.values()):
            bw_sel = np.logical_and(rng[0] < pk_cf, pk_cf < rng[1])
            # skip if peak combi not present
            if len(bw_sel) == 0: continue
            # add lists of peak-values to correct list
            ft_temp[source]['peak_freq'][bw].extend(pk_cf[bw_sel])
            ft_temp[source]['peak_size'][bw].extend(pk_pw[bw_sel])
            ft_temp[source]['peak_logHeight'][bw].extend(pk_pw[bw_sel])

    # convert ft_lists to means per source (and bw)
    for src in sources_incl:
        for ft in ['ap_off', 'ap_exp']:
            if ft_temp[src][ft] == []: continue  # skip empty lists
            ft_out[src][ft]['mean'] = np.nanmean(ft_temp[src][ft])
            ft_out[src][ft]['sd'] = np.nanstd(ft_temp[src][ft])
            ft_out[src][ft]['cv'] = variation(ft_temp[src][ft], nan_policy='omit')     
        
        for ft in ['peak_freq', 'peak_size', 'peak_logHeight']:
            for bw in bw_ranges.keys():
                if ft_temp[src][ft][bw] == []: continue  # skip empty lists
                ft_out[src][ft][bw]['mean'] = np.nanmean(ft_temp[src][ft][bw])
                ft_out[src][ft][bw]['sd'] = np.nanstd(ft_temp[src][ft][bw])
                ft_out[src][ft][bw]['cv'] = variation(ft_temp[src][ft][bw], nan_policy='omit')
        
    return ft_out, ft_temp, goodness_fits


def get_spectral_ft_names(ex_epoch):

    ft_names = []

    for src in ex_epoch:
        for ft in ex_epoch[src].keys():
            if ft.startswith('ap'):
                for m in ex_epoch[src][ft].keys():
                    ft_names.append(f'{src}_{ft}_{m}')
            elif ft.startswith('peak'):
                for bw in ex_epoch[src][ft].keys(): 
                    for m in ex_epoch[src][ft][bw].keys():
                        ft_names.append(f'{src}_{ft}_{bw}_{m}')

    return ft_names


def create_windowFrame_specFeats(
    epoch_feat_dict, save_csv=False,
    csv_path=None, csv_fname=None,
    fooof_fits=None,
):
    
    ft_names = get_spectral_ft_names(epoch_feat_dict[0])

    df = DataFrame(
        data=np.array([[np.nan] * len(ft_names)] * len(epoch_feat_dict)),
        columns=ft_names
    )

    for row, epoch in enumerate(epoch_feat_dict.values()):
        for src in epoch:
            for ft in epoch[src].keys():
                if ft.startswith('ap'):
                    for m in epoch[src][ft].keys():
                        df.iloc[row][f'{src}_{ft}_{m}'] = epoch[src][ft][m]
                elif ft.startswith('peak'):
                    for bw in epoch[src][ft].keys(): 
                        for m in epoch[src][ft][bw].keys():
                            df.iloc[row][f'{src}_{ft}_{bw}_{m}'] = epoch[src][ft][bw][m]

    if save_csv:
        df.to_csv(os.path.join(csv_path, csv_fname), header=True)

        if fooof_fits:  # save fits            
            n_col = max([len(fooof_fits[i]) for i in fooof_fits.keys()])
            fit_df = DataFrame(np.array([[np.nan] * n_col] * len(fooof_fits)))

            for i, f in enumerate(fooof_fits.values()):
                fit_df.iloc[i, :len(f)] = f
            
            fit_df.to_csv(
                os.path.join(csv_path, f'fits_{csv_fname}'),
                header=False)
                
    return df

