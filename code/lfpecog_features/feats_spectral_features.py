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
from scipy import signal
from scipy.stats import variation
import matplotlib.pyplot as plt
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

