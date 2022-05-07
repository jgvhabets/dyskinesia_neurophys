'''
Functions to calculate bandwidth-based
spectral features from neurophysiology
data (LFP and ECOG) in ReTune's B04 Dyskinesia Project

Containing:
- bandpass filter
- coherence
- Phase Amplitude Coupling (tensorpac)
'''
# Import general packages and functions
import os
import numpy as np
from scipy import signal

or_wd = os.getcwd()  # get original working directory

# if necessary: change working directory to import tensorpac
if or_wd.split('/')[-1] == 'dyskinesia_neurophys':
    os.chdir(os.path.join(or_wd, 'code/PAC/tensorpac'))
# import functions for Phase-Amplitude-Coupling
from tensorpac import Pac

os.chdir(or_wd)  # set back work dir


def bandpass(sig, freqs, fs, order=3, twice=False):
    """
    Bandpass filtering using FIR window (based on scipy
    firwin bandpas filter documentation)
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html?highlight=bandpass%20firwin
    
    Input:
        - sig: neurophys-signal to filter
        - freqs: list or tuple with lower and upper freq
        - fs: sampling freq
        - order (int): n-order of filter
        - twice (Boolean): perform filter twice (docu-
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
    
    # taps = signal.firwin(order, cutoff=freqs, fs=fs, pass_zero=False,)
    # a = 1
    # zi = signal.lfilter_zi(taps, a)
    # filtsig, _ = signal.lfilter(taps, a, sig, zi=zi*sig[0])
    # if twice:
    #     filtsig, _ = signal.lfilter(taps, a, filtsig, zi=zi*filtsig[0])

    return filtsig


def coherence(
    sig1, sig2, fs, name1, name2,
    nseg=256, imag=True, freq_bws={},
):
    """
    Coherence, calculated per bandwidth frequencies based on
    Qasim et al Neurobiol Dis 2016/ De Hemptinne 2015.
    Imaginary coherence ignores signals with zero-time lag
    and thereby reduces the probability of picking up
    non-physiological correlations (based on Rolston &
    Chang, Cereb Cortex 2018).

    Input:
        - sig1-2: two arrays with signals to include
        - fs: sample frequency (should be same for sig1-2)
        - name1-2: signal names to use in parameter name
        - nseg= length of welch window
        - imag: include imaginary coherence or not
        - freq_bws (dict with tuples): define freq bandwidth
            to extract from. if not given, std freqs-battery
            is extracted

    Returns:
        - coh_values: list of (imaginary) coherence values
        - coh_names: list of corresponding names
    """
    coh_values = []
    coh_names = []
    if not freq_bws:
        # if not specified: std freq-bw-battery
        freq_bws = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'lowbeta': (13, 20),
            'highbeta': (20, 30),
            'bbgamma': (60, 400),
            'lowgamma': (30, 60),
            'midgamma': (60, 90),
            'gamma': (30, 90),
        }
    f, Px = signal.welch(sig1, fs=fs, nperseg=nseg,)
    _, Py = signal.welch(sig2, fs=fs, nperseg=nseg,)
    _, Pxy = signal.csd(sig1, sig2, fs=fs, nperseg=nseg,)

    coh = np.abs(Pxy) / np.sqrt(Px * Py)
    icoh = Pxy.imag / (np.sqrt(Px * Py))

    for bwkey in freq_bws.keys():
        # take mean of values within bandwidths
        coh_names.append(f'Coh_{bwkey}_{name1}_{name2}')
        bw = freq_bws[bwkey]  # take freq-ranges of bandwidth
        coh_values.append(np.mean(
            coh[[bw[0] < fi < bw[1] for fi in f]]
        ))

        if imag == True:  # if imaginary coh wanted, same for imag values
            coh_names.append(f'iCoh_{bwkey}_{name1}_{name2}')
            coh_values.append(np.mean(
                icoh[[bw[0] < fi < bw[1] for fi in f]]
            ))

    return coh_values, coh_names


def PAC_matrix(
    sig,
    Fs,
    signame,
    freqs_pha=np.arange(12, 30.5, 2),  # default full beta
    freqs_amp=np.arange(50, 200.5, 2),
    nperm=1000,
    zscore=False,
):
    """
    Calculating Phase Amplitude Coupling following the tensorpac package,
    based on methodology from De Hemptinne et al Nature Neuroscience 2015,
    and Qasim et al Neurobiol of Disease 2018.

    Input:
        - sig (arr): 1d-array with timedomain signal to extract PAC from
        - Fs (int): corresponding sampling freq
        - signame (str): name of channel used for sig
        - freqs_pha (arr): freq-borders used for phase-component. Between
            all consecutive freq's, one bin is calculated. Number of
            freq-bins is len(freqs_pha) - 1
        - freqs_amp (arr): freq-borders used for amplitude-component,
            similar to freqs_pha
        - nperm (int): number of permuted surrogates (if z-score is True)
        - zscore (Bool): express PAC in zscore, if True -> surrogate
            permutations are calculated for zscore calculation
    
    Returns:
        - pacmatrix (nd-arr): matrix containing n rows for amplitudes, and
            m columns for phases
        - pacname (str): label for PAC matrix and source-signal
        - freqs_pha, freqs_amp (arr): freq-borders used for resp. phase- and
            amplitude-component. Number of freq-bins is len(freqs_pha) - 1.
            Can be used for selecting specific freq-widths of PAC-values.

    Doc: https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac
        first idpac integer is PAC-method: 2 codes for MI;
        second idpac integer is surrogate-method: 3 codes for time lag
        (surrogate method (1): swap-phase surrogates are equal to real-values
    """
    if zscore:
        idpac = (2, 3, 0)
        pacname = f'zPAC_{signame}'
    else:
        idpac = (2, 0, 0)
        pacname = f'PAC_{signame}'
    
    tp = Pac(
        idpac=idpac,  # PAC-settings
        f_pha=freqs_pha,
        f_amp=freqs_amp,
        dcomplex='hilbert',
        verbose=False
    )
    pacmatrix = tp.filterfit(
        Fs, sig, mcp='bonferroni',
        n_perm=nperm, n_jobs=1, random_state=2711, 
        edges=80,  # confirm correctness of 80
    ).squeeze()  # squeeze makes 3d -> 2d

    if zscore:
        # Z-score PAC values based on surrogate-mean and -sd
        # every matrix-pac (e.g. xy) is z-scored individually
        # against all permuted xy-pac-values
        perm_sd = np.nanstd(tp.surrogates.squeeze(), axis=0)
        perm_m = np.nanmean(tp.surrogates.squeeze(), axis=0)
        pacmatrix = ((pacmatrix - perm_m) / perm_sd)
    
    # # calc significancies compared to surrogates (if zscore)
    # p95 = np.percentile(tp.surrogates.squeeze(), 95, axis=0)
    # sign95 = epochpac > p95
    # # to visualise surrogate means
    # tp.comodulogram(
    #     tp.surrogates.squeeze().mean(0),
    #     # colorbar=False,
    #     ylabel='Amplitude Freq (Hz)',
    #     xlabel='Phase Freq (Hz)',
    #     # title='Mean surrogates,
    # )

    return pacmatrix, pacname, freqs_pha, freqs_amp


def PAC_select(
    pacmatrix, freqs_pha, freqs_amp,
    pha_bw=(13, 30), amp_bw=(60, 90)):
    """Function to select specific bandwidths to include in
    summarising mean-score of total caluclated PAC matrix
    Input:
        - pacmatrix (arr): resulting PAC-values of PAC_comod...
        - freqs_pha, freqs_amp (arr): corresponding freq-bin
            borders, result of PAC_comodolugram
        - pha_bw (tuple): lower and upper border to include
            phase-frequencies
        - amp_bw (tuple): idem but for ampl-frequencies
    
    Returns:
        - meanpac (float): mean PAC-value within freq-ranges
    """
    # create Boolean list to select freqs within ranges
    sel_pha = [np.logical_and(freqs_pha[i - 1] >= pha_bw[0],
        freqs_pha[i] <= pha_bw[1]) for i in range(1, len(freqs_pha))]
    sel_amp = [np.logical_and(freqs_amp[i - 1] >= amp_bw[0],
        freqs_amp[i] <= amp_bw[1]) for i in range(1, len(freqs_amp))]
    # select PAC-values based on freq-border-lists
    if np.logical_and(np.any(sel_pha), np.any(sel_amp)):
        sel = pacmatrix[:, sel_pha]  # select phases
        sel = sel[sel_amp, :]  # select amplitudes

    meanpac = np.mean(sel)

    return meanpac

# # IMPORT MATPLOTLIB -> PLT
# def visualise_PAC():
#     """Plot Heatmap (colormesh) of PAC-comodulogram"""
#     fig,ax = plt.subplots(1,1, figsize=(8,4))
#     c = ax.pcolormesh(pacmatrix, cmap='viridis',
#         vmin=-3, vmax=3, )
#     # plot make up
#     fig.colorbar(c, ax=ax)
#     ax.set_xticks(np.arange(len(freqs_pha)))
#     ax.set_xticklabels(freqs_pha.astype(int))
#     ax.set_yticks(np.linspace(0, len(freqs_amp), 5))
#     ax.set_yticklabels(np.linspace(
#         freqs_amp[0], freqs_amp[-1], 5).astype(int))
#     plt.show()

