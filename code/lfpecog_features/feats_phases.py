'''
Functions to calculate Phase-Amplitude-
Coupling features and phase-difference features

Based on results by:
    - Swann et al (Phase differences)
    - De Hemptinne Nat Neurosc 2015 (PAC)
    - Tort 2014 (PAC method)
    - Combrisson 2020 (tensorpac: open-source PAC py-methods)
'''
# Import general packages and functions
import os
import numpy as np
from scipy import signal
import tensorpac

import lfpecog_features.feats_spectral_helpers as specHelpers



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
    TODO: REVISE

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

    Tensorpac article (Combrisson 2020): https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008302
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

