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
import numpy as np
from pandas import isna
from tensorpac import Pac



def calculate_PAC_matrix(
    sig_pha: np.ndarray,
    sig_amp: np.ndarray,
    fs: int,
    freq_range_pha,
    freq_range_amp,
    window_times=None,
    pac_binwidths: dict = {"phase": 2, "ampl": 4},
    pac_method='MI',
    surrogate_method=None,
    norm_method=None,
):
    """
    Calculating Phase Amplitude Coupling following the tensorpac package,
    based on methodology from De Hemptinne et al Nature Neuroscience 2015,
    and Qasim et al Neurobiol of Disease 2018.

    Input:
        - sig_pha (arr): 1d or 2d--array with timedomain signal(s) to
            extract phase of PAC from
        - sig_amp (arr): 1d or 2d--array with timedomain signal(s) to
            extract amplitude of PAC from
        NOTE: for local PAC: sig_pha and sig_amp should be the same,
            otherwise, they can be from different location but should
            be of the same size
        - fs (int): corresponding sampling freq
        - freq_range_pha/amp: lower and upper border of frequency range
            for phase and amplitudes to use
        - window_times (array): optionally, timestamps of windows inserted
        - pac_bindwidths: dict containing phase and ampl: integers for
            bin width in Hz
        - pac_method/ surrogate_method/ norm_method: defaults to MI
            (Modulation Index, Tort/Hemptinne), no surrogates,
            no normalisation (see documentation).
        
    Returns:
        - pac_matrix (nd-arr): 2d or 3d matrix containing n rows for number
            of amplitude binss, and m columns for n of phase-bins
        - (if window_times defined) pac_times: array of timestamps
            correspoding to pac_values windows
        
            
    Doc: https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac
        first idpac integer is PAC-method: 2 codes for MI;
        second idpac integer is surrogate-method: 0=None, 3 codes for time lag
        (surrogate method (1): swap-phase surrogates are equal to real-values
        third digit is normalisation: 0=None, 4=z-score

        Tensorpac article (Combrisson 2020): https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008302
    """
    # data checks
    assert sig_amp.shape == sig_pha.shape, "PAC Phase- and Amp-array need same shape"
    if window_times:
        assert len(window_times) == len(sig_amp), (
            'if PAC window times are given, length should equal data length'
        )  # length times equals n-rows sigs
        if isinstance(window_times, list): window_times = np.array(window_times)
        incl_times = True
    else:
        incl_times = False
    
    # clean nan rows (parallel for data and times if given)
    if isna(sig_pha).any():
        nan_rows = isna(sig_pha).any(axis=1)
        sig_pha = sig_pha[~nan_rows]
        sig_amp = sig_amp[~nan_rows]
        if incl_times: window_times = window_times[~nan_rows]
    if isna(sig_amp).any():
        nan_rows = isna(sig_amp).any(axis=1)
        sig_pha = sig_pha[~nan_rows]
        sig_amp = sig_amp[~nan_rows]
        if incl_times: window_times = window_times[~nan_rows]

    assert len(freq_range_pha) == 2 and len(freq_range_amp) == 2, (
        'lengths of PAC amp/phase freq ranges have to be 2'
    )

    # define pac_id
    if pac_method == 'MI': method_i = 2
    if surrogate_method == None: surr_i = 0
    if norm_method == None: norm_i = 0
    elif norm_method == 'z-score': norm_i = 4

    PAC_id = (method_i, surr_i, norm_i)

    # define PAC-freq-bins: 2 lists of lists with lower and upper border
    pac_bins = {}
    pac_bins['phase'] = get_pac_bins(
        freq_range=freq_range_pha,
        binwidth=pac_binwidths['phase']
    )
    pac_bins['ampl'] = get_pac_bins(
        freq_range=freq_range_amp,
        binwidth=pac_binwidths['ampl']
    )
    # pac_bins['phase'] = [
    #     [start, start + pac_binwidths['phase']]
    #      for start in np.arange(freq_range_pha[0],
    #                             freq_range_pha[1],
    #                             pac_binwidths['phase'])
    # ]
    # pac_bins['ampl'] = [
    #     [start, start + pac_binwidths['ampl']]
    #      for start in np.arange(freq_range_amp[0],
    #                             freq_range_amp[1],
    #                             pac_binwidths['ampl'])
    # ]
    
    # calculate pac borders (len is 1 more than length of bins)
    pac_borders = {}
    for var in pac_bins.keys():
        bins = pac_bins[var]
        pac_borders[var] = [bins[0][0]] + [b[1] for b in bins]

    # run PAC calculation
    pac_model = Pac(idpac=PAC_id, dcomplex='hilbert',
                    f_pha=pac_bins['phase'], f_amp=pac_bins['ampl'],)
    
    # calculate PAC matrix/matrices)
    pac_matrix = pac_model.filterfit(sf=fs, x_pha=sig_pha,
                                     x_amp=sig_amp,)
    # if n-window == 1: reduce 3d to 2d
    if pac_matrix.shape[2] == 1: pac_matrix = pac_matrix[:, :, 0]
    
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 1, 1)
    # pac_model.comodulogram(pac_matrix[:, :, :50].mean(-1),
    #                     title="Local ECoG beta-gamma PAC "
    #                     "(first 50 windows)",
    #             cmap='Reds')
    # plt.show()

    # plt.subplot(1, 1, 1)
    # pac_model.comodulogram(pac_matrix[:, :, -50:].mean(-1),
    #                     title="Local ECoG beta-gamma PAC "
    #                     "(last 50 windows)",
    #                         cmap='Reds')
    # plt.show()
    
    if incl_times: return pac_matrix, window_times
    else: return pac_matrix


def get_pac_bins(
    freq_range, binwidth
):
    pac_bins = [[start, start + binwidth]
                for start in np.arange(freq_range[0],
                                       freq_range[1],
                                       binwidth)]
    
    return pac_bins


def PAC_select(
    pacmatrix, freqs_pha, freqs_amp,
    pha_bw=(13, 30), amp_bw=(60, 90)):
    """
    TODO: REVISE

    Function to select specific bandwidths to include in
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

