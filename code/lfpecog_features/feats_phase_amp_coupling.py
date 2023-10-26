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
# from tensorpac import Pac



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
    surr_n_perm=200,
    norm_method=None,
    dynamic_bin_widths=False,
    set_bin_widths=False,
    return_1_MI=False,
    verbose=False,
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
        - dynamic_bin_widths: use standard bin-width for phase, adjust
            amplitude bin-width to value of phase-bin-width
        - set_bin_widths: always use pre-set bin-widhts for phase and ampl
        - return_1_MI: return one summary MI-value without binning
        
    Returns:
        - pac_matrix (nd-arr): 2d or 3d matrix containing n rows for number
            of amplitude bins, and m columns for n of phase-bins
        - (if window_times defined) pac_times: array of timestamps
            correspoding to pac_values windows
        
            
    Doc: https://etiennecmb.github.io/tensorpac/generated/tensorpac.Pac.html#tensorpac.Pac
        - first idpac integer is PAC-method: 2 codes for MI;
        - second idpac integer is surrogate-method: 0: None, 1: Swap-phase/amp across trials,
            2: swapamp time blocks, 3: time lag
            (surrogate method (1): swap-phase surrogates are equal to real-values
        - third digit is normalisation: 0=None, 4=z-score

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
    if pac_method == 'MI':
        method_i = 2
    
    if surrogate_method == 'swap':
        surr_i = 1
    else:
        surr_i = 0
    
    if norm_method == None:
        norm_i = 0
    elif (norm_method.lower() == 'z-score' or
          norm_method.lower() == 'zscore'):
        norm_i = 4

    PAC_id = (method_i, surr_i, norm_i)

    # get bins to extract
    if dynamic_bin_widths:
        # get PAC dynamic bins
        pac_bins = get_dynamic_pac_bins(pha_range=freq_range_pha,
                                        amp_range=freq_range_amp)
        pac_arr_dyn = np.zeros((len(pac_bins['amp'][0]),
                                len(pac_bins['pha']),
                                sig_pha.shape[0]))

        for i_pha in np.arange(len(pac_bins['pha'])):
            # run PAC calculation
            pac_model = Pac(idpac=[2,0,0], dcomplex='hilbert',
                            f_pha=pac_bins['pha'][i_pha],
                            f_amp=pac_bins['amp'][i_pha],
                            verbose=verbose,)

            # calculate PAC matrix/matrices)
            pac_values = pac_model.filterfit(sf=fs, x_pha=sig_pha,
                                             x_amp=sig_amp,)
            if pac_values.shape[1] == 1: pac_values = pac_values[:, 0, :]
            pac_arr_dyn[:, i_pha, :] = pac_values

        if incl_times: return pac_arr_dyn, window_times
        else: return pac_arr_dyn
        

    elif set_bin_widths:
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
    
        # run PAC calculation
        pac_model = Pac(idpac=PAC_id, dcomplex='hilbert',
                        f_pha=pac_bins['phase'], f_amp=pac_bins['ampl'],
                        verbose=verbose)
        
        # calculate PAC matrix/matrices)
        pac_matrix = pac_model.filterfit(sf=fs, x_pha=sig_pha,
                                        x_amp=sig_amp, n_perm=surr_n_perm)
        # if n-window == 1: reduce 3d to 2d
        if pac_matrix.shape[2] == 1: pac_matrix = pac_matrix[:, :, 0]
    
        if incl_times: return pac_matrix, window_times
        else: return pac_matrix

    elif return_1_MI:
        # can be used as single value per window for e.g.
        # dynamic-time-pac analysis of movement (tap VS LID)

        # run PAC calculation
        pac_model = Pac(idpac=PAC_id, dcomplex='hilbert',
                        f_pha=freq_range_pha,
                        f_amp=freq_range_amp,
                        verbose=verbose)
        
        # calculate PAC matrix/matrices)
        pac_matrix = pac_model.filterfit(
            sf=fs, x_pha=sig_pha, x_amp=sig_amp,
        )
        # n-phase-bins and n-amp-bins is 1: reduce 3d to 1d
        pac_matrix = pac_matrix[0, 0, :]
        
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


def get_dynamic_pac_bins(
    pha_range, amp_range,
    pha_bin_width=2, amp_bin_resolution=4,
):
    """
    Creates bins for phase and amplitude
    of PAC calculation.
    Takes phase centers of default 2 Hz bins,
    adjusts the amplitude bin-widths according
    to the phase-center freq (e.g. phase center 20 Hz,
    (with bin-width 19 - 21 Hz) has ampl-bin-widths
    of 20 Hz, e.g. 60 - 80 Hz)

    Arguments:
        - pha_range, amp_range (tuples): (low, hi)
        - pha_bin_width (int): width and resolution of bins
        - amp_bin_resolution (int): resolution of bins,
            width is dynamic to phase-center-freq
    """
    pac_bins = {}
    # creates center frequencies from start until incl end of freq-range
    center_phases = np.arange(pha_range[0],
                              pha_range[1] + .1,
                              pha_bin_width).astype(int)
    # create e.g. 2 Hz bins for phase freqs
    pac_bins['pha'] = [[c - int(pha_bin_width/2),
                          c + int(pha_bin_width/2)]
                         for c in center_phases]
    # create center freqs for amp freqs (i.e., a 4 Hz)
    center_amps = np.arange(amp_range[0],
                            amp_range[1] + .1,
                            amp_bin_resolution).astype(int)
    # create bins with width of resp. phase-center-freq
    pac_bins['amp'] = []

    for pha_c in center_phases:
    
        pac_bins['amp'].append(
            [[amp_c - int(pha_c/2), amp_c + int(pha_c/2)]
              for amp_c in center_amps]
        )
    
    return pac_bins

