"""
Function to perform SSD
(Spatio-Spectral-Decomposition) of
timeseries

Based on:
- https://github.com/neurophysics/meet
- Waterstraat G, Curio G, Nikulin VV.
    On optimal spatial filtering for the detection of phase coupling in multivariate neural recordings.
    Neuroimage. 2017 Jun 13;157:331-340. doi: 10.1016/j.neuroimage.2017.06.025.
"""

# import packages
import numpy as np

import meet.meet as meet  # https://github.com/neurophysics/meet

def get_SSD_component(
    data_2d, fband_interest, s_rate,
    flank_distance=5,
    use_freqBand_filtered=True, return_comp_n=0,
):
    """
    Function to freq-specifically filter 2d-data
    with SSD (using meet toolbox).

    Inputs:
        - data_2d (ndarray): 2d array n-channels x n-samples
        - fband_interest: tuple or list with lower
            and upper freq-band of interest border
        - fs (int): sampling rate
        - flank_distance (int/float): distance distal of freq-band
            of interest to define flank bands,
            defaults to +/- 5 Hz
        - use_freqBand_filtered (bool): use the freq-band
            filtered signal to multiply with the
            SSD-filter and to return
        - return_comp_n (int): the nth SSD-component to
            return (0 is the first component)
    
    Returns:
        - SSD_filtered_data
        - SSD_pattern
        - SSD_eigvals

    """
    assert type(return_comp_n) in [int, np.ndarray, list], (
        'return_comp_n should be integer or list/array,'
        f' instead {type(return_comp_n)} was given'
    )

    if isinstance(return_comp_n, int):
        assert return_comp_n < data_2d.shape[1], (
            'return_comp_n cannot be larger than n-channels'
        )
    else:
        for c in return_comp_n:
            assert c < data_2d.shape[1], (
                'return_comp_n cannot be larger than n-channels'
            )
    

    freqOfInt_range = np.r_[fband_interest]
    flank_range = np.r_[freqOfInt_range[0] - flank_distance,
                        freqOfInt_range[1] + flank_distance]
    # correct negative freq-values to 1 Hz
    if (flank_range < 1).any(): flank_range[flank_range < 1] = 1


    # filter signals for SSD
    interestBand_filt = meet.iir.butterworth(data_2d, fp=freqOfInt_range,
                                    fs=np.r_[0.8,1.2]*freqOfInt_range, s_rate=s_rate)

    wide_filt= meet.iir.butterworth(data_2d, fp=flank_range,
                                    fs=np.r_[0.8,1.2]*flank_range, s_rate=s_rate)
    flank_filt = meet.iir.butterworth(wide_filt, fs=freqOfInt_range,
                                    fp=np.r_[0.8,1.2]*freqOfInt_range, s_rate=s_rate)

    # Perform SSD
    SSD_filter, SSD_eigvals = meet.spatfilt.CSP(interestBand_filt, flank_filt)
    SSD_pattern = meet.spatfilt.pattern_from_filter(SSD_filter, interestBand_filt)
    #normalize the pattern to have a maximum absolute value of 1
    SSD_pattern /= np.abs(SSD_pattern).max(-1)[:,np.newaxis]

    if use_freqBand_filtered:
        # Apply freq-spec SSD filter on beta-filtered signals
        SSD_filtered_data = SSD_filter.T @ interestBand_filt

    else:
        # Apply freq-spec SSD filter on origin data containing all frequencies
        SSD_filtered_data = SSD_filter.T @ data_2d

    # if only one component is asked
    if isinstance(return_comp_n, int):
        return SSD_filtered_data[return_comp_n], SSD_pattern, SSD_eigvals
    
    # if multiple components are asked
    elif np.logical_or(
        isinstance(return_comp_n, list),
        isinstance(return_comp_n, np.ndarray)
    ):
        return_array = np.zeros((len(return_comp_n), data_2d.shape[1]))
        for i, c in enumerate(return_comp_n):
            return_array[i, :] = SSD_filtered_data[c]
        return return_array, SSD_pattern, SSD_eigvals
        





# # Import public packages and functions
# import numpy as np

# from mne import create_info
# from mne.decoding import SSD

# def SSD_on_array(
#     array, fs, freqband_to_optim
# ):
#     """
#     Spatial Signal Decompositino increases
#     the signal to noise ratio within a specific
#     freq-bandwidth by correcting the signal
#     for the 'noise-floor' measured in
#     neighbouring frequency bins

#     Based on mne.Decoding.SDD:
#     (https://mne.tools/stable/generated/mne.decoding.SSD.html#mne.decoding.SSD)

#     TODO: eigenvector failure (positive definitive error)
#     when performed on 2d data array
    
#     Input:
#         - array to transform: should be without nans
#         - fs
#         - freqband_to_optim: tuple or list
#             with lower and higher freq border
    
#     Returns:
#         - ssd_array: array with optimised SNR
#             for defined freq-band
    
#     Raises:
#         - Assertion if array contains NaNs
#     """
#     assert ~np.isnan(array).any(), (
#         'array to perform SSD on cannot contain NaNs'
#     )
#     if len(array.shape) == 1: array = np.array([array])  # transform to 2d array 1 x samples
#     if array.shape[0] > array.shape[1]: array = array.T  # transpose in horizontal format
    
#     # create mne info object
#     ch_names=[f'n{n}' for n in range(array.shape[0])]
#     mne_info = create_info(
#         ch_names=ch_names,
#         sfreq=fs,
#         ch_types=['eeg'] * len(ch_names)
#     )
    
#     freqs_noise = (freqband_to_optim[0] - 4, freqband_to_optim[1] + 4)
#     # freqs_noise = (1, 50)

#     ssd_obj = SSD(
#         info=mne_info,
#         reg='oas',
#         sort_by_spectral_ratio=False,  # False for purpose of example.
#         filt_params_signal=dict(l_freq=freqband_to_optim[0],h_freq=freqband_to_optim[1],
#                                 l_trans_bandwidth=1, h_trans_bandwidth=1),
#         filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
#                                l_trans_bandwidth=1, h_trans_bandwidth=1),
#         n_components=len(array),
#         n_fft=len(array),
#     )
#     # perform SSD
#     ssd_obj.fit(array)

#     ssd_sources = ssd_obj.transform(X=array)

#     return ssd_obj, ssd_sources
