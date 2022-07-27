# Import packages and functions
from array import array
import os
import numpy as np
import mne
import matplotlib.pyplot as plt


def filters_for_dict(
    dataDict: dict, chNamesDict:dict,
    settings:dict, filtertype:str
):
    """
    Use bp_filter() and notch_filter()
    for dictionaries containing several
    groups of data (ephys/acc/...)
    """
    for group in dataDict.keys():

        n_timerows = sum([
            'time' in n for n in chNamesDict[group]
        ])

        if group[:3] == 'acc': dtype = 'acc'

        elif group[:3] in ['lfp', 'eco']: dtype = 'ephys'
        
        if filtertype == 'bandpass':
            
            dataDict[group] = bp_filter(
                data=dataDict[group],
                n_timeRows=n_timerows,
                Fs=settings[dtype]['orig_Fs'],
                l_freq=settings[dtype]['bandpass'][0],
                h_freq=settings[dtype]['bandpass'][1],
                method='iir',
            )

        elif filtertype == 'notch':

            dataDict[group] = notch_filter(
                data=dataDict[group],
                n_timeRows=n_timerows,
                Fs=settings[dtype]['orig_Fs'],
                transBW=settings[dtype]['transitionWidth'],
                notchW=settings[dtype]['notchWidth'],
                method='fir',
                verbose=False,
            )


    return dataDict

def bp_filter(
    data: array,
    n_timeRows: int,
    Fs: int,
    l_freq: int,
    h_freq: int,
    method='fir',
    fir_window='hamming',
    verbose=False,
):
    '''
    Function to execute bandpass filter.

    Arguments:
        - data (array): 3d or 2d array with data,
            3d: [windows, rows, time points]
            2d: [rows, time points]
        - Fs (int): sampling freq
        - l_freq (int): lower freq of bandpass
        - h_freq (int): higher freq of bandpass
        - method: method of filter to use (fir / iir)
        - fir_window: type of fir_window, if fir applied
    
    Returns:
        - data_out (array): corresponding 3d or 2d array
         with filtered data
    '''
    data_out = data.copy()

    if len(data.shape) == 3:
        for w in np.arange(data.shape[0]):
            try:
                data_out[w, n_timeRows:, :] = mne.filter.filter_data(
                    data=data[w, n_timeRows:, :],
                    sfreq=Fs,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method=method,
                    fir_window=fir_window,
                    verbose=verbose,
                )
            except ValueError:
                '''If there are no channels available after
                artefact removal: fill with zeros'''
                data_out = np.zeros((data.shape))
                print('### BandPass Filter Warning ###\n'
                    'filled with zeros')

    if len (data.shape) == 2:
        data_out[n_timeRows:, :] = mne.filter.filter_data(
            data=data[n_timeRows:, :],
            sfreq=Fs,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            fir_window=fir_window,
            verbose=verbose,
        )

    return data_out


def notch_filter(
    data: array,
    n_timeRows: int,
    transBW: int,  # circa 10 Hz
    notchW: int,  # not too small / steep
    Fs: int,  # sample freq, default 4000 Hz
    method='fir',
    fir_win='hamming',
    fir_design='firwin',
    verbose='Warning',
):
    '''
    Applies notch-filter to filter local peaks due to powerline
    noise. Uses mne-fucntion (see doc).
    
    Arguments:
        - data (array): 3d array with data, first dimension
        are windows, second dim are rows, third dim are
        the data points over time within one window
        - ch_names (list): channel-names
        - n_timeRows (int): number of time-rows in array
        - transBW (int): transition bandwidth, circa 10 Hz
        - notchW (int): notch width, not too steep
        - Fs
        # - save (str): if pre and post-filter figures should be saved,
            directory should be given here
        - verbose (str): amount of documentation printed.
    
    Returns:
        - data_out: filtered data array.
    '''
    data_out = data.copy()

    freqs = np.arange(50, int(Fs / 4), 50)

    if len(data.shape) == 3:

        for w in np.arange(data.shape[0]):
            data_out[w, n_timeRows:, :] = mne.filter.notch_filter(
                x=data[w, n_timeRows:, :],
                Fs=Fs,
                freqs=freqs,
                trans_bandwidth=transBW,
                notch_widths=notchW,
                method=method,
                fir_window=fir_win,
                fir_design=fir_design,
                verbose=verbose,
            )

    if len(data.shape) == 2:

        data_out[n_timeRows:, :] = mne.filter.notch_filter(
            x=data[n_timeRows:, :],
            Fs=Fs,
            freqs=freqs,
            trans_bandwidth=transBW,
            notch_widths=notchW,
            method=method,
            fir_window=fir_win,
            fir_design=fir_design,
            verbose=verbose,
        )

    return data_out

