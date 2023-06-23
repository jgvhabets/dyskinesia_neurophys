# Import packages and functions
from typing import Any
import numpy as np
from itertools import compress
from scipy.signal import find_peaks

# Import own functions
import lfpecog_preproc.preproc_plotting as plotting
from lfpecog_features.feats_helper_funcs import nan_array


def artf_removal_dict(
    dataDict: dict,
    Fs_dict: dict,
    namesDict: dict,
    runInfo,
    edge_removal_sec: int=5,
    settingsVersion='vX',
):
    """
    Performs artefact selection and removal
    functions for all data arrays in dict

    Inputs:
        - ...
        - edge_removal_sec: number of seconds that will
            be removed at beginning and end of every
            recording, if 0, no removal performed
    """
    for group in dataDict.keys():
        
        print(f'\n\n\tSTART ARTEFACT REMOVAL: {group}\n')

        if group[:3] not in ['lfp', 'eco']:
            
            print(f'artefact removal skipped for {group}')
            continue

        dataDict[group], namesDict[group] = artf_removal_array(
            data=dataDict[group],
            Fs=Fs_dict[group],
            names=namesDict[group],
            group=group,
            runInfo=runInfo,
            to_plot=runInfo.mainSettings['report_plots'],
            edge_removal_sec=edge_removal_sec,
            settingsVersion=settingsVersion,
        )

    return dataDict, namesDict



def artf_removal_array(
    data,
    Fs: int,
    names: list,
    group: str,
    runInfo,
    win_len = 2,
    win_buffer = 1,
    ch_nan_limit=.5,
    edge_removal_sec=5,
    to_plot: bool=False,
    settingsVersion='vX',
    verbose: bool = False,
):
    '''
    Function to remove artefacts and visualize the resulting
    selection.
    Blocks-values are converted to NaNs when an outlier (value
    exceeds thresholds of n_std_cut times std dev of full recording).
    Also converted to NaN's if more than 25% of block is 0.

    Removes channels with XX % NaNs.
    
    Inputs:
        - data (2d array): array containing ephys-data, including
            time rows.
        - Fs (int): sampling frequency
        - names (list): list of corresponding array-row-names
        - group (str): names of group (lfp_left, ecog, lfp_right)
        - runInfo (class): settings extracted from input-JSONs in
            main script
        - win_len: window length in seconds to calculate artefacts
        - ch_nan_limit
        - remove_edge_secs: number of seconds on start and end-
            edge to be removed

    Returns:
        - clea_data (array): array with all channels in which
            artefacts are replaced by np.nan's
        - names: channel names
    '''
    sd_cut = runInfo.mainSettings['ephys']['artf_sd']
    # define artefact removal based on kernel or std-dev
    if sd_cut == 'kernel':
        kernel = True
    elif isinstance(sd_cut, float) or isinstance(sd_cut, int):
        kernel = False
    
    # remove edges
    if edge_removal_sec > 0:
        data = data[:, int(edge_removal_sec * Fs):
                       -int(edge_removal_sec * Fs)]

    w_samples = int(win_len * Fs)
    n_wins = int(data.shape[-1] / w_samples)
    buff_samples = int(win_buffer * Fs)

    clean_data = data[:, :int(n_wins * w_samples)].copy()  # delete redundant end not falling in window
    artf_data = nan_array(list(clean_data.shape))  # for plotting
    artf_data[:2, :] = clean_data[:2, :]

    timerowNames = [name for name in names if 'time' in name]
    
    group_nans = {}

    for ch in np.arange(len(timerowNames), data.shape[0]):

        if verbose: print(f'\nstart channel #{ch}: {names[ch]} (# {n_wins} win"s)')
        group_nans[ch] = []

        if not kernel:
            ### DEFINE if artf removal is necessary
            sig_absmedian = np.nanmedian(abs(data[ch, :]))
            sig_max = np.max([abs(np.nanmin(data[ch, :])), np.nanmax(data[ch, :])])
            sd_ratio = sig_max / sig_absmedian
            if verbose: print(f'ARTEFACT REMOVAL CHECK: ratio Max / median-abs: {sd_ratio}')

            if sd_ratio < 12:  # was 2 in v3
                if verbose: print(f'artefact removal skipped for {ch}')
                # dont change channel row in clean_data
                continue
                
            # define thresholds
            threshs = (
                np.nanmean(data[ch, :]) - (sd_cut * np.nanstd(data[ch, :])),
                np.nanmean(data[ch, :]) + (sd_cut * np.nanstd(data[ch, :]))
            )  # tuple with lower and upper threshold
            if verbose: print('Tresholds:', threshs)

            nancount = 0
        
            for w in np.arange(n_wins):
                w_i_start = w_samples * w

                tempdat = data[ch, w_i_start:w_i_start + w_samples]
                
                if not kernel:
                    # indicate as artefact if at least 1 sample exceeds threshold
                    if np.logical_or(
                        (tempdat < threshs[0]).any(),
                        (tempdat > threshs[1]).any()
                    ):
                        # if threshold is exceeded: fill window + buffer with nans
                        w_i_end = w_i_start + w_samples + buff_samples
                        w_i_start -= buff_samples
                        if w_i_start < 0: w_i_start = 0
                        if w_i_end > clean_data.shape[1]: w_i_end = clean_data.shape[1]
                        nan_fill = [np.nan] * (w_i_end - w_i_start)
                        try:
                            clean_data[ch, w_i_start:w_i_end] = nan_fill
                        except ValueError:
                            print(f'OG end: {w_i_start + w_samples + buff_samples}')
                            print(f'end: {w_i_end}')
                            print(f'OG start: {(w_samples * w) - buff_samples}')
                            print(f'start: {w_i_start}')
                            print(f'length nanfill: {len(nan_fill)}')
                            print(f'shape data {data.shape}, shape cleandata {clean_data.shape}')
                            raise ValueError
                        # artf_data collected for plotting reasons
                        artf_data[ch,
                                  w_samples * w:
                                  w_samples * w + len(tempdat)] = tempdat

                        nancount += 1
                    
                    elif sum(clean_data[
                        ch, w_i_start:w_i_start + w_samples
                    ] == 0) > (.25 * w_samples):

                        clean_data[
                            ch, w_i_start:w_i_start + w_samples
                        ] = [np.nan] * w_samples

                        nancount += 1
            
        elif kernel:
            kernel_value = 10
            k1 = [0] * (Fs-1) + [kernel_value]  # kernel for starts
            k2 = [kernel_value] + [0] * (Fs-1)  # kernel for ends
            window_shape = (len(k1),)  # equals kernel size, equals one-second
            thr = np.mean(abs(data[ch, :])) * 4 * kernel_value
            # convert 1d-data into windows for fast dot product calculation 
            d_vector_view = np.lib.stride_tricks.sliding_window_view(data[ch, :], window_shape)
            # calculate dot products between data-snippets and kernels
            k1_products = abs(np.dot(d_vector_view, k1))
            k2_products = abs(np.dot(d_vector_view, k2))
            # find peaks, where kernel was array was similar to kernel
            k1_prod_idx, _ = find_peaks(k1_products, distance=len(k1),
                                        prominence=thr)
            k2_prod_idx, _ = find_peaks(k2_products, distance=len(k2),
                                        prominence=thr)
            k2_prod_idx += Fs  # take one second after detected end

            for i_start, i_end in zip(k1_prod_idx, k2_prod_idx):
                clean_data[ch, i_start:i_end] = [np.nan] * (i_end - i_start)


        ## DROP CHANNELS WITH TOO MANY NAN'S
        n_nans = np.isnan(clean_data[ch, :]).sum()
        if verbose: print(f'\n\tChannel #{ch} ({names[ch]} has {n_nans} '
                          f'({round(n_nans / clean_data.shape[1], 1)}) NaN"s')
        if not kernel: print(f'\t... NaNs coming from {nancount} POS windows')
            
    if to_plot:
        plotting.plot_groupChannels(
            ch_names=names,
            groupData=clean_data,
            Fs=Fs,
            groupName=group,
            runInfo=runInfo,
            moment='post-artefact-removal',
            artf_data=artf_data,
            settingsVersion=settingsVersion,
        )

    return clean_data, names