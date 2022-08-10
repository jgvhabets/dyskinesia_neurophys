# Import packages and functions
from typing import Any
import numpy as np
from itertools import compress

# Import own functions
import preproc_plotting as plotting
from tapping_feat_calc import nan_array


def artf_removal_dict(
    dataDict: dict,
    namesDict: dict,
    runInfo,
):
    """
    Performs artefact selection and removal
    functions for all data arrays in dict
    """
    for group in dataDict.keys():
        
        print(f'\n\n\tSTART ARTEFACT REMOVAL: {group}\n')

        if group[:3] not in ['lfp', 'eco']:
            
            print(f'artefact removal skipped for {group}')
            continue

        dataDict[group], namesDict[group] = artf_removal_array(
            data=dataDict[group],
            names=namesDict[group],
            group=group,
            runInfo=runInfo,
            to_plot=runInfo.mainSettings['report_plots'],
        )

    return dataDict, namesDict



def artf_removal_array(
    data,
    names: list,
    group: str,
    runInfo,
    ch_nan_limit=.5,
    to_plot: bool=False,
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
        - names (list): list of corresponding array-row-names
        - group (str): names of group (lfp_left, ecog, lfp_right)
        - runInfo (class): settings extracted from input-JSONs in
            main script
        # - win_len (int): block window length in milliseconds,
        # - n_stds_cut, int: number of std-dev's above and below mean that
        #     is used for cut-off's in artefact detection,
        # - save (str): 1) directory where to store figure, 2) 'show' to only
        #     plot in notebook, 3) None to not plot.

    Returns:
        - sel_bids (array): array with all channels in which artefacts
        are replaced by np.nan's.
    '''
    Fs = runInfo.mainSettings['ephys']['resample_Fs']
    sd_cut = runInfo.mainSettings['ephys']['artf_sd']
    win_len = 1  # window length in seconds
    w_samples = win_len * Fs
    n_wins = int(data.shape[-1] / w_samples)

    clean_data = data[:, :int(n_wins * w_samples)].copy()
    artf_data = nan_array(list(clean_data.shape))  # for plotting
    artf_data[:2, :] = clean_data[:2, :]

    timerow_sel = ['time' in name for name in names]
    timerowNames = list(compress(names, timerow_sel))
    
    group_nans = {}

    print(f'\n\nARTEFACT INDEX CHECK: all-names: {names}; data-shape: {data.shape}')

    for ch in np.arange(len(timerowNames), data.shape[0]):

        print(f'\nstart channel #{ch}: {names[ch]} (# {n_wins} win"s)')
        group_nans[ch] = []

        thresh_dat = data[ch, int(5 * Fs):-int(5 * Fs)]
        # define thresholds not over first and last seconds
        threshs = (
            np.nanmean(thresh_dat) - (sd_cut * np.nanstd(thresh_dat)),
            np.nanmean(thresh_dat) + (sd_cut * np.nanstd(thresh_dat))
        )  # tuple with lower and upper threshold
        ### TODO: TRY OUT WITH SELECTION BASED ON PSD OFF-SET

        print('Tresholds:', threshs)
        nancount = 0
    
        for w in np.arange(n_wins):

            tempdat = data[ch, w_samples * w:w_samples * (w + 1)]

            if np.logical_or(
                (tempdat < threshs[0]).any(),
                (tempdat > threshs[1]).any()
            ):

                clean_data[
                    ch, w_samples * w:w_samples * (w + 1)
                ] = [np.nan] * w_samples

                artf_data[
                    ch, w_samples * w:w_samples * (w + 1)
                ] = tempdat

                nancount += 1
            
            elif sum(clean_data[
                ch, w_samples * w:w_samples * (w + 1)
            ] == 0) > (.25 * w_samples):

                clean_data[
                    ch, w_samples * w:w_samples * (w + 1)
                ] = [np.nan] * w_samples

                nancount += 1

        ## DROP CHANNELS WITH TOO MANY NAN'S
        n_nans = np.isnan(clean_data[ch, :]).sum()
        print(f'\n\tChannel #{ch} ({names[ch]} has {n_nans} '
            f'({round(n_nans / clean_data.shape[1], 1)}) NaN"s'
            f'\n\t(coming from {nancount} POS detections')
            
    if to_plot:
        plotting.plot_groupChannels(
            ch_names=names,
            groupData=clean_data,
            Fs=Fs,
            groupName=group,
            runInfo=runInfo,
            moment='post-artefact-removal',
            artf_data=artf_data,
        )

    return clean_data, names