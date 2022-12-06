"""
Utility-Functions used for windowing
and epoching purposes during signal-
processing
"""
# Import public packages
import numpy as np
from pandas import DataFrame, isna
from dataclasses import dataclass, field
from array import array
from typing import Any
from itertools import compress


def get_windows(
    data,
    fs,
    col_names = None,
    winLen_sec=180,
    part_winOverlap=.5,
    min_winPart_present=.66,
    remove_nan_timerows: bool = True,
    movement_part_acceptance: float = 1.,
    return_as_class=False,
):
    """
    Select windows with size nWin * Fs,
    exclude windows with nan's, and save corresponding
    dopa-times to included windows.
    Function selects windows based on dopa-times using
    .loc function within a pd.DataFrame.

    TODO: make function hybrid, A: for getting times only,
    B: for getting data nd-array

    Inputs:
        - data: dataframe or array
        - fs (int): sample freq
        - col_names, time_index: needed if data is given as array
        - win_len: length of window in seconds
        - part_winOverlap (float): part of window used as overlap
            between consecutive windows
        - min_winPart_present (float): exclude window if
            smaller part than this variable is present
        - movement_part_acceptance (float): if given, a window
            will be excluded when the part of accelerometer-
            detected movement exceeds this number. Defaults
            to 1, meaning that all windows are accepted.
        - return_as_class: bool defines whether results are
            returned as class consisting of data, keys, times;
            or as separate variable (tuple)
    
    Returns:
        - win_array: 3d array of n-windows x n-samples x n-channels
            windows can have different sizes of present data,
            therefore windows are nan-padded at the end, to make
            them all fit the same 3d array (nan-rows are deleted
            in later ft-extraction functions)
        - fs: int
        - arr_keys: names of columns
        - arr_times_sec: corresponding times of window
            start dopa-times
    """
    if isinstance(data, DataFrame):
        try:  # if dopa-time is as column
            times = np.around(data['dopa_time'], 6).values
            data['dopa_time'] = times

        except:  # if dopa is df-index
            times = np.around(data.index.values, 6)
            data.insert(loc=0, column='dopa_time', value=times)
        
        arr_keys = data.keys()
    
    elif isinstance(data, np.ndarray):
        
        if data.shape[1] != len(col_names):
            data = data.T
        if data.shape[1] != len(col_names):
            raise ValueError(
                'get_windows() DATA ARRAY SHAPE DOESNT MATCH COL_NAMES'
            )
            
        times_col = np.where(col_names == 'dopa_time')[0][0]
        times = data[:, times_col]
        # times = time_index   # CURRENTLY NUMERICAL INDEX !!!
        arr_keys = col_names

    else:
        raise ValueError('data inserted in get_windows() has wrong datatype')
    
    # remove channels with too many NaNs to prevent deletion of all data
    # this happens due to different missing channels in different recordings
    if remove_nan_timerows:
        good_cols = [sum(isna(data[:, i])) < 10000 for i in range(data.shape[1])]
        # print(good_cols)
        
        # print(data.shape)
        # print(len(arr_keys))
        # print(len(good_cols))
        data = data[:, good_cols]
        arr_keys = list(compress(arr_keys, good_cols))


    nWin = int(fs * winLen_sec)  # n samples within a full window

    # define time (in seconds) after which a new window starts (incl overlap)
    if part_winOverlap > 0:
        sec_hop = winLen_sec * part_winOverlap
    else:
        sec_hop = winLen_sec

    # start with rounded winLen positions
    tStart = round(times[0] / winLen_sec) - (winLen_sec * .5)

    # create 3d array with windows
    arr_list, arr_times_sec = [], []

    # epoch windows based on dopatime-seconds and index of dataframe
    for win0_sec in np.arange(
        tStart, times[-1], sec_hop
    ):
        # SELECT ROWS WITHIN TIME WINDOW
        if isinstance(data, DataFrame):
            wintemp = data.loc[win0_sec:win0_sec + winLen_sec]

        elif isinstance(data, np.ndarray):
            sel = np.logical_and(
                win0_sec < times, times < (win0_sec + winLen_sec)
            )
            wintemp = data[sel]
            print(win0_sec)
            print(wintemp.shape)
            
            if remove_nan_timerows:
                nansel = isna(wintemp).any(axis=1)
                wintemp = wintemp[~nansel]
                print('after nan remove', wintemp.shape)

        ### INCLUDE FILTERING ON PRESENT DATA
        # (skip window if less data present than defined threshold)
        if wintemp.shape[0] < (nWin * min_winPart_present): continue

        ### INCLUDE ACCELEROMETER ACTIVITY FILTERING
        if movement_part_acceptance < 1:  # 1 is default and means no filtering
            move_part = 1 - (sum(wintemp['no_move']) / wintemp.shape[0])
            # skip window if it contains too many movement samples
            if move_part > movement_part_acceptance:
                print(f'\t window skipped due to MOVEMENT ({win0_sec} s)')
                continue

        # NaN-PAD WINDOWS NOT FULLY PRESENT (to fit 3d-array shape)
        if wintemp.shape[0] < nWin:

            rows_pad = nWin - wintemp.shape[0]
            nanpad_arr = np.array(
                [[np.nan] * wintemp.shape[1]] * rows_pad
            )
            wintemp = np.concatenate([wintemp, nanpad_arr], axis=0)
            arr_times_sec.append(win0_sec)
        
        else:
            # no nan-padding necessary
            arr_times_sec.append(win0_sec)

        arr_list.append(wintemp)  # list with 2d arrays
    
    win_array = np.array(arr_list)  # create 3d array
    


    if return_as_class:
        
        windows = windowedData(
            data=win_array,
            fs=fs,
            keys=arr_keys,
            win_starttimes=arr_times_sec
        )
        return windows

    else:
        
        return win_array, fs, arr_keys, arr_times_sec


@dataclass(repr=True, init=True,)
class windowedData:
    """
    Class to store and access data, keys,
    and timestamps from windowed data defined by
    get_windows() above.
    """
    data: array
    fs: int
    keys: list
    win_starttimes: list


def window_to_epochs(
    win_array,
    fs,
    epochLen_sec,
    remove_nan: bool = True,
    mne_format: bool = True,
):
    """
    Takes 2d array of one window (e.g. 60 seconds)
    with shape n-samples-per-window x n-columns,
    and divides windows in smaller epochs (eg for
    mne-multi-variate connectivity analyses).
    Resulting array has shape
    [n-epochs x n-samples-per-epoch x n-columns]
    """
    assert len(win_array.shape) == 2, (
        f'win_array needs to be 2d, got shape {win_array.shape}'
    )
    # check shape of input array
    if win_array.shape[0] < win_array.shape[1]:
        win_array = win_array.T
    n_channels = win_array.shape[1]
    
    if remove_nan:
        nan_sel = isna(win_array).any(axis=1)
        win_array = win_array[~nan_sel]
    
    # define number of epochs fitting in array
    epoch_samples = int(fs * epochLen_sec)
    n_epochs = int(win_array.shape[0] / epoch_samples)
    
    # removing redundant samples not fitting in epoch at end
    samples_to_incl = n_epochs * epoch_samples
    win_array = win_array[:samples_to_incl]

    epoched_array = win_array.reshape(
        (n_epochs, epoch_samples, n_channels), order='C',
    )  # in format n_epochs, n_samples, n_channels (columns)

    if mne_format:  # requires n_epochs, n_channels, n_samples
        epoched_array = np.array(
            [e.T for e in epoched_array]
        )

    assert ~ isna(epoched_array).any(), (
        "resulting epoched 3d-array has NaN's"
    )

    return epoched_array


from mne import create_info, EpochsArray

def create_mne_epochs(
    epoched_windows, fs, ch_names,
    pick_only_ephys: bool = True,
):
    """
    Create MNE-Objects (Epoched) from all
    3d-arrays (n_epochs, n_channels, n_times)
    per window, for all epoched windows in list

    Input:
        - epoched_window_list: list with 3d-array
            arrays per window
        - fs
        - ch_names: corresponding with 2nd axis of
            3d arrays
    
    Returns:
        - list_mne_epochs: list with mne-EpochedArray
    """
    assert len(ch_names) == epoched_windows[0].shape[1], (
        'length of ch_names and n_channels in epoched '
        'window 3d-arrays does not match'
    )
    # only include ephys data in mne-Epochs
    if pick_only_ephys:
        ephys_sel = [
            np.logical_or('ECOG' in col, 'LFP' in col)
            for col in ch_names
        ]
        ch_names = list(compress(ch_names, ephys_sel))
        epoched_windows = [
            e[:, ephys_sel, :] for e in epoched_windows
        ]
    
    # create obligatory mne-info
    mne_info = create_info(
        ch_names=list(ch_names),
        sfreq=fs,
        ch_types=['eeg'] * len(ch_names)
    )
    # convert np-arrays into mne Epoched Arrays
    list_mne_epochs = []
    for e, epochs_arr in enumerate(epoched_windows):
        # loop over all 3d array within list
        new_arr = EpochsArray(
            epochs_arr,
            info=mne_info,
            verbose=False,
            # events=events,
            # event_id={'arbitrary': 1}
        )
        list_mne_epochs.append(new_arr)
        if e % 10 == 0:
            print(f'...added MNE Epoch #{e}')
    
    return list_mne_epochs


def get_noNanSegm_from_singleWindow(
    windat,
    segLen_n: int,
    fs: int,
    part_segmOverlap=0,
    win_times=np.array([0]),
):
    """
    Reshapes 1-dimensional time-series data
    (typically of one window), in 2-d
    array of segments, removes segments with nan's.
    If parallel timestamps are given,
    corresponding timestamps are returned

    Input:
        - windat (array): uni-dimensional timeseries
        - segLen_n (int): number of samples per segment
        - n_overlap (int): number of samples of
            overlap between consecutive segments
        - win_times (array): defaults to zero-array
            if not defined, if given: should be
            all timestamps corresponding to windat
    
    Returns:
        - windat (nd-array): 2d-array with segmented
            timeseries data. Number of segments
            depending on overlap
        - win_times: if given win_times was defined:
            corresponding start-times to windat
    """
    if win_times.size == 1:
        timing = False
    else:
        timing = True
        if len(windat) != len(win_times):

            raise ValueError(
                'Unequal shapes of windat and win_times'
            )

    # get rid of redundant data at end of window
    n_segmOverlap = int(part_segmOverlap * fs)
    nSegments = int(len(windat) / (segLen_n - n_segmOverlap))
    windat = windat[:int((segLen_n - n_segmOverlap) * nSegments)]

    if timing: win_times = win_times[:len(windat)]

    # reshape to [n-segments x n-samples per segment]
    if n_segmOverlap == 0:
        segdat = windat.reshape(nSegments, segLen_n)
        if timing: seg_times = win_times[::segLen_n]
    
    else:
        tempdat = []
        if timing: temptimes = []
        
        for i in np.arange(nSegments):

            istart = int(i * (segLen_n - n_segmOverlap))
            if (len(windat) - istart) < segLen_n: continue  # not enough samples left
            tempdat.append(windat[istart:istart + segLen_n])

            if timing:
                temptimes.append(win_times[istart])
        
        segdat = np.array(tempdat)
        if timing: seg_times = np.array(temptimes)
    
    if timing:
        assert len(seg_times) == segdat.shape[0], print(
            'segment times and number is not equal'
        )

    
    # delete segments (rows) with nan's
    nanrows = [np.isnan(list(row)).any() for row in segdat]
    segdat = segdat[~np.array(nanrows)]
    if timing: seg_times = seg_times[~np.array(nanrows)]

    if timing: return segdat, seg_times
    
    else: return segdat



def get_epochedData_perChannel(
    windows,
    channelName: str,
    fs,
    winLen_sec: float = 180,
    segLen_sec: float = .5,
    part_segmOverlap: float = .5,
    return_3d_segmData: bool = False,
    return_noNan_windows: bool = False,
):
    """
    Function to combine all 2d-arrays of segmented
    data per window.
    Takes one ephys-channel, per time!

    Inputs:
        - windows: class with data/keys/win_starttimes
            resulting from windowedData() class
        - channelName: channel name to handle
        - fs: sampling freq
        - winLen_sec: segment length in seconds
        - part_segmOverlap: part of overlap in small
            segment epoch
    
    Returns:
        - segDat: 3d-array (# windows, max # of segments
            per window, # samples per segment). All
            segment rows with data are without NaN;
            full rows of NaNs are padded at the end
            of windows, to enable 3d-stacking
        - segTimes: chronological timestamps of all
            segments without nan
    
    Raises:
        - valueError: if return_3d_segmData and return_
            noNan_windows are both True or None is True
    """
    if np.logical_or(
        return_3d_segmData == 0 and return_noNan_windows == 0,
        return_3d_segmData == 1 and return_noNan_windows == 1,
    ):
        raise ValueError('Choose EXACTLY 1 return method')
    
    i_times = np.where(windows.keys == 'dopa_time')[0][0]       
    i_ch = np.where(windows.keys == channelName)[0][0]
    
    if return_3d_segmData:
        # max shape of array, aka max #-segments in window
        max_n_segs = (fs * winLen_sec) / (fs * segLen_sec)
        
        for i_win in range(windows.data.shape[0]):
            # extract segmented data per window (2d array) with
            # shape: # segments-in-window, #samples-per-segm)
            tempDat, tempTimes = get_noNanSegm_from_singleWindow(
                windows.data[i_win, :, i_ch],
                fs=fs,
                segLen_n=int(fs * segLen_sec),
                part_segmOverlap=part_segmOverlap,
                win_times=windows.data[i_win, :, i_times],
            )
            
            # pad with nans to equalise shape to max-shape
            if tempDat.shape[0] < max_n_segs:

                pad = np.array([
                    [np.nan] * tempDat.shape[1]  # defines # columns
                ] * int(max_n_segs - tempDat.shape[0])  # defines # rows
                )
                tempDat = np.concatenate([tempDat, pad], axis=0)
                # times are not padded, because nan's are removed from data later
            
            if i_win == 0:  # first iteration create the variable
                tempSegDats = [tempDat,]
                segTimes = tempTimes

            else:  # then store in existing variable
                tempSegDats.append(tempDat)
                segTimes = np.concatenate([segTimes, tempTimes])
            
        segDat = np.stack(tempSegDats)

        epoched_channel = epochedChannel(
            data=segDat,
            segmTimes=segTimes,
            winTimes=windows.win_starttimes
        )


    elif return_noNan_windows:

        newDat = []
        newTimes = []

        for i_win in range(windows.data.shape[0]):
            
            tempdat = windows.data[i_win, :, i_ch]
            # only include full windows w/o NaN-data
            if ~ np.isnan(list(tempdat)).any():
                # convert py-float to np-float if necessary            
                if type(tempdat[0]) == float:
                    tempdat = tempdat.astype(np.float64)

                newDat.append(tempdat)
                newTimes.append(windows.win_starttimes[i_win])

        epoched_channel = epochedChannel(
            data=np.array(newDat),
            winTimes=np.array(newTimes)
        )


    return epoched_channel


@dataclass(repr=True, init=True,)
class epochedChannel:
    """
    Class to store and access (3d) data
    and timestamps of segments per
    windows.
    Contains one ephys-channel per class
    """
    data: array
    winTimes: Any
    segmTimes: array = field(default_factory=lambda: np.array([]))


@dataclass(init=True, repr=True,)
class epochedData_multipleChannels:
    """
    main class contains data and times per
    ephys-Channel (epochedChannel, containing data,
    window-times, and segment-times if applicable).
    
    Data can be epoched in windows and segments, meaning
    data is organised in 3d-arrays, existing of
    n-windows x n-segments (per window) x n-samples
    (per segment). In case of overlap, every window
    or segment has a separate array/row, meaning
    last part of row N can be equal to first part
    of row N + 1.
    OR Data can be epoched in windows, in which only
    windows WITHOUT NaN-values are included.
    """
    windows: Any  # must be class windowedData()
    channels_incl: list
    fs: int
    winLen_sec: float
    segLen_sec: float
    part_segmOverlap: float = .5

    def __post_init__(self,):
        # loop over included channels, and create class
        # with data and times per channel
        for ch in self.channels_incl:

            epochedChannel = get_epochedData_perChannel(
                windows=self.windows,
                channelName=ch,
                fs=self.fs,
                winLen_sec=self.winLen_sec,
                segLen_sec=self.segLen_sec,
                part_segmOverlap=self.part_segmOverlap,
                return_3d_segmData=False,
                return_noNan_windows=True,
            )

            setattr(
                self,
                ch,
                epochedChannel
            )