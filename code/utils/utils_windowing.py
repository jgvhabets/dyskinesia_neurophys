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

from lfpecog_analysis.prep_movement_psd_analysis import (
    select_taps_out_window, get_tap_times
)

def get_windows(
    data,
    fs,
    col_names = None,
    winLen_sec = 180,
    part_winOverlap = .5,
    min_winPart_present = .66,
    EXCL_TAPS=False,
    DATA_VERSION=None,
    sub=None,
    remove_nan_timerows: bool = True,
    movement_part_acceptance: float = 1.,
    return_as_class = False,
    only_ipsiECoG_STN = True,
    verbose: bool = False
):
    """
    Select windows with size nWin * Fs,
    exclude windows with nan's, and save corresponding
    dopa-times to included windows.
    Function selects windows based on dopa-times.


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
        - verbose: print debugging info, defaults False
    
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
    # import tap timings if necessary
    if EXCL_TAPS:
        tap_starts, tap_ends = get_tap_times(sub=sub,
                                             return_in_secs=True,
                                             DATA_VERSION=DATA_VERSION,
                                             tap_border_sec=1)

    if isinstance(data, DataFrame):
        try:  # if dopa-time is as column
            times = np.around(data['dopa_time'], 6).values
            data['dopa_time'] = times

        except:  # if dopa is df-index
            times = np.around(data.index.values, 6)
            data.insert(loc=0, column='dopa_time', value=times)
        
        arr_keys = data.keys().copy()
        data = data.values()
    
    elif isinstance(data, np.ndarray):
        
        if data.shape[1] != len(col_names):
            data = data.T
        if data.shape[1] != len(col_names):
            raise ValueError(
                'get_windows() DATA ARRAY SHAPE DOESNT MATCH COL_NAMES'
            )
            
        times_col = np.where(c == 'dopa_time' for c in col_names)[0][0]
        times = data[:, times_col]
        # times = time_index   # CURRENTLY NUMERICAL INDEX !!!
        arr_keys = col_names.copy()

    else:
        raise ValueError('data inserted in get_windows() has wrong datatype')
    
    # save few ECoG channels in case of many missings (eg 016)
    problem_cols = [
        ('ECOG_R_02_04', 'ECOG_R_02_05'),
        ('ECOG_R_04_05', 'ECOG_R_05_06')
    ]
    for cols in problem_cols:

        if np.logical_and(cols[0] in arr_keys, cols[1] in arr_keys):
            
            i1 = np.where(np.array(arr_keys) == cols[0])[0][0]
            i2 = np.where(np.array(arr_keys) == cols[1])[0][0]

            newcol = np.nansum([data[:, i1], data[:, i2]], axis=0)
            data[:, i1] = newcol
            data = np.delete(data, i2, axis=1)
            arr_keys.remove(cols[1])

    
    # SELECT ONLY IPSILATERAL STN TO ECOG
    if only_ipsiECoG_STN:
        ecog_side = [c for c in arr_keys if 'ECOG' in c][0][5]
        assert ecog_side in ['L', 'R'], print('incorrect ecog side detected')
        if ecog_side == 'R': contra_side = 'L'
        elif ecog_side == 'L': contra_side = 'R'
        del_sel = np.array([f'_{contra_side}_' in c and 'LFP' in c for c in arr_keys])
        data = data[:, ~del_sel]
        arr_keys = np.array(arr_keys)[~del_sel]
    
    # filter out double ecog channels if present
    if sum('ECOG' in c.upper() for c in arr_keys) == 12:
        if sum('SMC_AT' in c.upper() for c in arr_keys) == 6:
            # this is true if 6 ECOG channels were doubled (none-bids convert)
            sel_dbl_ecog = np.array(['SMC_AT' in c.upper() for c in arr_keys])
            if isinstance(arr_keys, list): arr_keys = np.array(arr_keys)
            print(f'...during get_windows(), sorted out doubles: {arr_keys[sel_dbl_ecog]}')
            arr_keys = arr_keys[~sel_dbl_ecog]
            data = data[:, ~sel_dbl_ecog]



    # remove channels with too many NaNs to prevent deletion of all data
    # this happens due to different missing channels in different recordings
    if remove_nan_timerows:
        max_nan_samples = 30 * fs
        good_cols = [sum(isna(data[:, i])) < max_nan_samples for i in range(data.shape[1])]
        bad_cols = [sum(isna(data[:, i])) > max_nan_samples for i in range(data.shape[1])]
        good_col_names = list(compress(arr_keys, good_cols))
        bad_col_names = list(compress(arr_keys, bad_cols))

        print([f'{arr_keys[i]}: {sum(isna(data[:, i]))} NaNs'
               for i in range(data.shape[1])])
        print(f'\n\n\tDELETED bad cols: {bad_col_names} due to >> NaNs')
        print(f'\n\n\tINCLUDED good cols: {good_col_names}')

        data = data[:, good_cols]
        arr_keys = list(compress(arr_keys, good_cols))
        
    nWin = int(fs * winLen_sec)  # n samples within a full window

    # define time (in seconds) after which a new window starts (incl overlap)
    if part_winOverlap > 0:
        sec_hop = winLen_sec * part_winOverlap
    else:
        sec_hop = winLen_sec

    # Variables to define time-range to loop windows with uniform start times over subjects
    tStart = round(times[0] / winLen_sec) - (winLen_sec * .5)
    tStart = tStart * winLen_sec  # convert rounded window number to seconds

    # create 3d array with windows
    arr_list, arr_times_sec = [], []


    # variables for hopping between 2 consecutive windows (to speed up looping)
    winLen_samples = int(winLen_sec * fs)  # sample length full window
    sample_hop = int(sec_hop * fs)
    # variables window inclusion based on minimal window length
    min_winPart_samples = int(min_winPart_present * winLen_samples)
    min_winPart_sec = int(min_winPart_present * winLen_sec)
    # variables to speed up timestamp comparison
    secs_present = times[::int(fs)]  # get roughly every second present in data
    allowed_dist_start = sec_hop
    allowed_dist_end = winLen_sec - min_winPart_sec
    
    # set index start to None at start
    i_start = None

    # LOOP OVER POSSIBLE WINDOWS IN TIME RANGE
    for win0_sec in np.arange(tStart, times[-1], sec_hop
                              ):
        # separate flow for DataFrames
        if isinstance(data, DataFrame):
            wintemp = data.loc[win0_sec:win0_sec + winLen_sec]

        elif isinstance(data, np.ndarray):

            if verbose: print(f'\nDEBUG: start {win0_sec} SEC')
            # if i_start is not defined in previous window
            if not i_start:
                # find closest second to start of window
                min_dist_start = min(abs(win0_sec - secs_present))
                # skip if start-index is too many seconds off to create window
                if min_dist_start > allowed_dist_start:
                    if verbose: print('\tskipped distance')
                    continue
                # define exact start-index if within acceptance range
                i_start = np.argmin(abs(win0_sec - times))  # get window start sample

            # find closest second to end of window
            min_dist_end = min(abs((win0_sec + winLen_sec) - secs_present))
            # if full window is present (last second also in secs_present), select on index
            if min_dist_end == 0:
                wintemp = data[i_start:i_start + winLen_samples]
                wintimes = times[i_start:i_start + winLen_samples]
                i_start += sample_hop  # SAVE START-INDEX TO SAVE MAJOR COMPUTATIONAL TIME
                if verbose: print('\tselected on end-sample-presence')
            # if end of window is too far off, skip window
            elif min_dist_end > allowed_dist_end:
                i_start = None  # reset start index
                if verbose: print('\tskipped on end-distance')
                continue
            
            # if window only partly present, select based on timestamps
            else:
                sel = np.logical_and(win0_sec < times,
                                     times < (win0_sec + winLen_sec))
                wintemp = data[sel]
                wintimes = times[sel]
                i_start = None  # reset start index
                if verbose: print('\tselected on time')
            
            # skip window if less data present than defined threshold  (back up, currently above statements filter out everything)
            if wintemp.shape[0] < min_winPart_samples:
                if verbose: print('\tskipped based on NaNs', win0_sec)
                continue

            if verbose: print(f'\tINCLUDED shape is {wintemp.shape}')


        # ### INCLUDE ACCELEROMETER ACTIVITY FILTERING  (currently only works for DataFrame data)
        # if movement_part_acceptance < 1:  # 1 is default and means no filtering
        #     move_part = 1 - (sum(wintemp['no_move']) / wintemp.shape[0])
        #     # skip window if it contains too many movement samples
        #     if move_part > movement_part_acceptance:
        #         print(f'\t window skipped due to MOVEMENT ({win0_sec} s)')
        #         continue
        
        ### EXCLUDE TAP-epochs for clean-PSDs (data v4.2)
        if EXCL_TAPS:
            win_tap_bool = select_taps_out_window(wintimes,
                                                  tap_starts,
                                                  tap_ends)
            # check whether data is excluded
            if any(win_tap_bool == False):
                # exclude data related to tapping
                wintemp = wintemp[win_tap_bool]
                print(f'\tin win@ {win0_sec}s: tap-removed: {sum(~win_tap_bool) / fs} seconds')


        # NaN-PAD WINDOWS NOT FULLY PRESENT (to fit 3d-array shape)
        if wintemp.shape[0] < nWin:

            rows_to_pad = nWin - wintemp.shape[0]
            nanpad_arr = np.array(
                [[np.nan] * wintemp.shape[1]] * rows_to_pad
            )
            wintemp = np.concatenate([wintemp, nanpad_arr], axis=0)
            arr_times_sec.append(win0_sec)
        
        else:
            # no nan-padding necessary
            arr_times_sec.append(win0_sec)

        arr_list.append(wintemp)  # list with 2d arrays
    
    # last check of window length of included windows
    arr_bool = [r.shape[0] == winLen_samples for r in arr_list]
    arr_list = list(compress(arr_list, arr_bool))
    win_array = np.array(arr_list)  # create 3d array
    # remove starttimes for same windows
    arr_times_sec = list(compress(arr_times_sec, arr_bool))
    print(f'...removed # {sum(~np.array(arr_bool))} windows bcs of length of end of windowing')
    print(f'STARTTIMES length ({len(arr_times_sec)}) vs '
          f'ARRAYS length ({len(win_array)})')

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
    
    i_times = np.where(k == 'dopa_time' for k in windows.keys)[0][0]       
    i_ch = np.where(k == channelName for k in windows.keys)[0][0]
    
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
    NOT USED IN MVC WORKFLOW 
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