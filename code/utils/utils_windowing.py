"""
Utility-Functions used for windowing
and epoching purposes during signal-
processing
"""
# Import public packages
import numpy as np
from dataclasses import dataclass, field
from array import array
from typing import Any


def get_windows(
    sigDf,
    fs,
    winLen_sec=180,
    part_winOverlap=.0,
    min_winPart_present=.66,
    movement_part_acceptance: float = 1.,
    return_as_class=False,
):
    """
    Select from one channel windows with size nWin * Fs,
    exclude windows with nan's, and save corresponding
    dopa-times to included windows.
    Function selects windows based on dopa-times using
    .loc function within a pd.DataFrame.

    TODO: make function hybrid, A: for getting times only,
    B: for getting data nd-array

    Inputs:
        - sigDg (dataframe)
        - fs (int): sample freq
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
        - win_arr (arr): 2d array of n-windows x n-samples
        - win_times (list): corresponding times of window
            start dopa-times
    """ 
    try:  # if dopa-time is as column
        times = np.around(sigDf['dopa_time'], 6).values
        sigDf['dopa_time'] = times

    except:  # if dopa is df-index
        times = np.around(sigDf.index.values, 6)
        sigDf.insert(loc=0, column='dopa_time', value=times)

    nWin = int(fs * winLen_sec)  # n samples within a full window

    # define seconds after which a new window starts (incl overlap)
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

        wintemp = sigDf.loc[win0_sec:win0_sec + winLen_sec]
        
        # skip window if smaller than given presence-threshold
        if wintemp.shape[0] < (nWin * min_winPart_present): continue

        move_part = 1 - (sum(wintemp['no_move']) / wintemp.shape[0])
        # skip window if it contains too many movement samples
        if move_part > movement_part_acceptance:
            print(f'\t window skipped due to MOVEMENT ({win0_sec} s)')
            continue

        # nan-pad windows which are not completely present
        elif wintemp.shape[0] < (winLen_sec * fs):

            rows_pad = nWin - wintemp.shape[0]
            nanpad_arr = np.array(
                [[np.nan] * wintemp.shape[1]] * rows_pad
            )
            wintemp = np.concatenate([wintemp, nanpad_arr], axis=0)
            arr_times_sec.append(win0_sec)
        
        else:
            # no nan-padding necessary
            arr_times_sec.append(win0_sec)
        

        arr_list.append(wintemp)
    
    win_array = np.array(arr_list)
    arr_keys = sigDf.keys()


    if return_as_class:
        
        windows = windowedData(
            win_array, arr_keys, arr_times_sec
        )
        return windows

    else:
        
        return win_array, arr_keys, arr_times_sec


@dataclass(repr=True, init=True,)
class windowedData:
    """
    Class to store and access data, keys,
    and timestamps from windowed data defined by
    get_windows() above.
    """
    data: array
    keys: list
    win_starttimes: list



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