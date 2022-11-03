"""
Utility-Functions used for windowing
and epoching purposes during signal-
processing
"""
# Import public packages
import numpy as np
from dataclasses import dataclass
from array import array
from typing import Any


def get_windows(
    sigDf,
    fs,
    ch=None,
    winLen_sec=180,
    part_winOverlap=.0,
    min_winPart_present=.33,
):
    """
    Select from one channel windows with size nWin * Fs,
    exclude windows with nan's, and save corresponding
    dopa-times to included windows.

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
    
    Returns:
        - win_arr (arr): 2d array of n-windows x n-samples
        - win_times (list): corresponding times of window
            start dopa-times
    """
    # # define ephys-signal en dopa-times
    
    try:  # if dopa-time is as column
        times = np.around(sigDf['dopa_time'], 6).values
        sigDf['dopa_time'] = times

    except:  # if dopa is df-index
        times = np.around(sigDf.index.values, 6)
        sigDf.insert(loc=0, column='dopa_time', value=times)

    nWin = int(fs * winLen_sec)  # n samples within a window

    # define seconds after which a new window starts (incl overlap)
    if part_winOverlap > 0:
        sec_hop = winLen_sec * part_winOverlap
    else:
        sec_hop = winLen_sec

    # start with rounded winLen positions
    tStart = round(times[0] / winLen_sec) - (winLen_sec * .5)

    # create 3d array with windows
    winCount = 0  # obsolete currently
    arr_list, arr_times_sec = [], []

    # epoch windows based on dopatime-seconds and index of dataframe
    for win0_sec in np.arange(
        tStart, times[-1], sec_hop
    ):

        wintemp = sigDf.loc[win0_sec:win0_sec + winLen_sec]

        # skip window present less than given threshold
        if wintemp.shape[0] < (nWin * min_winPart_present): continue

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
        winCount += 1
    
    win_array = np.array(arr_list)
    arr_keys = sigDf.keys()

    return win_array, arr_keys, arr_times_sec


@dataclass(repr=True, init=True,)
class windowedData:
    """
    Class to store and access data, keys,
    and timestamps from windowed data.
    """
    data: array
    keys: list
    win_starttimes: list


@dataclass(repr=True, init=True,)
class segmentedChannel:
    """
    Class to store and access (3d) data
    and timestamps of segments per
    windows.
    Contains one ephys-channel per class
    """
    data: array
    segmTimes: array
    winTimes: Any


def get_noNanSegm_from_singleWindow(
    windat, segLen_n: int, n_overlap=0,
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
    nSegments = int(len(windat) / (segLen_n - n_overlap))
    windat = windat[:int((segLen_n - n_overlap) * nSegments)]

    if timing: win_times = win_times[:len(windat)]

    # reshape to [n-segments x n-samples per segment]
    if n_overlap == 0:
        segdat = windat.reshape(nSegments, segLen_n)
        if timing: seg_times = win_times[::segLen_n]
    
    else:
        tempdat = []
        if timing: temptimes = []
        
        for i in np.arange(nSegments):

            istart = int(i * (segLen_n - n_overlap))
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



def get_3dSegmArray_allWindows(
    windows,
    channelName: str,
    fs,
    winLen_sec: float = 180,
    segLen_sec: float = .25,
):
    """
    Function to combine all 2d-arrays of segmented
    data per window.
    Takes one channel, per time!

    Inputs:
        - windows: class with data/keys/win_starttimes
            resulting from windowedData() class
        - channelName: channel name to handle
        - fs: sampling freq
        - winLen_sec: segment length in seconds
    
    Returns:
        - segDat: 3d-array (# windows, max # of segments
            per window, # samples per segment). All
            segment rows with data are without NaN;
            full rows of NaNs are padded at the end
            of windows, to enable 3d-stacking
        - segTimes: chronological timestamps of all
            segments without nan
    """
    i_times = np.where(windows.keys == 'dopa_time')[0][0]       
    i_ch = np.where(windows.keys == channelName)[0][0]
    # max shape of array, aka max #-segments in window
    max_n_segs = (fs * winLen_sec) / (fs * segLen_sec)
    
    for i_win in range(windows.data.shape[0]):
        # extract segmented data per window (2d array) with
        # shape: # segments-in-window, #samples-per-segm)
        tempDat, tempTimes = get_noNanSegm_from_singleWindow(
            windows.data[i_win, :, i_ch],
            segLen_n=int(fs * segLen_sec),
            n_overlap=0,
            win_times=windows.data[i_win, :, i_times],
        )
        
        # pad with nans to equalise shape to max-shape
        if tempDat.shape[0] < max_n_segs:

            pad = np.array([
                [np.nan] * tempDat.shape[1]  # defines # columns
            ] * int(max_n_segs - tempDat.shape[0])  # defines # rows
            )
            tempDat = np.concatenate([tempDat, pad], axis=0)
        
        if i_win == 0:
            tempSegDats = [tempDat,]
            segTimes = tempTimes

        else:
            tempSegDats.append(tempDat)
            segTimes = np.concatenate([segTimes, tempTimes])
        
    segDat = np.stack(tempSegDats)
    winTimes = windows.win_starttimes

    return segDat, segTimes, winTimes


@dataclass(init=True, repr=True,)
class segmArrays_multipleChannels:
    """
    
    """
    windows: Any  # must be class windowedData()
    channels_incl: list
    fs: int

    def __post_init__(self,):

        for ch in self.channels_incl:

            segdat, segtimes, winTimes = get_3dSegmArray_allWindows(
                windows=self.windows,
                channelName=ch,
                fs=self.fs,
            )

            setattr(
                self,
                ch,
                segmentedChannel(
                    data=segdat,
                    segmTimes=segtimes,
                    winTimes=winTimes
                )
            )