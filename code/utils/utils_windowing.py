"""
Utility-Functions used for windowing
and epoching purposes during signal-
processing
"""
# Import public packages
import numpy as np

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
            parallel to windat
    
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

    # get rid of redundant data at end of window
    nSegments = int(len(windat) / (segLen_n - n_overlap))
    windat = windat[:int((segLen_n - n_overlap) * nSegments)]

    if timing: win_times = win_times[:len(windat)]

    # reshape to [n-segments x n-samples per segment]
    if n_overlap == 0:
        windat = windat.reshape(nSegments, segLen_n)
        if timing: win_times = win_times[::segLen_n]
    
    else:
        tempdat = []
        if timing: temptimes = []
        
        for i in np.arange(nSegments):

            istart = int(i * (segLen_n - n_overlap))
            if (len(windat) - istart) < segLen_n: continue  # not enough samples left
            tempdat.append(windat[istart:istart + segLen_n])

            if timing:
                temptimes.append(win_times[istart])
        
        windat = np.array(tempdat)
        if timing: win_times = np.array(temptimes)
    
    if timing: assert (
        len(win_times) == windat.shape[0]
    ), print('segment times and number is not equal')

    
    # delete segments (rows) with nan's
    nanrows = [np.isnan(list(row)).any() for row in windat]
    windat = windat[~np.array(nanrows)]
    if timing: win_times = win_times[~np.array(nanrows)]

    if timing: return windat, win_times
    
    else: return windat



def get_3dArray_from_segmWindows(

):
    ### TODO: ADJUST FOR PY USE

    sub = '008'
    fs=1600
    winLen_sec = 180
    chNames = {
        'seed': ['LFP_L_1_2',],
        'target': ['ECOG_L_02',],
    }
    segLen_sec = .25

    segDat, segTimes = {}, {}

    winDat = {}
    # perform per sub
    # data_arr, data_keys, dataWinTimes = ftsMain.get_windows(
    #     rest_dfs[sub],
    #     fs=fs,  
    #     winLen_sec=winLen_sec
    # )
    winDat[sub] = windowedData(data_arr, data_keys, dataWinTimes)

    i_times = np.where(winDat[sub].keys == 'dopa_time')[0][0]

    for ch_group in ['seed', 'target']:
    # later add all needed channel-names and extract noNan-Segments in once
        for ch in chNames[ch_group]:
            
            print(f'start {ch}')
            
            i_ch = np.where(winDat[sub].keys == ch)[0][0]

            for i_win in range(winDat[sub].data.shape[0]):
                # define max sizes array
                max_n_segs = (fs * winLen_sec) / (fs * segLen_sec)
                # extract segmented data per window (2d array)
                tempDat, tempTimes = utils_win.get_noNanSegm_from_singleWindow(
                    winDat[sub].data[i_win, :, i_ch],
                    segLen_n=int(fs * segLen_sec),
                    n_overlap=0,
                    win_times=winDat[sub].data[i_win, :, i_times],
                )
                # pad with nans to equalise shapes
                if tempDat.shape[0] < max_n_segs:

                    pad = [
                        [np.nan] * tempDat.shape[1]  # defines # columns
                    ] * int(max_n_segs - tempDat.shape[0])  # defines # rows
                    tempDat = np.concatenate([tempDat, pad], axis=0)
                    
            
                if i_win == 0:
                    tempSegDats = [tempDat,]
                    segTimes[ch] = [tempTimes, ]
                else:
                    tempSegDats.append(tempDat)
                    segTimes[ch].append(tempTimes)
                
                segDat[ch] = np.stack([tempSegDats])
                # remove 4-th 1 dimension
                # segDat[ch], segTimes[ch] = 
                # print(i_win, tempDat.shape, tempTimes.shape)

# calculate features over segmented data
# incl times0checking per ch-combi

# segmentConnectFts(
#     sub = '012',
#     data_arr
#     data_keys: list
#     winTimes: list
#     fs: int
#     seed_ephyGroup: str
#     target_ephyGroup: str
#     segLen_sec: float = .5
#     part_overlap: float = 0.