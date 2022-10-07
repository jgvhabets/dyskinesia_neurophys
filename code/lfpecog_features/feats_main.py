'''
Functions to extract spectral from neurophysiology
data (LFP and ECOG) in ReTune's Dyskinesia Project

Creates features for moments classified as [Rest,
Movement, Tap] and annotates the time relative to
L-Dopa intake.
'''
# Import general packages and functions
from dataclasses import field, dataclass
import os
from re import sub
from typing import Any
import json
import numpy as np
from scipy.signal import welch, cwt, morlet2

# Import own functions
from lfpecog_features.feats_read_proc_data import subjectData
import lfpecog_features.feats_spectral_baseline as baseLine
import lfpecog_features.feats_ftsExtr_Classes as ftClasses
import lfpecog_features.feats_helper_funcs as ftHelpers


"""
Include in feature class:
    per window (1 s?, 0.5 s?, .25 sec with >> sample freq):
        - state (nomove, tap (after tap/taprest), move)
        - beta peak
        - gamma peak
        - beta peak bandw
        - peak heigth (Swann '16)
"""

@dataclass(init=True, repr=True,)
class create_dopaTimed_ftSpace:
    """
    Extract feature-set alligned to dopa-times.

    Input:
        - sub (str): subject-code
        - window_len (int): seconds of windows, segments
            are half windows per default
        - win_overlap (float): part of window that  
            overlaps, default 0.5
        - subData: subjectData Class with data
        - subBaseline Class resulting from createBaseline()
            if not given, no baseline subtraction is
            performed
    """
    sub: str
    window_len: int = 5
    win_overlap: float = .5
    nSec_Seg = .5
    subData: Any = None
    subBaseline: Any = None
    n_bl_minutes: int = 5
    # excl_task: list = field(default_factory = ['Free'])
    
    def __post_init__(self,):
        ### TODO: include task and acc-labels

        if self.subData == None:

            self.subData = subjectData(
                sub=self.sub,
                data_version=self.data_version,
                project_path=self.project_path,
            )

        if self.subBaseline == None:

            self.subBaseline = baseLine.createBaseline(
                    subData=self.subData,
                    nSec_blWins=self.window_len,
                    nSec_Seg=self.nSec_Seg,
                    n_bl_minutes=self.n_bl_minutes,
                )

        for dType in self.subData.dtypes:

            if dType[:3] not in ['eco', 'lfp']:
                continue  # only include ephys-types
                
            setattr(
                self,
                dType,
                dType_ftExtraction(
                    subData=self.subData,
                    dType=dType,
                    win_len=self.window_len,
                    win_overlap=self.win_overlap,
                    baseline=self.subBaseline,
                    segSec=self.nSec_Seg,
                )
            )

            print(f'\t{dType} finished ({self.sub})')
        
        print(f'\n\tFinished {self.sub}\n')



@dataclass(init=True, repr=True,)
class dType_ftExtraction:
    """
    Extract feature-set alligned to dopa-times from
    all ephys-channels from one dType (LFP L/R, ECoG)

    Input:
        - sub (str): subject-code
        - incl_baseline (bool): set True to include
            baseline values
    """
    subData: Any
    dType: str
    win_len: Any
    baseline: Any
    win_overlap: float = .5
    segSec: float = .5


    def __post_init__(self,):

        df = getattr(self.subData, self.dType).data
        fs = int(getattr(self.subData, self.dType).fs)
        nperseg = int(fs * self.segSec)

        for ch_key in df.keys():

            if not np.logical_or(
                'LFP' in ch_key, 'ECOG' in ch_key
            ): continue  # skip non-ephys channels

            # create data array per window
            win_arr, win_times = get_windows(
                sigDf=df,
                fs=fs,
                ch=ch_key,
                winLen_sec=self.win_len,
                overlap=self.win_overlap,
            )

            # find corresponding tasks per window
            win_tasks = get_window_tasks(
                win_times,
                df['dopa_time'].values,
                df['task'].values
            )

            # calculate PSD per window
            setattr(  # set all defined baseline values
                self,  # as classes directly under their channelnames
                ch_key,
                ftClasses.getFeatures_singleChannel(
                    winData=win_arr,
                    winTimes=win_times,
                    fs=fs,
                    nperseg=nperseg,
                    overlap=self.win_overlap,
                    win_tasks=win_tasks,
                )
            )

            # Extract combi ECoG-LFP features
            if not 'ecog' in self.dType.lower(): continue

            # PM: transform into functions which finds combi ch and windows
            ## e.g. ftClasses.getCombiWindows()
            if 'R' in ch_key: side = 'right'
            elif 'L' in ch_key: side = 'left'

            lfpDf = getattr(
                self.subData,
                f'lfp_{side}'
            ).data

            for combi_ch in lfpDf.keys():

                if 'lfp' not in combi_ch.lower(): continue  # skip non-ephys lfp-channels

                # find available windows in stn-channel
                (
                    lfpWins,
                    lfpTimes,
                ) = get_windows(
                    sigDf=lfpDf,
                    ch=combi_ch,
                    fs=fs,
                    winLen_sec=self.win_len,
                    overlap=self.win_overlap,
                )

                # skip stn-channel if no windows available
                if len(lfpTimes) == 0: continue
                
                try:  # assign ch-combi-specific baseline
                    combiCh_baseline = getattr(
                        self.baseline,
                        f'{ch_key}_{combi_ch}'
                    )
                # if the ch-combi is not available
                except AttributeError:  # fill to prevent
                    combiCh_baseline = None

                # set basal-ganglia-cortex baseline values
                setattr(  # stored as classes with 'ecog-ch_stn-ch' name
                    self,
                    f'{ch_key}_{combi_ch}',
                    ftClasses.getFeatures_BGCortex(
                        ecogDat=win_arr,
                        ecogTimes=win_times,
                        ecogCh=ch_key,
                        stnDat=lfpWins,
                        stnTimes=lfpTimes,
                        stnCh=combi_ch,
                        fs=fs,
                        nperseg=nperseg,
                        overlap=self.win_overlap,
                        extr_baseline=False,
                        combiCh_baseline=combiCh_baseline,
                        ecogTasks=win_tasks,
                    )
                )



def get_windows(
    sigDf, fs, ch, winLen_sec, overlap=.5,
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
        - ch (str): key name string
        - win_len: length of window in seconds
        - overlap (float): part of window used as overlap
            between consecutive windows
    
    Returns:
        - win_arr (arr): 2d array of n-windows x n-samples
        - win_times (list): corresponding times of window
            start dopa-times
    """
    # # define ephys-signal en dopa-times
    # sig = sigDf[ch].values
    
    try:  # if dopa-time is as column
        times = np.around(sigDf['dopa_time'], 5).values
        sigDf['dopa_time'] = times
        # del(sigDf['dopa_time'])

    except:  # if dopa is df-index
        times = np.around(sigDf.index.values, 5)
        sigDf.insert(loc=0, column='dopa_time', value=times)

    nWin = int(fs * winLen_sec)  # n samples within a window

    # define number of samples for overlap
    if np.logical_or(
        overlap == 0, overlap == None
    ):  # if no overlap, move a full window-length further per window

        nHop = nWin
    
    else:
        nHop = int(nWin * overlap)  # n samples used as overlap

    win_list, win_times = [], []

    # start with rounded 3 min positions
    tStart = round(times[0] / winLen_sec) - (winLen_sec * .5)
    tStart_sec = tStart * winLen_sec
    # create 3d array with windows
    nWin = 0
    arr_list, arr_times_sec = [], []

    for win0_sec in np.arange(
        tStart, times[-1], winLen_sec
    ):  # currently no overlap included in windowing

        wintemp = sigDf.loc[win0_sec:win0_sec + winLen_sec]

        # skip window less than half present
        if wintemp.shape[0] < (winLen_sec * fs * .5): continue

        # nan-pad windows more than half present
        elif wintemp.shape[0] < (winLen_sec * fs):

            rows_pad = (winLen_sec * fs) - wintemp.shape[0]
            pad = ftHelpers.nan_array((rows_pad, wintemp.shape[1]))
            wintemp = np.concatenate([wintemp, pad], axis=0)
            arr_times_sec.append(win0_sec)
        
        else:
            # no pnan-padding necessary
            arr_times_sec.append(win0_sec)
        

        arr_list.append(wintemp)
        nWin += 1
    
    win_array = np.array(arr_list)
    arr_keys = sigDf.keys()

    return win_array, arr_keys, arr_times_sec


    ### PREVIOUS SINGLE SIG-CHANNEL BASED METHOD
    # for win_i0 in np.arange(0, len(sig) - nWin, nHop):

    #     # skip window if data is not consecutive
    #     if times[win_i0 + nWin] - times[win_i0] > winLen_sec:

    #         continue

    #     winSig = sig[win_i0:win_i0 + nWin]
    #     win_list.append(winSig)
    #     win_times.append(times[win_i0])
    
    # # convert list of lists into np array
    # win_arr = np.array(win_list)

    # return win_arr, win_times


def get_window_tasks(
    win_times, times, tasks
):
    """
    Find matching task for the window-
    times, based on the total parallel
    lists of dopa-times and tasks

    Input:
        - win_times: times of selected windows
        - times: all dopa times of dataframe
        - tasks: all tasks corresponding to
            all dopa-times
    
    Returns:
        - win_tasks (list): matching tasks to
            win_times
    """
    try:  # convert Series to Array and decrease size
        times = times.values[::60]
        tasks = tasks.values[::60]

    except:  # in case not given as pd.Series
        times = times[::60]
        tasks = tasks[::60]

    win_tasks = []

    for t in win_times:
        # find time closest to t
        i = np.argmin(abs(t - times))
        # find corresponding task
        tsk = tasks[i]
        win_tasks.append(tsk)
    
    return win_tasks
