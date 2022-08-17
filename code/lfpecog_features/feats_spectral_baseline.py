'''
Functions to define spectral baselines for neuro-
physiology data (LFP and ECOG) in ReTune's Dyskinesia Project

Based on the run that is analyzed, a class is createsd
with baseline raw signals, PSDs (Welch) and wavelet
decomposition based on predefined minutes from the Rest
recording of the corresponding Session. Minutes from Rest
recording are selected to contain no movement.
'''
# Import general packages and functions
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.signal import welch, cwt, morlet2

from lfpecog_features.feats_read_proc_data import subjectData
import lfpecog_features.moveDetection_preprocess as movePrep

@dataclass(init=True, repr=True,)
class createBaseline:
    """
    Create baseline values for spectral features
    per datatype, and per available ephys-channel.
    Takes first Rest-task minutes around Dopamine-
    intake and selects windows without movement 
    (acceleration defined).

    Main Class per Subject contains baseline values
    per subClasses which are defined per ephys-
    channel-name, or combi of names (LFPxECoG)
    for e.g. coherence-values.
    """
    subData: Any

    def __post_init__(self,):
        
        for dType in self.subData.dtypes:

            if dType[:3] not in ['eco', 'lfp']:
                continue
                
            df = getattr(self.subData, dType).data
            fs = int(getattr(self.subData, dType).fs)

            for key in df.keys():

                if not np.logical_or(
                    'LFP' in key, 'ECOG' in key
                ): continue

                (
                    blWins,
                    blTimes,
                    fs,
                    overlap
                ) = find_baseline_windows(
                    self.subData, dType, key
                )

                setattr(  # set all defined baseline values
                    self,  # as classes directly under their channelnames
                    key,
                    getBaselineValues(blWins, blTimes, fs),
                    overlap
                )
        
                # calculate every ECoG x LFP combi
                if 'ECOG' in key:

                    if 'R' in key: side = 'right'
                    elif 'L' in key: side = 'left'

                    lfpDf = getattr(
                        self.subData,
                        f'lfp_{side}'
                    ).data

                    for combiCh in lfpDf.keys():

                        if 'LFP' not in combiCh: continue

                        (
                            lfpWins,
                            lfpTimes,
                            fs,
                            overlap
                        ) = find_baseline_windows(
                            self.subData, f'lfp_{side}', combiCh
                        )

                        # TODO: create dataclass which creates
                        # cross-PSD and coherence ()
                        # TODO: include exclusion of windows
                        # which are not present in both ECoG
                        # and LFP windows (take all window-
                        # start times, exclude if not present
                        # in other data-times-array)


            

@dataclass(init=True, repr=True,)
class getBaselineValues:
    """
    Calculate the spectral feature values of
    selected baseline windows (containing Rest) for
    one given channel.

    Input:
        - blData (ndarray): containing windows  
            of pre-selected Rest data
        - blTimes (ndarray): dopa-timed timestamps
            corresponding to window starts
        - fs (int): Fs
        - overlap: % of window-length used as overlap,
            between 0 and 1, or None

    Class output:
        - psd (dict): f and psd from Welch PSD, nperseg
            default as 0.5 seconds
        - wav (dict): wavelet containing time, freq, and psd
        - wavlog (dict): wavelet containing time, freq,
            and psd (log-transformed)
    """
    blData: Any
    blTimes: Any
    fs: int
    overlap: Any
            
    def __post_init__(self,):
        # calculate psd with welch
        f_psd, psd = welch(
            self.blData,
            fs=self.fs,
            nperseg=self.fs // 2,
        )
        self.psd = {
            'fs': f_psd,
            'psd': psd
        }
        # calculate wavelet
        w = 8
        base_f = np.linspace(1, self.fs / 2, 100)
        widths = (
            self.fs * w) / (2 * base_f * np.pi
        )
        # make one-dimensional data for wavelet
        # !!! PM: may not be originally chronological !!!
        if self.overlap == None:
            # no overlap, no double-data: incl all data
            wavSig = self.blData.reshape(
                1,
                self.blData.shape[0] * self.blData.shape[1]
            )
        
        elif self.overlap == .5:
            # exclude double-data due to overlap
            wavSig = self.blData[::2, :]  # take every other window
            wavSig = wavSig.reshape(
                1,
                wavSig.shape[0] * wavSig.shape[1]
            )
    
        time = np.arange(len(wavSig))
        scp_cfs = cwt(
            wavSig, morlet2,
            widths=widths, w=w, dtype='complex128'
        )
        self.wav = {
            'time': time,
            'freq': base_f,
            'psd': np.abs(scp_cfs)
        }
        self.wavlog = {
            'time': time,
            'freq': base_f,
            'psd': np.log10(np.abs(scp_cfs))
        }
    

def find_baseline_windows(
    subData, dType, ephysCh, nSec_per_win=5,
    overlapPerc=.5, acc_thr=1e-6, ):
    """
    Create for first 10-minutes of data, windows
    where task is Rest, there is no activity in both
    acc-sides, and there are no nan's in the ephys data.

    Do this for each ephys-channel. And order by dopa-time.
    Combine afterwards dual channel availibility for eg
    coherence baselinen determination.

    Input:
        - subData: subjectData Class
        - dType (str): data type e.g. "lfp_left"
        - ephysCh (str): name of ephys channel within
            dType dataframe
        - nSec_per_win (int): seconds per window
        - overlapPerc (float): percentage of nperseg
            used for overlap between neighbouring windows,
             defined 0 - 1. If None: no overlap is used.
        - mov_thr: signal vector magn cut off for
            movement
    """
    assert 0 < overlapPerc < 1

    ephysFs = getattr(subData, dType).fs
    nperseg = ephysFs * nSec_per_win
    times = getattr(subData, dType).data['dopa_times']
    sig = getattr(subData, dType).data[ephysCh].values

    acc_keys = [
        'ACC' in k for k in subData.acc_left.data.keys()
    ]
    if len(sum(acc_keys)) == 0:
        acc_keys = [
            k in ['X', 'Y', 'Z'] for k in subData.acc_left.data .keys()
        ]

    svm = {}
    for side in ['left', 'right']:
        
        accDat = getattr(
            subData, f'acc_{side}'
        ).data[acc_keys].values
        svm[side] = movePrep.signalvectormagn(accDat)

        assert len(sig) == len(svm[side]), print(
            f'Non-equal sizes of ACC {side} and {dType} {ephysCh}'
        )

    blWin_tempLists = []  # empty lists to fill with data and ...
    blWin_times = []  # ... corresponding time points

    if overlapPerc == None:  # define sample-distance between two windows
        winHop = nperseg
    else:
        winHop = int(nperseg * overlapPerc)

    # defines number of first minutes to include in baseline data
    min_incl = 8

    for iStart in np.arange(0, ephysFs * 60 * min_incl, winHop):

        for side in ['left', 'right']:

            if any(
                svm['left'][iStart:iStart + nperseg] > acc_thr
            ): continue  # skip window if movement in ACC
        
        if any(
            np.isnan(sig[iStart:iStart + nperseg])
        ): continue  # skip window if nans present

        # add windows without movement, without nan's
        blWin_tempLists.append(sig[iStart:iStart + nperseg])
        blWin_times.append(times[iStart])
    
    blDatWins = np.array(blWin_tempLists)  # array of bl-data windows
    blDatTimes = np.array(blWin_times)  # array of bl-data times

    assert blDatWins.shape[0] == len(blDatTimes), print(
        'Baseline # data-windows and # times are not equal'
    )

    print(
        f'\tBaseLine for {dType}, {ephysCh}: {len(blWin_times)} '
        f'windows of {nperseg} samples ({ephysFs} Hz)')

    return blDatWins, blDatTimes, ephysFs, overlapPerc



# class EphyBaseData:
#     """Create data per ecog/ lfp-L / lfp-R"""
#     def __init__(self, runClass, runname, dtype):
#         self.dtype = dtype
#         base_ind_f = os.path.join(  # make sure projectpath is cwd
#             'data/analysis_derivatives/'
#             'base_spectral_run_indices.json'
#         )
#         with open(base_ind_f) as jsonfile:
#             base_ind = json.load(jsonfile)
#         sub = runClass.sub
#         ses = runClass.ses
#         base_ind = base_ind[sub][ses][dtype]

#         for row, level in enumerate(
#             getattr(runClass, f'{dtype}_names')
#         ):
#             # iterate all levels, skip first time-row
#             if np.logical_and(row == 0, level == 'time'):
#                 continue
#             setattr(
#                 self,
#                 level,  # key of attr
#                 EphyBaseLevel(
#                     runClass,
#                     dtype,
#                     level,
#                     row,
#                     base_ind
#                 )
#             )
    
#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__} Class '
#             f'for {self.dtype}')


# class EphyBase():
#     '''Baseline creation for spectral analyses'''
#     def __init__(self, runClass, runname: str):

#         self.runClass = runClass.runs[runname]
#         self.runname = runname
#         self.ecog = EphyBaseData(
#             runClass=self.runClass,
#             runname=self.runname,
#             dtype='ecog',
#         )
#         self.lfp_left = EphyBaseData(
#             runClass=self.runClass,
#             runname=self.runname,
#             dtype='lfp_left',
#         )
#         self.lfp_right = EphyBaseData(
#             runClass=self.runClass,
#             runname=self.runname,
#             dtype='lfp_right',
#         )
    
#     def __repr__(self):
#         return (f'{self.__class__.__name__}: '
#             f'Main EphysBaseline Class')

