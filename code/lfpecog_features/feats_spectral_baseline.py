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
from scipy.signal import welch, cwt, morlet2, csd

from lfpecog_features.feats_read_proc_data import subjectData
import lfpecog_features.moveDetection_preprocess as movePrep
import lfpecog_features.feats_ftsExtr_Classes as ftClasses
import lfpecog_features.feats_spectral_features as spec_feats

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

    Input:
        - subData (class): subject specific Class
            with preprocessed data (subjectData())
        - incl_psd, incl_wav, incl_coh: booleans to
            define whether resp. power spectra,
            wavelet decompositions, and/or coherences
            should be calculated from baseline periods
    
    Returns:
        - class containing
    """
    subData: Any
    incl_psd: bool = True
    incl_wav: bool = False
    incl_coh: bool = True
    n_bl_minutes: int = 5
    nSec_blWins: int = 5

    def __post_init__(self,):
        
        for dType in self.subData.dtypes:

            if dType[:3] not in ['eco', 'lfp']:
                continue
                
            df = getattr(self.subData, dType).data
            fs = int(getattr(self.subData, dType).fs)

            for ch_key in df.keys():

                if not np.logical_or(
                    'LFP' in ch_key, 'ECOG' in ch_key
                ): continue  # skip non-ephys channels

                (
                    blWins,
                    blTimes,
                    fs,
                    overlap
                ) = find_baseline_windows(
                    self.subData,
                    dType,
                    ch_key,
                    n_minutes=self.n_bl_minutes,
                    nSec_per_win=self.nSec_blWins,
                )

                setattr(  # set all defined baseline values
                    self,  # as classes directly under their channelnames
                    ch_key,
                    ftClasses.getFeatures_singleChannel(
                        blWins,
                        blTimes,
                        fs,
                        overlap,
                        incl_psd=self.incl_psd,
                        incl_wav=self.incl_wav
                    )
                )
        
                # calculate every ECoG x LFP combi
                if 'ECOG' in ch_key:

                    if not self.incl_coh: continue

                    if 'R' in ch_key: side = 'right'
                    elif 'L' in ch_key: side = 'left'

                    lfpDf = getattr(
                        self.subData,
                        f'lfp_{side}'
                    ).data

                    for combi_ch in lfpDf.keys():

                        if 'LFP' not in combi_ch: continue  # skip non-ephys lfp-channels

                        # find available windows in stn-channel
                        (
                            lfpWins,
                            lfpTimes,
                            fs,
                            overlap
                        ) = find_baseline_windows(
                            self.subData, f'lfp_{side}', combi_ch
                        )

                        # skip stn-channel if no windows available
                        if len(lfpTimes) == 0: continue

                        # set basal-ganglia-cortex baseline values
                        setattr(  # stored as classes with 'ecog-ch_stn-ch' name
                            self,
                            f'{ch_key}_{combi_ch}',
                            ftClasses.getFeatures_BGCortex(
                                ecogDat=blWins,
                                ecogTimes=blTimes,
                                ecogCh=ch_key,
                                stnDat=lfpWins,
                                stnTimes=lfpTimes,
                                stnCh=combi_ch,
                                fs=fs,
                                overlap=overlap,
                                extr_baseline=True,
                            )
                        )



def find_baseline_windows(
    subData, dType, ephysCh, nSec_per_win=5,
    overlapPerc=.5, n_minutes=5, acc_thr=1e-6, ):
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
             defined 0 - 1. If None: no overlap is used
        - n_minutes (int): number of first minutes of data
            used for baseline definition
        - mov_thr: signal vector magn cut off for
            movement
        
    Returns:
        - blDatWins (array): 2d-array with timewindows as
            rows, # windows == # of rows
        - blDatTimes (array): 1d-array with corresponding
            start times of windows
        - ephysFs (int): sample freq
        - overlapPerc (float): between 0 and 1, overlap
            in windows expressed as part of nperseg
    """
    assert 0 < overlapPerc < 1

    ephysFs = getattr(subData, dType).fs
    accFs = getattr(subData, 'acc_left').fs
    nperseg = ephysFs * nSec_per_win
    times = np.around(getattr(subData, dType).data['dopa_time'], 5)
    sig = getattr(subData, dType).data[ephysCh].values

    acc_keys = [
        'ACC' in k for k in subData.acc_left.data.keys()
    ]
    if sum(acc_keys) == 0:
        acc_keys = [
            k in ['X', 'Y', 'Z'] for k in subData.acc_left.data .keys()
        ]

    svm = {}
    for side in ['left', 'right']:
        
        accDat = getattr(
            subData, f'acc_{side}'
        ).data.values[:, acc_keys]

        svm[side] = movePrep.signalvectormagn(accDat)

    blWin_tempLists = []  # empty lists to fill with data and ...
    blWin_times = []  # ... corresponding time points

    if overlapPerc == None:  # define sample-distance between two windows
        winHop = nperseg
    else:
        winHop = int(nperseg * overlapPerc)

    acc_nSeg = int(nperseg / ephysFs * accFs)

    for ephy_i0 in np.arange(
        0, ephysFs * 60 * n_minutes, winHop
    ):

        # skip window if data is not consecutive
        if (
            times[ephy_i0 + nperseg] - times[ephy_i0]
         ) > nSec_per_win:

            continue

        acc_i0 = int(ephy_i0 / ephysFs * accFs)

        for side in ['left', 'right']:

            if any(
                svm['left'][acc_i0:acc_i0 + acc_nSeg] > acc_thr
            ): continue  # skip window if movement in ACC
        
        if any(
            np.isnan(sig[ephy_i0:ephy_i0 + nperseg])
        ): continue  # skip window if nans present

        # add windows without movement, without nan's
        blWin_tempLists.append(sig[ephy_i0:ephy_i0 + nperseg])
        blWin_times.append(times[ephy_i0])
    
    blDatWins = np.array(blWin_tempLists)  # array of bl-data windows
    blDatTimes = np.array(blWin_times)  # array of bl-data times

    assert blDatWins.shape[0] == len(blDatTimes), print(
        'Baseline # data-windows and # times are not equal'
    )

    print(
        f'\tBaseLine for {dType}, {ephysCh}: {len(blWin_times)} '
        f'windows of {nperseg} samples ({ephysFs} Hz)')

    return blDatWins, blDatTimes, ephysFs, overlapPerc



@dataclass(init=True, repr=True,)
class getFeatures_singleChannel:
    """
    Calculate the spectral feature values of
    selected windows for one given channel.

    Input:
        - blData (ndarray): containing windows  
            of pre-selected data
        - blTimes (ndarray): dopa-timed timestamps
            corresponding to window starts
        - fs (int): Fs
        - overlap: % of window-length used as overlap,
            between 0 and 1, or None

    Class output:
        - psd (dict): f and psd from Welch PSD, nperseg
            default as 0.5 seconds; psd as mean over time
        - wav (dict): wavelet containing time, freq, and psd
        - wavlog (dict): wavelet containing time, freq,
            and psd (log-transformed)
    """
    blData: Any
    blTimes: Any
    fs: int
    overlap: Any
    incl_psd: bool
    incl_wav: bool
            
    def __post_init__(self,):

        if self.incl_psd:
            # calculate psd with welch
            self.psd_f, self.psd = welch(
                self.blData,
                fs=self.fs,
                nperseg=self.fs // 2,
            )

        if self.incl_wav:
            ## TODO: DEBUG Wavelet decomposition code

            # calculate wavelet
            w = 8
            base_f = np.linspace(1, self.fs / 2, 100)
            widths = (self.fs * w) / (2 * base_f * np.pi)

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


@dataclass(init=True, repr=True,)
class getFeatures_BGCortex:
    """
    Calculate the spectral feature values of
    selected windows for one given channel.

    Input:
        - ecogCh (str): name of ECoG-channel-name
        - ecogData (ndarray): containing windows  
            of pre-selected data from ECoG-ch
        - ecogTimes (ndarray): dopa-timed timestamps
            corresponding to window starts
        - stnCh (str): name of ECoG-channel-name
        - stnData (ndarray): containing windows  
            of pre-selected data from ECoG-ch
        - stnTimes (ndarray): dopa-timed timestamps
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
    ecogCh: str
    ecogDat: Any
    ecogTimes: Any
    stnCh: str
    stnDat: Any
    stnTimes: Any
    fs: int
    overlap: Any
    
    def __post_init__(self,):

        # ecogPs_list, stnPs_list, crossPs_list = [], [], []
        icoh_list, coh_list, coh_times = [], [], []

        # loop over ecog-windows, containing n seconds
        # as defined in find_basel..() (default: 5 sec)
        for w, ecog_t in enumerate(self.ecogTimes):

            # skip if window is not present in both data sources
            if np.around(ecog_t, 5) not in np.around(
                self.stnTimes, 5
            ): continue

            coh_times.append(np.around(ecog_t, 5))

            stn_w = np.where(
                np.around(self.stnTimes, 5) == np.around(ecog_t, 5)
            )  # find stn window matching with ecog time
            stn_w = stn_w[0]

            freqs, icoh, coh = spec_feats.calc_coherence(
                stn_sig=self.stnDat[stn_w, :],
                ecog_sig=self.ecogDat[w, :],
                fs=self.fs,
                nperseg=self.fs // 2,
            )

            icoh_list.append(icoh)
            coh_list.append(coh)
        
        # store freq list
        self.freqs = freqs

        # convert to arrays
        self.icoh = np.array(icoh_list)
        self.coh = np.array(coh_list)
        self.coh_times = coh_times

        # calculate standardized detectable coherencies (Nolte ea 2008)
        for c in ['coh', 'icoh']:

            dtc_lists = []

            for w in np.arange(getattr(self, c).shape[0]):

                values = getattr(self, c)[w, :]  # select window
                rest = np.delete(getattr(self, c), w, axis=0)  # select other windows for std
                dtc_lists.append(abs(values / np.std(rest)))  # standardize coh with std

            # store (i)-coh detectable (standardised) for all windows
            setattr(self, f'{c}_dtc', np.array(dtc_lists))
            
            # store std-dev to standardise all other windows
            # std-dev over whole 'baseline-period' is taken
            setattr(self, f'{c}_stddev', np.std(getattr(self, c)))



        



        

        



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

