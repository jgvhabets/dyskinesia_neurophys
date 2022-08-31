"""
Contains Classes to help extract features
from either single channels or combined
ECoG-LFO channels
"""

# Import public functions and packages
import numpy as np
from dataclasses import dataclass
from typing import Any
from scipy.signal import welch, cwt, morlet2

# Import own functions
import lfpecog_features.feats_spectral_features as spec_feats


@dataclass(init=True, repr=True,)
class getFeatures_singleChannel:
    """
    Calculate the spectral feature values of
    selected windows for one given channel.

    Input:
        - winData (ndarray): containing windows  
            of pre-selected data
        - winTimes (ndarray): dopa-timed timestamps
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
    winData: Any
    winTimes: Any
    fs: int
    nperseg: int
    overlap: Any
    incl_psd: bool = True
    incl_wav: bool = False
            
    def __post_init__(self,):

        if self.incl_psd:
            # calculate psd with welch
            self.psd_f, self.psd = welch(
                self.winData,
                fs=self.fs,
                nperseg=self.nperseg,
            )

        if self.incl_wav:
            ## TODO: DEBUG Wavelet decomposition code and move to own .py file

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
                wavSig = self.winData[::2, :]  # take every other window
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
    nperseg: int
    overlap: Any
    extr_baseline: bool = True
    combiCh_baseline: Any = None
    
    def __post_init__(self,):

        # ecogPs_list, stnPs_list, crossPs_list = [], [], []
        icoh_list, coh_list, coh_times = [], [], []

        # loop over ecog-windows, containing n seconds
        # as defined in find_basel..() (default: 5 sec)
        for w, ecog_t in enumerate(self.ecogTimes):

            # check presence of window-time in both data sources
            if np.around(ecog_t, 5) not in np.around(
                self.stnTimes, 5
            ): continue

            coh_times.append(np.around(ecog_t, 5))

            stn_w = np.where(
                np.around(self.stnTimes, 5) == np.around(ecog_t, 5)
            )[0]  # find stn window matching with ecog time

            if w < 5:
                print(f'Coherence time check:\n\tECoG: {ecog_t} vs STN: {self.stnTimes[stn_w]}')

            freqs, icoh, coh = spec_feats.calc_coherence(
                stn_sig=self.stnDat[stn_w, :],
                ecog_sig=self.ecogDat[w, :],
                fs=self.fs,
                nperseg=self.nperseg,
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
        
            if self.extr_baseline:

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

            elif self.combiCh_baseline != None:  # if not extracting for baseline

                coh_std = getattr(self.combiCh_baseline, f'{c}_stddev')

                setattr(
                    self,
                    f'{c}_dtc',
                    abs(getattr(self, c) / coh_std)
                )


