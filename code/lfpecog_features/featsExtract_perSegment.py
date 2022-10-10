'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
import os
from scipy.signal import welch
from array import array
from dataclasses import dataclass

# import own functions
import lfpecog_features.feats_helper_funcs as ftHelpers

@dataclass(init=True, repr=True, )
class segmentFeatures:
    """
    Calculate features per segment
    """
    sub: str
    data_arr: array
    data_keys: list
    winTimes: list
    fs: int
    ephyGroup: str
    segLen_sec: float = .5
    part_overlap: float = 0.
    
    def __post_init__(self,):
        # check input
        assert self.ephyGroup in [
            'LFP_R', 'LFP_L', 'ECOG'
        ], print('wrong input for ephyGroup')

        # define segment and window lengths in samples
        segLen_n = self.segLen_sec * self.fs  # n-samples per segment
        nOverlap = self.part_overlap * self.fs
        # nHop = (1 - self.part_overlap) * segLen_n

        # select col-names of ephys-group
        self.ephyCols = [
            k for k in self.data_keys if self.ephyGroup in k
        ]

        # segFts, segStartTimes = {}, {}  # dict's to store feats and times
        # segStartTicks, segAllTimes = {}, {}

        for col in self.ephyCols:  # loop over selected columns/ contacts
            # find col-index in array
            icol = np.where(self.data_keys == col)[0][0]

            ch_arr = self.data_arr[:, :, icol]
            temp_times = self.winTimes.copy()
            orig_n_times = len(temp_times)

            del_wins = []

            for i_win in np.arange(ch_arr.shape[0]):

                windat = ch_arr[i_win, :]
                rev_win_i = orig_n_times - i_win
                
                # remove NaNs if present
                if np.isnan(list(windat)).any():

                    windat = windat[~np.isnan(list(windat))]  # take only non nans
                    
                    if len(windat) < segLen_n:
                        # skip if window w/o nan's not long enough
                        del_wins.append(i_win)
                        continue
                    
                    nseg = int(len(windat) / segLen_n)  # n of segments in window
                    # reshape data to 2d to get segment-psds
                    windat = windat[:int(nseg * segLen_n)].reshape(
                        nseg, int(segLen_n)
                    )  # discard end of window not fitting in full-segment anymore
                
                else:  # if no nan's present

                    windat = windat.reshape(
                        int(len(windat) / segLen_n), int(segLen_n)
                    )  # reshape data to 2d to get segment-psds

                ### TODO: differentiate between different FEATURES-TYPES (bursts, PAC, coherence)

                # get power spectra per segment (axis=1)
                f, ps = welch(
                    windat, fs=self.fs, nperseg=segLen_n,
                    noverlap=nOverlap, axis=1,
                )
                ps = abs(ps).astype(float)  # take real-value, discard imaginary part
                
                if len(ps) == 0:  # if psd is empty, do not add
                    del_wins.append(i_win)  # remove window time later
                    continue

                try:  # later windows are added to existing array
                    nSamplesAtTime.append(len(winPsds))
                    winPsds = np.concatenate([winPsds, ps])
                    allTimes.extend(list(ftHelpers.spaced_arange(
                        start=temp_times[-rev_win_i],
                        step=self.segLen_sec,
                        num=len(ps)
                    )))

                except NameError:  # array created with first window
                    winPsds = ps
                    nSamplesAtTime = [0]
                    allTimes = list(ftHelpers.spaced_arange(
                        start=temp_times[-rev_win_i],
                        step=self.segLen_sec,
                        num=len(ps)
                    ))

            # delete times of skipped windows due to missing data
            for i_w in sorted(del_wins, reverse=True):
                del(temp_times[i_w])
            
            # store features and times per contact
            setattr(
                self,
                col,
                segmentFts_perContact(
                    segmPsds=winPsds,
                    psdFreqs=f,
                    winStartTimes=temp_times,
                    winStartIndices=nSamplesAtTime,
                    segmentTimes=allTimes,
                    contactName=col,
                )
            )
            # segFts[col] = winPsds
            # segStartTimes[col] = temp_times
            # segStartTicks[col] = nSamplesAtTime
            # segAllTimes[col] = allTimes

            # reset values for next contact
            del(winPsds, nSamplesAtTime, allTimes)
        
        # return segFts, segStartTimes, segStartTicks, segAllTimes


@dataclass(init=True, repr=True, )
class segmentFts_perContact:
    """
    Store segment-features and timings per column/contact
    """
    segmPsds: array
    psdFreqs: array
    winStartTimes: list
    winStartIndices: list
    segmentTimes: list
    contactName: str