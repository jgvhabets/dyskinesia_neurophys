'''
Connectivity-Feature Extraction
based on two ephys-channels
'''

# Import public packages and functions
import numpy as np
import os
from scipy.signal import welch
from array import array
from dataclasses import dataclass

# import own functions
from utils import utils_windowing as utils_win
import lfpecog_features.feats_helper_funcs as ftHelpers

@dataclass(init=True, repr=True, )
class segmentConnectFts:
    """
    Calculate features per segment
    """
    sub: str
    data_arr: array
    data_keys: list
    winTimes: list
    fs: int
    seed_ephyGroup: str
    target_ephyGroup: str
    segLen_sec: float = .5
    part_overlap: float = 0.
    
    def __post_init__(self,):
        # check input
        assert self.ephyGroup in [
            'LFP_R', 'LFP_L', 'ECOG'
        ], print('wrong input for ephyGroup')

        # define segment and window lengths in samples
        segLen_n = int(self.segLen_sec * self.fs)  # n-samples per segment
        nOverlap = self.part_overlap * self.fs

        # select col-names of ephys-group
        self.ephyCols = [
            k for k in self.data_keys if self.ephyGroup in k
        ]

        # find col-index in array
        icol = np.where(self.data_keys == 'dopa_time')[0][0]
        times_arr = self.data_arr[:, :, icol]

        for col in self.ephyCols:  # loop over selected columns/ contacts
            # find col-index in array
            icol = np.where(self.data_keys == col)[0][0]
            ch_arr = self.data_arr[:, :, icol]
            # set time variables
            # win_times = self.winTimes.copy()
            # orig_n_times = len(temp_times)  # OLD

            del_wins = []

            # LOOP OVER ALL WINDOWS
            for i_win in np.arange(ch_arr.shape[0]):
                # DATA SELECTION, NaN REMOVING
                windat = ch_arr[i_win, :]  # select column-data
                wintimes = times_arr[i_win, :]

                # TODO: SAVE no-non-segment-array per window??
                segm_dat, segm_times = utils_win.get_noNanSegm_from_window(
                    windat=windat,
                    segLen_n=segLen_n,
                    win_times=wintimes,
                    n_overlap=200
                )
                
                if len(segm_times) == 0:
                    
                    del_wins.append(i_win)
                    continue

                # TODO: correct variables names fornew functionn
                # ### OLD
                # rev_win_i = orig_n_times - i_win  # take reverse-index for later deletion
                
                # # remove NaNs if present
                # if np.isnan(list(windat)).any():
                #     nansel = ~np.isnan(list(windat))
                #     windat = windat[nansel]  # take only non nan-data
                #     wintimes = wintimes[nansel]  # take times for only non nan-data
                    
                #     if len(windat) < segLen_n:
                #         # skip if window w/o nan's not long enough
                #         del_wins.append(i_win)
                #         continue
                    
                #     nseg = int(len(windat) / segLen_n)  # n of segments in window
                #     # reshape data to 2d to get segment-psds
                #     windat = windat[:int(nseg * segLen_n)].reshape(
                #         nseg, int(segLen_n)
                #     )  # discard end of window not fitting in full-segment anymore
                
                # else:  # if no nan's present

                #     windat = windat.reshape(
                #         int(len(windat) / segLen_n), int(segLen_n)
                #     )  # reshape data to 2d to get segment-psds

                ### TODO: differentiate between different FEATURES-TYPES (bursts, PAC, coherence)
                
                # FEATURE EXTRACTION

                # get power spectra per segment (axis=1)
                f, ps = welch(
                    segm_dat, fs=self.fs, nperseg=segLen_n,
                    noverlap=nOverlap, axis=1,
                )
                ps = abs(ps).astype(float)  # take real-value, discard imaginary part
                
                # get corresponding times  OLD
                # psd_times = wintimes[
                #     :int(windat.shape[0] * windat.shape[1])  # times until n samples included
                #     :int(segLen_n - nOverlap)  # hop over segment start-times
                # ].astype(float)
                
                # if len(ps) == 0:  # if psd is empty, do not add
                #     del_wins.append(i_win)  # remove window time later
                #     continue
                
                assert ps.shape[0] == len(segm_times), print(
                    'PSD-# and times-# are UNEQUAL '
                )

                try:  # later windows are added to existing array
                    index_win_start.append(len(psd_out_array))  # tracks window-start-indices in 2d segm arr
                    # currently creating 2d-array; TODO: consider 3d-array especially with window-overlap
                    psd_out_array = np.concatenate([psd_out_array, ps])
                    times_out_array = np.concatenate(
                        [times_out_array, segm_times]
                    )
                    # allTimes.extend(list(ftHelpers.spaced_arange(
                    #     start=temp_times[-rev_win_i],
                    #     step=self.segLen_sec,
                    #     num=len(ps)
                    # )))


                except NameError:  # array created with first window
                    index_win_start = [0]
                    psd_out_array = ps
                    times_out_array = segm_times
                    # allTimes = list(ftHelpers.spaced_arange(
                    #     start=temp_times[-rev_win_i],
                    #     step=self.segLen_sec,
                    #     num=len(ps)
                    # ))

            # delete times of skipped windows due to missing data
            for i_w in sorted(del_wins, reverse=True):
                del(self.winTimes[i_w])
            
            # store features and times per contact
            setattr(
                self,
                col,
                segmentFts_perContact(
                    segmPsds=psd_out_array,
                    psdFreqs=f,
                    segmTimes=times_out_array,
                    winStartTimes=self.winTimes,
                    winStartIndices=index_win_start,
                    # segmentTimes=None,  # OLD
                    contactName=col,
                    sub=self.sub,
                )
            )
            
            # reset values for next contact-iteration
            del(
                psd_out_array,
                index_win_start,
                # allTimes,
                times_out_array
            )
        


@dataclass(init=True, repr=True, )
class segmentFts_perContact:
    """
    Store segment-features and timings per column/contact
    """
    segmPsds: array
    psdFreqs: array
    segmTimes: array
    winStartTimes: list
    winStartIndices: list
    # segmentTimes: list
    contactName: str
    sub: str