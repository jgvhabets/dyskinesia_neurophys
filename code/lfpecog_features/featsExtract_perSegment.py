'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
import os
from scipy.signal import welch
from array import array
from dataclasses import dataclass

# import own functions
import utils.utils_windowing as utils_win
# from utils import utils_windowing
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


                ### TODO: differentiate between different FEATURES-TYPES (bursts, PAC, coherence)
                
                # FEATURE EXTRACTION

                # get power spectra per segment (axis=1)
                f, ps = welch(
                    segm_dat, fs=self.fs, nperseg=segLen_n,
                    noverlap=nOverlap, axis=1,
                )
                ps = abs(ps).astype(float)  # take real-value, discard imaginary part
                
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

                except NameError:  # array created with first window
                    index_win_start = [0]
                    psd_out_array = ps
                    times_out_array = segm_times
                    

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
                    channelName=col,
                    sub=self.sub,
                )
            )
            
            # reset values for next contact-iteration
            del(
                psd_out_array,
                index_win_start,
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
    channelName: str
    sub: str