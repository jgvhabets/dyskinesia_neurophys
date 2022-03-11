'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import os


def data_2d_to_3d(
    runclass,
    win_time: float,
    source_incl: list = None,
    source_excl: list = None,
):
    '''
    Arguments:
        - dclass: dataclass containing runs
        - run to use
        - win_time: window length in seconds
    Output:
        - arr_3d: 3d array with windowed data
    '''
    # iterate over present data sources
    for src in runclass.present_datatypes:
        src = src.lower()
        if source_excl: if src in source_excl: continue
        if source_incl: if src not in source_incl: continue
      
        data2d = getattr(runclass, f'{src}_arr')

        assert len(data2d.shape) == 2, (
            f'Data {runclass.sub} {runclass.ses} '
            f'{runclass.acq} {src} is not 2D!')
        
        fs = getattr(runclass, f'{src}_Fs')
        win_samples = int(win_time / (1 / fs))
        n_wins = data2d.shape[1] // win_samples
    

    
    return arr_3d

# if source[:3] == 'acc':  # create empty 3d-array
#         arr_3d = np.empty(  # add row for sign vector mgn
#             (n_wins, data2d.shape[0] + 1, win_samples))
#     else:
#             arr_3d = np.empty(  # create empty 3d-array
#         (n_wins, data2d.shape[0], win_samples))

#     for nwin in range(arr_3d.shape[0]):
#         # fill every window with corresp 2d-data
#         arr_3d[nwin, :, :] = data2d[
#             :,
#             nwin * win_samples:(nwin + 1) * win_samples]
#         if source[:3] == 'acc':
#             arr_3d[nwin, -1, :] = np.sqrt(
#                 arr_3d[nwin, 1, :]**2 +
#                 arr_3d[nwin, 2, :]**2 +
#                 arr_3d[nwin, 3, :]**2
#             )
    
#     if art_removal:
#         if source[:3] == 'acc':
#             arr_3d = find_ACC_artefact(arr_3d, tresh_SD=8)

# def find_ACC_artefact(acc, tresh_SD: float = 8):
#     '''
#     '''
#     assert acc.shape[-2] > 4, (
#         '\n### ERROR: Acc has no SVM\n')
    
#     if len(acc.shape) == 3:
#         # calc signal vec magn means per window
#         win_svmeans =  np.mean(acc[:, 4, :], axis=1)
#         for w in range(acc.shape[0]):
#             if win_svmeans[w] > (
#                 np.mean(win_svmeans) + (
#                 tresh_SD * np.std(win_svmeans))
#             ):
#                 # check print to check
#                 print(f'{w} detected as outlier')
#                 plt.plot(acc[w, 1:, :].T,
#                     label=['x', 'y', 'z', 'SVM'])
#                 plt.title(w)
#                 plt.legend()
#                 plt.show()
#                 # set nan's in case of artefact
#                 acc[w, 1:, :] = [
#                     [np.nan] * acc.shape[2]] * 4


#     elif len(data.shape) == 2:
#         ...
    

#     return acc