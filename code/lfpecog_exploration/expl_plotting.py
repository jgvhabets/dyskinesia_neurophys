'''Data Exploration Plotting Functions'''

# import packages and modules
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

# import own functions
import lfpecog_preproc.preproc_filters as fltrs





def meanPSDs_session_channels(
    data, names, norm: str, nseg: int=256,
    RunInfo=None, save=False, plot=True
):
    """
    Function plots overview of mean PSDs (beta-gamma) per session
    over the different rereferenced channels in every group.
    PSDs are normalized per window, and averaged over all win's.
    Can be used for channel exploration/selection.

    timeit: periodogram 33% faster than welch (both scipy)
    normalisation, welch seem to better show subtle activity

    Arguments:
        - data (dict): 3d-array of each group
        - names (dict): corresponding ch-names (incl time)
        - norm (str): normalisation method (norm / z-score)
        - nseg (int): number of samples per welch/fft-window
        - RunInfo (dataclass): containing info of run
        - save: directory to save fig, or False
        - plot (Bool): if False, figure is not plotted
    
    Returns:
        - plots and saves figure
    """
    if ~ np.logical_or(norm == 'norm', norm == 'z-score'):
        print('Normalisation method should be z-score or norm')
        return

    methods = ['fft', 'welch']
    method = 'welch'
    
    fig, axes = plt.subplots(len(data), 2,
        sharex=False, sharey='row', figsize=(12, 16)
    )
    ls = 14  # labelsize
    ts = 16  # titlesize

    for s, src in enumerate(data):
        print(f'START {src} with {method}')

        psx = np.empty((
            data[src].shape[0],
            data[src].shape[1] - 1,
            (nseg // 2) + 1),
        )  # if nseg=256: 129 freq's, etc
        Zpsx = np.empty_like(psx)

        # data[src] = 

        for w in np.arange(data[src].shape[0]):
            if method == 'fft':
                f, ps = periodogram(
                    data[src][w, 1:, :], fs=800, nfft=nseg
                )
            elif method == 'welch':
                f, ps = welch(
                    data[src][w, 1:, :], fs=800, nperseg=nseg
                )
            psx[w, :, :] = ps

            # normalize psd's per channel, per window
            for r in np.arange(psx.shape[1]):
                if norm == 'z-score':
                    m = np.nanmean(psx[w, r, :])
                    sd = np.nanstd(psx[w, r, :])
                    Zpsx[w, r, :] = (psx[w, r, :] - m) / sd
                elif norm == 'norm':
                    Zpsx[w, r, :] = psx[w, r, :] / np.max(psx[w, r, :])
        # create mean matrices over windows [channels x freqs]
        # ch_ps = np.nanmean(psx, axis=0)  # only visualizing norm PSDs
        ch_zps = np.nanmean(Zpsx, axis=0)
        band = ['Beta', 'Gamma']

        for n, (xlow, xhigh) in enumerate(zip(
            [0, 60], [30, 90]
        )):
            ihigh = [i for i in range(len(f)) if f[i] >= xhigh][0]
            ilow = [i for i in range(len(f)) if f[i] >= xlow][0]
            # Plot
            im = axes[s, n].pcolormesh(ch_zps, cmap='viridis',)
            if n == 1: fig.colorbar(im, ax=axes[s, n])
            axes[s, n].set_yticks(np.arange(ch_zps.shape[0]) + .5)
            axes[s, n].set_yticklabels(names[src][1:], size=ls)
            # if n == 0: axes[s, n].set_ylabel('Channels')
            if s == 2: axes[s, n].set_xlabel('Frequency (Hz)', size=ls)
            axes[s, n].set_title(f'{band[n]} PSDs for {src}', size=ts)
            axes[s, n].set_xlim(ilow, ihigh)  # show beta
            xticks = np.linspace(ilow, ihigh, 5).astype(int)
            axes[s, n].set_xticks(xticks)
            axes[s, n].set_xticklabels(f[xticks].astype(int), size=ls)
    plt.suptitle(f'{RunInfo.store_str}, {RunInfo.preproc_sett}',
                 color='gray', alpha=.5, x=.25, y=.99, size=ts)
    plt.tight_layout(h_pad=.2, w_pad=.05)

    if save:
        fname = f'Beta_Gamma_ContactReview_({method}_{nseg})'
        plt.savefig(
            os.path.join(save, fname + '.jpg'),
            dpi=150, facecolor='white',
        )
    if plot:
        plt.show()
    else:
        plt.close()







