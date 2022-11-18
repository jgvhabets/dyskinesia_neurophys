'''Data Exploration Plotting Functions'''

# import packages and modules
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch

# import own functions
from utils.utils_fileManagement import get_project_path


def plot_act_per_window(
    classData, winlength,
):
    savepath = os.path.join(
        get_project_path('figures'),
        'ft_exploration', 'rest',
    )
    if not os.path.exists(savepath): os.makedirs(savepath)

    plt.figure(figsize=(12, 6))

    for sub in classData.list_of_subs:
        tempdata = getattr(classData.rest, f'sub{sub}')

        if tempdata.index.name != 'dopa_time':
            tempdata.set_index('dopa_time')

        sec_first = np.round(tempdata.index[0], 0)
        sec_last = np.round(tempdata.index[-1], 0)

        act_list, sec_list = [], []

        for sec0 in np.arange(sec_first, sec_last, winlength):

            act = tempdata.loc[sec0:sec0 + winlength]['no_move']
            if len(act) > 0:
                act_list.append(sum(act) / len(act) * 100)
                sec_list.append(sec0 / 60)
            else:
                act_list.append(np.nan)
                sec_list.append(sec0 / 60)
                
        plt.plot(sec_list, act_list, label=sub)
    
    plt.title(f'Activity-% per {winlength} sec windows')
    plt.ylabel('% WITHOUT accelerometer-activity')
    plt.xlabel('Time after L-Dopa intake (min)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    fname = f'act_perc_per_sub_{winlength}s_windows'
    plt.savefig(
        os.path.join(savepath, fname),
        dpi=150, facecolor='w',)
    plt.close()


def blank_empty_axes(axes):
    '''
    Function to make empty axes in a 2d figure
    blank. For example, subplot, 4 rows, 2 cols.
    In second col, last row is empty.
    -> Function deletes x/y-axis for that ax.
    '''
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if not axes[row, col].lines:
                axes[row, col].axis('off')
            # scatterplots would need if not .collections


def electrode_spectral_check(
    sessionClass,
    savedir: str,
    fname_add: str = '',
    sides: list = ['lfp_left', 'lfp_right', 'ecog'],
    tasks_incl: list = ['Rest',],
    method: str = 'Welch',
    nseg = 512,
):
    '''
    
    '''
    # Data Selection
    exmpl_run = sessionClass.runs_incl[0]
    chs = {}
    for side in sides: 
        chs[side] = getattr(
            sessionClass.runs[exmpl_run], f'{side}_names'
        )[1:]
    fig_len = np.max([
        len(chs[side]) for side in sides
    ])

    fig, axes = plt.subplots(fig_len, len(sides),
                            figsize=(12, 16))
    cols = {  # TODO: AUTOMIZE
        'Dopa00': 'r',
        'Dopa10': 'orange',
        'Dopa15': 'orange',
        'Dopa30': 'olive',
        'Dopa35': 'olive',
        'Dopa50': 'g',
        'Dopa60': 'g',
    }
    lstyles = {  # indicate acq-activity with first 5 letters
        'Rest': 'solid',
        'Selfp': 'dotted',
    }
 
    ### TODO: add movement run's as dotted lines
    for col, side in enumerate(sides):
        for row, ch in enumerate(range(axes.shape[0])):
            ch = row + 1

            for run in sessionClass.runs_incl:
                if ch >= len(getattr(
                        sessionClass.runs[run], f'{side}_names')):
                    continue  # prevent out of range

                # loop over different runs/dopa states
                sub = sessionClass.runs[run].sub
                ses = sessionClass.runs[run].ses[5:]
                task = sessionClass.runs[run].task[:5]
                chname = getattr(
                    sessionClass.runs[run], f'{side}_names')[ch]
                if task in tasks_incl:
                    # Get and prepare data
                    Dmin = run[-6:]
                    d = getattr(sessionClass.runs[run], f'{side}_arr')
                    d = d[ch, :]  # insert a end index to decrease incl time
                    windows = len(d) // nseg
                    d = d[:windows * nseg]
                    d = np.reshape(d, newshape=(windows, nseg))
                    # calculate FFT's, take 10*log10 and mean over run
                    if method == 'FFT': 
                        f, ps = periodogram(d, fs=800, nfft=nseg,
                            axis=1, window='hanning')
                    elif method == 'Welch':
                        f, ps = periodogram(d, fs=800, nfft=nseg,
                            axis=1, window='hanning')
                    logps = 10 * np.log10(ps)
                    meanps = np.nanmean(logps, axis=0)

                    # Plotting
                    axes[row, col].plot(
                        f, meanps, c=cols[Dmin], alpha=.7,
                        label=f'{Dmin} {task}',
                        ls=lstyles[task],
                    )
                    axes[row, col].set_title(
                        f'{chname} (sub {sub}, ses {ses})'
                    )
                    axes[row, col].set_xlim(0, 120)
                    axes[row, col].set_ylabel(
                        'Spectral Power (10 * log10(power (FFT))'
                    )
                    axes[row, col].set_xlabel('Freq (Hz)')
                    axes[row, col].legend(ncol=2, fontsize=10)
    # for row in range(axes.shape[0]):
    #     for col in range(axes.shape[1]):
            
    plt.suptitle(sessionClass.runs[run].run_string,
                 color='gray', alpha=.5, x=.4, y=.99, size=16)
    # tidy figure up
    blank_empty_axes(axes)  # remove empty ax
    plt.tight_layout()
    # save figure
    fname = f'Electrodes_Spect_Pow_{sub}_{ses}'
    if fname_add: fname += fname_add
    plt.savefig(os.path.join(savedir, fname), dpi=150, facecolor='white',)
    plt.close()

    return 'Figure saved!'



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







