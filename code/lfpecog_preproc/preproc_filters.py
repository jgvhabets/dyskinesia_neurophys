# Import packages and functions
import os
import numpy as np
import mne
import matplotlib.pyplot as plt


def bp_filter(
    clean_dict, group, sfreq, l_freq, h_freq,
    method='fir', fir_window='hamming', verbose=False
):
    '''
    DOCT STRING TO WRITE
    '''
    data_out = clean_dict[group].copy()
    for w in np.arange(clean_dict[group].shape[0]):
        data_out[w, 1:, :] = mne.filter.filter_data(
            data=clean_dict[group][w, 1:, :],
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            fir_window=fir_window,
            verbose=verbose,
        )

    return data_out


def notch_filter(
    bp_dict: dict,
    group: str,
    transBW: int,  # Default in doc is 1
    notchW: int,  # Deafult in doc is f/200
    Fs: int = 4000,  # sample freq, default 4000 Hz
    freqs: list = [50, 100, 150, 200, 250, 350, 400],  # power line freqs EU
    method='fir',
    ch_names: dict = None,  # clean incl channelnames per group
    save=None,
    verbose='Warning'
):
    '''
    Applies notch-filter to filter local peaks due to powerline
    noise. Uses mne-fucntion (see doc).
    Inputs:
    - data (array): 2D data array with channels to filter,
    - transBW (int): transition bandwidth, try out and decide on
        defaults for LFP and ECOG seperately,
    - notchW (int): notch width, try out and decide on
        defaults for LFP and ECOG seperately,
    - save (str): if pre and post-filter figures should be saved,
        directory should be given here,
    - verbose (str): amount of documentation printed.
    Output:
    - data: filtered data array.
    '''
    data = bp_dict[group]
    data_out = data.copy()
    ch_names = ch_names[group]  # select list for corresponding group
    if save:
        # select range of win's to plot
        plot_wins = np.arange(int(data.shape[0] / 2), int(data.shape[0] / 2) + 20)
        # visualize before notch filter
        fig, axes = plt.subplots(data.shape[1] - 1, 2, figsize=(12, 12),
            sharex=True, sharey=True,)
        for C in np.arange(1, data.shape[1]):
            # plotting x example windows (plot_wins)
            for w in plot_wins:
                axes[C-1, 0].psd(data[w, C, :], Fs=4000,
                               NFFT=1024, label=str(w))
            # plot details per channel row
            for f in freqs:
                    axes[C-1, 0].axvline(f, color='red', alpha=.3,
                                         lw=3, ls='dotted')
            axes[C-1, 0].set_xlim(0, 160)
            axes[C-1, 0].set_ylabel(ch_names[C][:8])
            axes[C-1, 0].set_xlabel('')
        axes[0, 0].set_title('PSD (dB/Hz) BEFORE Notch Filter')

    # apply notch filter
    for w in np.arange(data.shape[0]):
        data_out[w, 1:, :] = mne.filter.notch_filter(
            x=data[w, 1:, :],
            Fs=Fs,
            freqs=freqs,
            trans_bandwidth=transBW,
            notch_widths=notchW,
            method='fir',
            fir_window='hamming',
            fir_design='firwin',
            # filter_length=int(Fs / 2),
            verbose=verbose,
        )

    if save:
        # visualize after notch filter
        for C in np.arange(1, data_out.shape[1]):
            for w in plot_wins:  # plot example windows
                axes[C-1, 1].psd(data_out[w, C, :], Fs=4000,
                               NFFT=1024, label=str(w))
            for f in freqs:
                axes[C-1, 1].axvline(f, color='red', alpha=.3,
                                     lw=3, ls='dotted')
            axes[C-1, 1].set_xlim(0, 160)
            axes[C-1, 1].set_ylabel(ch_names[C][:8])
            axes[C-1, 1].set_xlabel('')
        axes[0, 1].set_title('PSD (dB/Hz) AFTER Notch Filter')

        lastrow = data_out.shape[1] - 2  # minus time, start 0
        axes[lastrow, 0].set_xlabel('Frequency (Hz)')
        axes[lastrow, 1].set_xlabel('Frequency (Hz)')

        plt.tight_layout(w_pad=.05, h_pad=0.01)

        fname = f'{group}_Notch_{method}_transBW{transBW}_notchW{notchW}.jpg'
        plt.savefig(os.path.join(save, fname), dpi=150,
                    faceccolor='white')
        plt.close()
    
    return data
