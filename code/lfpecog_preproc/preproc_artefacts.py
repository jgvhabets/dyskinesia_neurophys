# Import packages and functions
import os
import numpy as np
import matplotlib.pyplot as plt


def artefact_selection(
    bids_dict: dict,
    group: str,
    win_len: float=.5,
    overlap=None,
    n_stds_cut: float=2.5,
    save=None
):
    '''
    Function to remove artefacts and visualize the resulting
    selection.
    Blocks-values are converted to NaNs when an outlier (value
    exceeds thresholds of n_std_cut times std dev of full recording).
    Also ocnnverted to NaN's if more than 25% of block is 0.
    
    Arguments:
        - bids_dict, Raw BIDS selection: grouped BIDS raw, e.g. rawRun1.ecog,
        - win_len (float): block window length in seconds,
        - overlap (float): time of overlap between consec blocks (seconds),
        - n_stds_cut, int: number of std-dev's above and below mean that
            is used for cut-off's in artefact detection,
        - save (str): 1) directory where to store figure, 2) 'show' to only
            plot in notebook, 3) None to not plot.

    Returns:
        - sel_bids (array): array with all channels in which artefacts
        are replaced by np.nan's.
    '''
    print(f'START ARTEFACT REMOVAL: {group}')
    data = bids_dict[group]
    ch_nms = data.ch_names
    fs = data.info['sfreq']  # ONLY FOR BLOCKS
    (ch_arr, ch_t) = data.get_data(return_times=True)
    # visual check by plotting before selection
    if save:
        fig, axes = plt.subplots(len(ch_arr), 2, figsize=(16, 16))
        for n, c in enumerate(np.arange(len(ch_arr))):
            axes[c, 0].plot(ch_t, ch_arr[c, :])
            axes[c, 0].set_ylabel(ch_nms[n], rotation=90)
        axes[0, 0].set_title('Raw signal BEFORE artefact deletion')

    # Artefact removal part
    win_n = int(win_len * fs)  # number of samples to fit in one window
    n_wins = int(ch_arr.shape[1] / win_n)  # num of windows to split in
    # new array to store data without artefact, ch + 1 is for time
    new_arr = np.zeros((n_wins, len(ch_nms) + 1, win_n), dtype=float)
    n_nan = {}  # number of blocks corrected to nan
    # first reorganize data
    for w in np.arange(new_arr.shape[0]):  # loop over new window's
        # first row of windows is time
        new_arr[w, 0, :] = ch_t[w * win_n:w * win_n + win_n]
        # other rows filled with channels
        new_arr[w, 1:, :] = ch_arr[:, w * win_n:w * win_n + win_n]
    # correct to nan for artefacts per channel
    cuts = {}  # to store thresholds per channel
    for c in np.arange(ch_arr.shape[0]):  # loop over ch-rows
        # cut-off's are X std dev above and below channel-mean
        cuts[c] = (np.mean(ch_arr[c]) - (n_stds_cut * np.std(ch_arr[c])),
                np.mean(ch_arr[c]) + (n_stds_cut * np.std(ch_arr[c])))
        n_nan[c] = 0
        for w in np.arange(new_arr.shape[0]):  # loop over windows
            if (new_arr[w, c + 1, :] < cuts[c][0]).any():
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1
            elif (new_arr[w, c + 1, :] > cuts[c][1]).any():
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1
            elif (new_arr[w, c + 1, :] == 0).sum() > (.25 * win_n):
                # more than 25% exactly 0
                new_arr[w, c + 1, :] = [np.nan] * win_n
                n_nan[c] = n_nan[c] + 1

    # visual check by plotting after selection
    if save:
        for c in np.arange(len(ch_arr)):
            plot_ch = []
            for w in np.arange(new_arr.shape[0]):
                plot_ch.extend(new_arr[w, c + 1, :])  # c + 1 to skip time
            plot_t = ch_t[:len(plot_ch)]
            axes[c, 1].plot(plot_t, plot_ch, color='blue')
            # color shade isnan parts
            ynan = np.isnan(plot_ch)
            ynan = ynan * 2
            axes[c, 1].fill_between(
                x=plot_t,
                y1=cuts[c][0],
                y2=cuts[c][1],
                color='red',
                alpha=.3,
                where=ynan > 1,
            )
            axes[c, 1].set_title(f'{n_nan[c]} windows deleted')
        fig.suptitle('Raw signal artefact deletion (blocks: {win_len} sec, cut'
                f' off: {n_stds_cut} std dev +/- channel mean', size=14)
        fig.tight_layout(h_pad=.2)

        if save != 'show':
            try:
                f_name = (f'{group}_artefact_removal_blocks_{win_len}_'
                        f'cutoff_{n_stds_cut}_sd.jpg')
                plt.savefig(os.path.join(save, f_name), dpi=150,
                            facecolor='white')
                plt.close()
            except FileNotFoundError:
                print(f'Directory {save} is not valid')
        elif save == 'show':
            plt.show()
            plt.close()

    ## DROP CHANNELS WITH TOO MANY NAN'S
    ch_sel = [0, ]  # channels to keep, incl time
    ch_nms_out = ['time', ]
    for c in np.arange(new_arr.shape[1]):  # loop over rows of windows
        if c == 0:
            continue  # skip time row
        else:
            nans = np.isnan(new_arr[:, c, :]).sum()
            length = new_arr.shape[0] * new_arr.shape[2]
            nanpart = nans / length
            print(f'Ch {c}: {round(nanpart * 100, 2)}'
                  f'% is NaN (artefact or zero)')
            if nanpart < .5:
                ch_sel.append(c)
                ch_nms_out.append(ch_nms[c - 1])
    # EXCLUDE BAD CHANNELS WITH TOO MANY Nan's and Zero's
    out_arr = new_arr[:, ch_sel, :]


    return out_arr, ch_nms_out