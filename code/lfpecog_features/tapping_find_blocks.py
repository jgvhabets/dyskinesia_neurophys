
"""
Function to detect continuous tapping blocks
"""

# Import public packages and functions
import numpy as np

# Import own functions
from lfpecog_features.tapping_featureset import signalvectormagn

def find_active_blocks(
    acc_arr, fs, buff=5, buff_thr=.1, blocks_p_sec=8,
    act_wins_for_block = 2, verbose = True
):
    """
    Function detects tapping blocks in triaxial acc
    array. Is based on blocks with enough activity above
    standard deviation of the recording. Doesn't
    select on block length.
    Input:
        - acc_arr: tri-axial acc-array
        - fs (int): sample frequency
        - ...
    Returns:
        - acc_blocks (list): list containing one
            2d array with triaxial acc-data per block
        - block_indices (dict): containing two lists
            with the start and end sample-indices
            of the detected blocks (in original
            sampled indices)
    """
    sig = signalvectormagn(acc_arr)
    thresh = np.nanstd(sig)

    winl = int(fs / blocks_p_sec)
    # activity per window (acc > std.dev)
    act = np.array([sum(sig[i_start:i_start + winl] > thresh) / winl for
        i_start in np.arange(0, sig.shape[0], winl)])
    # blocks of windows with sufficient activity
    blocks = [sum(
        act[i_start - buff:i_start + buff] > buff_thr
    ) > act_wins_for_block for i_start in np.arange(buff, len(act) - buff)]
    # finding start and end indices of blocks
    block_indices = {'start': [], 'end': []}
    block_active = False
    for n, b in enumerate(blocks):
        if block_active:
            if b: continue
            else:
                block_indices['end'].append(n)
                block_active = False
        else:
            if b:
                block_indices['start'].append(n)
                block_active = True
            else:
                continue

    for rep in range(4):
        block_indices = merge_close_blocks(
            block_indices=block_indices,
            min_distance=blocks_p_sec * 2,
            verbose=verbose
        )
    
    # PM later: removing too short blocks/ in analysis

    block_indices = convert_win_ind_2_sample_ind(
        block_indices=block_indices, fs=fs, winl=winl,
    )
    acc_blocks = convert_sample_ind_2_acc_arrays(
        acc_arr, block_indices
    )

    if verbose: report_detected_blocks(block_indices, fs)

    return acc_blocks, block_indices


def merge_close_blocks(
    block_indices, min_distance, verbose
):
    """
    """
    new_block_indices = {'start': [], 'end': []}

    mergecount = 0
    skip_next = False

    for win, end in enumerate(block_indices['end']):
        try:
            start = block_indices['start'][win + 1]
        except IndexError:
            if skip_next: continue
            else:
                new_block_indices['start'].append(
                    block_indices['start'][win]
                )
                new_block_indices['end'].append(
                    block_indices['end'][win]
                )
                continue

        if skip_next:
            skip_next = False
            continue

        elif (start - end) < min_distance:        
            new_block_indices['start'].append(
                block_indices['start'][win]
            )
            new_block_indices['end'].append(
                block_indices['end'][win + 1]
            )
            mergecount += 1
            skip_next = True

        else:
            new_block_indices['start'].append(
                block_indices['start'][win]
            )
            new_block_indices['end'].append(
                block_indices['end'][win]
            )
    if verbose: print(f'Blocks merged: {mergecount}')
    return new_block_indices


def convert_win_ind_2_sample_ind(
    block_indices: dict, fs: int, winl: int,
):
    """
    Set indices back to original sample-indices
    of high freq acc array instead of indices of
    window lengths
    """
    sample_indices = {'start': [], 'end': []}
    for key in block_indices.keys():
        sample_indices[key] = np.around(np.array(
            block_indices[key]) * winl, 0
        ).astype(int)
    
    return sample_indices


def convert_sample_ind_2_acc_arrays(
    acc_arr, block_indices
):
    """
    Stores tri-axial acc-arrays per block, in a
    Python list.
    """
    acc_blocks = [
        acc_arr[:, i1:i2] for i1, i2 in zip(
            block_indices['start'], block_indices['end']
        )
    ]

    return acc_blocks


def report_detected_blocks(block_indices, fs):
    """
    Report on detected block number and lengths, takes
    block_indices after conversion to sample-indices
    """
    block_lengths = []
    for b in np.arange(len(block_indices['start'])):
        block_lengths.append(
            (block_indices['end'][b] - 
            block_indices['start'][b]) / fs
        )

    print(f'# {len(block_lengths)} tapping blocks detec'
            f'ted, lengths (in sec): {block_lengths}')



"""
For visualisation:
thresh = np.nanstd(sig)
act = np.array([sum(svm[i_start:i_start + winl] > thresh) / winl for
        i_start in np.arange(0, sig.shape[0], winl)])
blocks = [sum(act[i_start - buff:i_start + buff] > buff_thr) > 5 for
        i_start in np.arange(buff, len(act) - buff)]

axes[n].plot(act, color='blue', alpha=.4,
        label=f' Part of sig > std.dev. (window: {np.round(winl / fs, 2)})')
    axes[n].plot(blocks, color='red', alpha=.5,
        label=f'>70% active in window {np.round(2 * buff * winl / fs)} s')

func_blocks = find_blocks.find_active_blocks(dat, 250)
for start_i, end_i in zip(func_blocks['start'], func_blocks['end']):
    axes[1].fill_betweenx(x1=start_i, x2=end_i, y=[1.1, 1.2],
        color='gray',alpha=.7,
    )
"""