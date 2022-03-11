'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import os


def handTapDetector(
    SubClass,
    buffsec: float=.1,
    buffthr: float=.5,
    svthr: float=.5e-7,
    runs_excl: list=[],
    min_length=True,
    check_plots=False,
    savedir=None,
):
    '''
    Function to detect blocks of movement in handtapping
    runs. Works based on threshold for signal vector
    magnitude, and applies that on a block. When the number
    of samples that is above threshold, is higher than the
    block threshold: whole block is set as active.
    In the end consecutive blocks are merged.

    Arguments:
        - subClass: Class with imported preproc-data
        - buffsec: seconds to include in a buff-block
        - buffthr: part of buffer that has to be higher
            than threshold
        - svthr: the actual SVM threshold
        - runs_excl: if runs have to be skipped
        - min_length: if None, all blocks are incl, if
            given (int) that are min # samples for block,
            if True: length corresponds to 512 LFP samples

    Returns:
        - RunMoveBlocks (dict): contains dict's per run
        with active blocks, contains the timestamps.

    TODO: Later include laterality of movement!!
    '''
    RunMovBlocks = {}
    for run in SubClass.runs_incl:
        if run in runs_excl: continue
        if run[:5] != 'Selfp': continue  # skip non-mov tasks
        print(f'\nStart {run}')
        
        time = SubClass.runs[run].acc_right_arr[0, :]
        # calculate signal vector magnitudes
        svR = np.sqrt(
            SubClass.runs[run].acc_right_arr[1, :]**2 +
            SubClass.runs[run].acc_right_arr[2, :]**2 +
            SubClass.runs[run].acc_right_arr[3, :]**2
        )
        svL = np.sqrt(
            SubClass.runs[run].acc_left_arr[1, :]**2 +
            SubClass.runs[run].acc_left_arr[2, :]**2 +
            SubClass.runs[run].acc_left_arr[3, :]**2
        )

        buff = int(buffsec / (1 / SubClass.runs[run].acc_left_Fs))
        mov_bool = np.zeros(len(svL))  # Boolean array movement presence
        # has same length as signal arrays
        thr_block= buffthr * buff  # threshold part of block active

        for n in np.arange(buff, len(svL)):
            # if enough SVM-values in block are above thr,
            # block is set to 1 (Left + Right evaluated)
            if sum(svL[n - buff:n] > svthr) > thr_block:
                mov_bool[n - buff:n] = np.ones(buff)
            elif sum(svR[n - buff:n] > svthr) > thr_block:
                mov_bool[n - buff:n] = np.ones(buff)
        # collect consecutive blocks in dict
        mov_blocks = {}
        ACT = 0  # steady state if a block is running
        block_n = 0  # number of next block to store
        block = []  # next block to store if block ends
        for (m, t) in zip(mov_bool, time):
            if ACT:  # currently 'within' active block
                if m:  # current value is also active
                    block.append(t)  # extend block with time
                else:  # if current bool is not active
                    mov_blocks[block_n] = block  # store block
                    block = []  # set block to empty
                    block_n += 1  # update block number
                    ACT = 0  # set block state to inactive
            else:  # currently no active block
                if m == 1:  # start new block
                    block.append(t)
                    ACT = 1  # change state
                else:   # stays not active
                    continue
        
        block_l = []  # block lengths
        for b in mov_blocks:
            block_l.append(len(mov_blocks[b]))
        print('Numer of total blocks:', len(block_l))
        print('Numer of blocks >= 128 samples:',
            len([b for b in block_l if b >= 128]))

        if min_length:
            if type(min_length) == Boolean:
                accFs = SubClass.runs[run].acc_left_Fs
                lfpFs = SubClass.runs[run].lfp_left_Fs
                req_sec = 512 * (1 / lfpFs)
                min_len = req_sec / (1 / accFs)
            else:
                min_len = min_length
            excl_b = [b for b in mov_blocks if len(
                mov_blocks[b]) < min_len]
            for b in excl_b: del(mov_blocks[b])
        chron_blocks = {}  # restore chron numbering
        for i, b in enumerate(mov_blocks):
            chron_blocks[i] = mov_blocks[b]
        RunMovBlocks[run] = chron_blocks
        print('CHECK Numer of total blocks:', len(RunMovBlocks[run]))
        if check_plots:
            check_plots_handTapDetect(
                SubClass,
                RunMovBlocks,
                run,
                savedir,
            )

    return RunMovBlocks


def check_plots_handTapDetect(
    SubClass, RunMovBlocks, run,
    savedir,
):
    for side in ['acc_left_arr', 'acc_right_arr']:
        for r, ax in zip([1, 2, 3], ['X', 'Y', 'Z']):
            plt.plot(
                getattr(SubClass.runs[run], side)[0, :],
                getattr(SubClass.runs[run], side)[r, :].T,
                label=f'{ax} {side[4:-4]}',
            )

    for n, b in enumerate(RunMovBlocks[run]):
        if n == 1:
            plt.fill_between(
            RunMovBlocks[run][b],
            y1=-2e-6, y2=4e-6,
            color='gray', alpha=.2, hatch='/',
            label='HandTap detected'
            )
        else:
            plt.fill_between(
            RunMovBlocks[run][b],
            y1=-2e-6, y2=4e-6,
            color='gray', alpha=.2, hatch='/',
            )
    plt.ylabel('ACC (m/s/s)')
    plt.xlabel('Time (sec)')
    plt.ylim(-2e-6, 4e-6,)
    plt.title(f'HandTap Detection {run}\n'
              f'({SubClass.runs[run].sub} '
              f'{SubClass.runs[run].ses})')
    plt.legend(
        loc='upper left', bbox_to_anchor=(-.1, -.13),
        ncol=4, frameon=False,
    )
    plt.tight_layout(pad=.1)
    sub = SubClass.runs[run].sub
    fname = f'ACC_MOV_TASK_detection_{sub}_{run}'
    plt.savefig(os.path.join(savedir, fname),
        dpi=150, facecolor='w',)
    plt.show()
        