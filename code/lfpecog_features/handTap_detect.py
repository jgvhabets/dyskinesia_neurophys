'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def handTapDetector(
    SubClass,
    buffsec: float=.1,
    buffthr: float=.5,
    svthr: float=.5e-7,
    runs_excl: list=[],
    min_length=True,
    check_plots=False,
    plot_annot=False,
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
        - check_plots (Boolean): make plots to check
            algorithm or not
        - plot_annot: if video-annotated times should be
            plotted -> gives file_path here, otherwise False
        savedir: diretory where to save plot if created

    Returns:
        - RunMoveBlocks (dict): contains dict's per run
        with active blocks, contains the ACC-timestamps.
        CAVE!! these timestamps are in 200 Hz, have to be
        converted to 800 Hz for Ephys!

    PM: FOR SUB-008 THIS WAS INDEX-FINGER TAPPING!
    '''
    RunMovBlocks = {}
    for run in SubClass.runs_incl:
        if run in runs_excl: continue
        # if run[:5] != 'Selfp': continue  # skip non-mov tasks
        print(f'\nStart {run}')
        
        time = SubClass.runs[run].acc_right_arr[0, :]
        sideblocks = {}
        # calculate signal vector magnitudes
        for side in ['left', 'right']:
            s_acc = f'acc_{side}_arr'
            svm = np.sqrt(
                getattr(SubClass.runs[run], s_acc)[1, :]**2 +
                getattr(SubClass.runs[run], s_acc)[2, :]**2 +
                getattr(SubClass.runs[run], s_acc)[3, :]**2
            )  # calculate sign vector magnitude

            buff = int(buffsec / (1 / SubClass.runs[run].acc_left_Fs))
            hbuff = int(buff / 2)
            mov_bool = np.zeros(len(svm))  # Boolean array movement presence
            # has same length as signal arrays
            thr_block= buffthr * buff  # threshold part of block active

            for n in np.arange(0, len(svm) - buff):
                # centers blocks in middle: middle value is 1
                # if enough SVM-values in block are above thr
                if sum(svm[n:n + buff] > svthr) > thr_block:
                    mov_bool[n + hbuff] = 1
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
            print(f'Number of total blocks {side}:', len(block_l))
            print('Number of blocks >= 128 samples:',
                len([b for b in block_l if b >= 128]))

            if min_length:
                if type(min_length) == Boolean:
                    accFs = SubClass.runs[run].acc_left_Fs
                    lfpFs = SubClass.runs[run].lfp_left_Fs
                    req_sec = 256 * (1 / lfpFs)
                    min_len = req_sec / (1 / accFs)
                else:
                    min_len = min_length
                excl_b = [b for b in mov_blocks if len(
                    mov_blocks[b]) < min_len]
                for b in excl_b: del(mov_blocks[b])
            chron_blocks = {}  # restore chron numbering
            for i, b in enumerate(mov_blocks):
                chron_blocks[i] = mov_blocks[b]
            sideblocks[side] = chron_blocks
            print(f'CHECK Number of total blocks {side}:',
                   len(sideblocks[side]))
        RunMovBlocks[run] = sideblocks

        # add blocks in timestamps next to second-floats
        # load tsv with start timestamps of neurophys+acc recording
        bids_dir =  ('/Users/jeroenhabets/OneDrive - Charité - Uni'
            'versitätsmedizin Berlin/BIDS_Berlin_ECOG_LFP/rawdata')
        sub = f'sub-{SubClass.runs[run].sub}'
        ses =  f'ses-{SubClass.runs[run].ses}'
        scans = os.path.join(
            bids_dir, sub, ses, f'{sub}_{ses}_scans.tsv'
        )
        scans = pd.read_csv(scans, sep='\t')
        # convert detected second-timestamps into pd-timestamps
        dopatime = run[-6:]
        # find matching starttime in scans.tsv
        for i in range(scans.shape[0]):
            if dopatime in scans['filename'][i]:
                RunMovBlocks[run]['starttime'] = pd.Timestamp(
                    scans['acq_time'][i]
                )
        # add timedeltas to starttime
        for side in ['left', 'right']:
            RunMovBlocks[run][f'{side}_stamps'] = {}
            for i in RunMovBlocks[run][side]:
                ds = [pd.Timedelta(t, 'sec') for t in 
                    RunMovBlocks[run][side][i]]
                RunMovBlocks[run][f'{side}_stamps'][i] = [
                    RunMovBlocks[run]['starttime'] + d for d in ds
                ]
        
        if check_plots:
            check_plots_handTapDetect(
                SubClass,
                RunMovBlocks,
                run,
                savedir,
                plot_annot,
            )

    return RunMovBlocks


def check_plots_handTapDetect(
    SubClass, RunMovBlocks, run,
    savedir, plot_annot,
):
    print(f'PLOTTING FIGURE {run} .....')
    # create range with timestamps along acc-array, instead of 
    # floats of seconds since start (use for x-axis plot)
    tstart = RunMovBlocks[run]['starttime']  # first timestamp in array
    nsamples = getattr(SubClass.runs[run],'acc_left_arr').shape[-1]
    arr_fs = getattr(SubClass.runs[run],'acc_left_Fs')
    tend = tstart + pd.Timedelta(1 / arr_fs, unit='s') * nsamples
    timeax = pd.date_range(start=tstart, end=tend, freq='5ms')[:-1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for side in ['acc_left_arr', 'acc_right_arr']:
        for r, axis in zip([1, 2, 3], ['X', 'Y', 'Z']):
            ax.plot(
                # getattr(SubClass.runs[run], side)[0, :],
                timeax,
                getattr(SubClass.runs[run], side)[r, :].T,
                label=f'{axis} {side[4:-4]}',
            )
    ylim1=-3e-6
    ylim2=5e-6
    clrs = {'left': 'steelblue', 'right': 'y'}  # colors for sides
    for side in ['left', 'right']:
        for n, b in enumerate(RunMovBlocks[run][side]):
            if n == 0:
                ax.fill_between(
                    # RunMovBlocks[run][side][b],
                    RunMovBlocks[run][f'{side}_stamps'][b],
                    y1=ylim1, y2=ylim2,
                    color=clrs[side], alpha=.1, hatch='/',
                    label=f'Tap {side} (Acc-detect.)'
                )
            else:
                ax.fill_between(
                    # RunMovBlocks[run][side][b],
                    RunMovBlocks[run][f'{side}_stamps'][b],
                    y1=ylim1, y2=ylim2,
                    color=clrs[side], alpha=.1, hatch='/',
                )    
        if plot_annot:
            annot = np.load(plot_annot, allow_pickle=True).item()
            try:
                ax.scatter(
                    annot[run][f'{side}_stamps'],
                    [4e-6] * len(annot[run][f'{side}_stamps']),
                    c=clrs[side], edgecolor='k', s=100, 
                    alpha=.5, marker='*',
                    label=f'Tap {side} (Video-ann.)',
                )
            except KeyError:
                print('No video-annotations for ', run)
                pass

    ax.set_ylabel('ACC (m/s/s)')
    ax.set_xlabel('Time (sec)')
    ax.set_ylim(ylim1, ylim2,)
    n_xticks = 7
    xticks = timeax[::len(timeax) // n_xticks]
    xtlabels = timeax[::len(timeax) // n_xticks].strftime('%X')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtlabels)

    if plot_annot:
        ax.set_title(f'Tap Acc-Detection vs Video-Annotation\n'
            f'{run} - {SubClass.runs[run].sub} '
            f'{SubClass.runs[run].ses})', size=14)
    else:
        ax.set_title(f'Tap Detection {run}\n'
            f'({SubClass.runs[run].sub} '
            f'{SubClass.runs[run].ses})', size=14)
    ax.legend(
        loc='upper left', bbox_to_anchor=(-.1, -.13),
        ncol=4, frameon=False, fontsize=12,
    )
    plt.tight_layout(pad=.1)
    sub = SubClass.runs[run].sub
    fname = f'ACC_MOV_TASK_detection_{sub}_{run}_sides_annot_alpha'
    plt.savefig(os.path.join(savedir, fname),
        dpi=150, facecolor='w',)
    plt.show()
        