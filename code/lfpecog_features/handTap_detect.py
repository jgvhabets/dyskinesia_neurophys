'''Feature Extraction Preparation Functions'''

# Import public packages and functions
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks





def handTapDetector(
    SubClass,
    buffsec: float=.05,
    TAPthr: float=5e-7,
    MOVthr: float=2e-7,
    runs_excl: list=[],
    min_len=0.1,
    check_plots=False,
    plot_annot=False,
    plotdir=None,
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
        - min_len (int): min block length in seconds
        - check_plots (Boolean): make plots to check
            algorithm or not
        - plot_annot: if video-annotated times should be
            plotted -> gives file_path here, otherwise False
        - plotdir: diretory where to save plot if created
        - savedir: diretory where to save taps

    Returns:
        - RunMoveBlocks (dict): contains dict's per run
        with active blocks, contains the ACC-timestamps.
        
    PM: FOR SUB-008 THIS WAS INDEX-FINGER TAPPING!
    '''
    RunMovBlocks = {}

    for run in SubClass.runs_incl:
        if run in runs_excl: continue
        print(f'\nStart {run}')
        
        # time = SubClass.runs[run].acc_right_arr[0, :]
        sideblocks = {}
        # calculate signal vector magnitudes
        for side in ['left', 'right']:
            sideblocks[side] = {}
            s_acc = f'acc_{side}_arr'
            svm = np.sqrt(
                getattr(SubClass.runs[run], s_acc)[1, :]**2 +
                getattr(SubClass.runs[run], s_acc)[2, :]**2 +
                getattr(SubClass.runs[run], s_acc)[3, :]**2
            )  # calculate sign vector magnitude
            accFs = getattr(SubClass.runs[run], f'acc_{side}_Fs')
            min_len_n = min_len / (1 / accFs)  # n of min samples in tap

            iState = {
                'Taps': np.where(svm > TAPthr)[0],
                'Moves': np.where(svm > MOVthr)[0]
            }
            for i in iState['Taps']:  # iterate every tap index
                # find Tap-i in Mov-i's, and delete: prevent doubles
                idel = np.where(iState['Moves'] == i)
                iState['Moves'] = np.delete(iState['Moves'], idel)
                
            gaps = 0.5  # seconds which ends a tap block
            gap_n = gaps / (1 / accFs)  # n of samples in gap
            
            for state in iState: sideblocks[side][state] = {}
            for state in iState:
                blockN = 0
                block = []
                for i, idiff in zip(
                    iState[state][:-1], np.diff(iState[state])
                ):
                    if idiff < gap_n:
                        # add consecut i's betw 2 i's in seconds!)
                        iadd = list(np.linspace(
                            start=i,
                            stop=i + idiff - 1,
                            num=idiff,
                        ) / accFs)
                        block.extend(iadd)
                    else:
                        if len(block) > min_len_n:
                            sideblocks[side][state][blockN] = block
                            blockN += 1
                        block = []
            
            # Check Tap-patterns (needs timestamp for annotation-check)
            newTaps, extraMoves = tap_pattern_checker(
                run=run, side=side,
                tapblocks=sideblocks[side]['Taps'],
                acc_og=getattr(SubClass.runs[run], s_acc),
                accFs=accFs,
                plot=check_plots,
                plotdir=plotdir,
            )
            sideblocks[side]['Taps'] = newTaps
            starti = len(sideblocks[side]['Moves'].keys())
            for movb in extraMoves:
                sideblocks[side]['Moves'][starti] = extraMoves[movb]
                starti += 1
            
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
            for state in ['Taps', 'Moves']:
                RunMovBlocks[run][f'{side}_{state}_stamps'] = {}
                for block in RunMovBlocks[run][side][state]:
                    ds = []
                    for t in RunMovBlocks[run][side][state][block]:
                        ds.append(pd.Timedelta(t, 'sec') )
                    RunMovBlocks[run][f'{side}_{state}_stamps'][block
                    ] = [RunMovBlocks[run]['starttime'] + d for d in ds]


        if check_plots:
            check_plots_handTapDetect(
                SubClass,
                RunMovBlocks,
                run,
                plotdir,
                plot_annot,
                fignamedetail=(f'buff{str(buffsec)[2:]}_Tap'
                    f'{str(TAPthr)[:1]}_{str(TAPthr)[-2:]}_Mov'
                    f'{str(MOVthr)[:1]}_{str(MOVthr)[-2:]}_'
                    f'gap{gaps * 1000}'
                )
            )
        
        if savedir:
            tap_saver(RunMovBlocks, savedir, sub)

    return RunMovBlocks


def check_plots_handTapDetect(
    SubClass, RunMovBlocks, run,
    plotdir, plot_annot, fignamedetail,
):
    print(f'PLOTTING FIGURE {run} .....')
    # create range with timestamps along acc-array, instead of 
    # floats of seconds since start (use for x-axis plot)
    tstart = RunMovBlocks[run]['starttime']  # first timestamp in array
    nsamples = getattr(SubClass.runs[run],'acc_left_arr').shape[-1]
    arr_fs = getattr(SubClass.runs[run],'acc_left_Fs')
    tend = tstart + pd.Timedelta(1 / arr_fs, unit='s') * nsamples
    timeax = pd.date_range(
        start=tstart, end=tend, freq=f'{1000 / arr_fs}ms')[:-1]

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

    # clrs = {'left': 'steelblue', 'right': 'y'}  # colors for sides
    alpha=.8
    alpha2=.2
    kwparms = {
        'left_Moves': {
            'color': 'springgreen',
            'alpha': alpha2,
        },
        'left_Taps': {
            'color': 'green',
            'alpha': alpha,
        },
        'right_Moves': {
            'color': 'gold',
            'alpha': alpha2,
        },
        'right_Taps': {
            'color': 'purple',
            'alpha': alpha,
        }
    }
    for side in ['left', 'right']:
        # color detected states
        for state in ['Taps', 'Moves']:
            for n, b in enumerate(RunMovBlocks[run][side][state]):
                if n == 0:  # add legend-label only once
                    ax.fill_between(
                        RunMovBlocks[run][f'{side}_{state}_stamps'][b],
                        y1=ylim1, y2=ylim2,
                        label=f'{state} {side} (Acc-detect.)',
                        **kwparms[f'{side}_{state}'],
                    )
                else:
                    ax.fill_between(
                        RunMovBlocks[run][f'{side}_{state}_stamps'][b],
                        y1=ylim1, y2=ylim2,
                        **kwparms[f'{side}_{state}'],
                    )
        # add manual annotations
        if plot_annot:
            annot = np.load(plot_annot, allow_pickle=True).item()
            try:
                ax.scatter(
                    annot[run][f'{side}_stamps'],
                    [4e-6] * len(annot[run][f'{side}_stamps']),
                    c=kwparms[f'{side}_Taps']['color'], edgecolor='k',
                    s=100, alpha=.5, marker='*',
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
    fname = f'ACC_TapvsMov_detection_{sub}_{run}_{fignamedetail}.png'
    plt.savefig(os.path.join(plotdir, fname),
        dpi=150, facecolor='w',)
    plt.close()
        

"""
copy function, include transfer from taps to move if not TAP-pattern
"""
def tap_pattern_checker(
    run, side, tapblocks, acc_og, accFs, tapAxis='y', 
    posThr=1e-7, negThr=1e-7, plot=False, plotdir=None,
):
    newTaps = {}  # true pattern tap-blocks: new Tap Dict
    extraMoves={}  # false pattern tap-blocks: convert to moves
    tap_i = 0  # indices to fill new dicts
    mov_i = 0
    smooth = False
    i_incl = 24
    if plot:
        fig, axes=plt.subplots(i_incl // 4, 4, figsize=(12,16),
            sharey='row')
        axes = axes.flatten()

    for b in np.arange(len(tapblocks)):
        if b >= i_incl: plot = False

        peakDict = {
            'pos': {
                'Thr': posThr,
                'dir': 1,
                'ind': [],
                'top': []
            },
            'neg': {
                'Thr': negThr,
                'dir': -1,
                'ind': [],
                'top': []
            }
        }

        try:
            i0 = tapblocks[b][0] * 200 
            i1 = tapblocks[b][-1] * 200 
        except KeyError:
            # print(f'Block {b} no more tap blocks')
            continue
        acc = {
            'x': acc_og[1, int(i0):int(i1)],
            'y': acc_og[2, int(i0):int(i1)],
            'z': acc_og[3, int(i0):int(i1)]
        }
        acc['svm'] = np.sqrt(acc['x']**2 + acc['y']**2 + acc['z']**2)
        
        for sig in acc.keys():
            # smoothinng
            if smooth:
                acc[sig] = pd.Series(acc[sig]).rolling(3).mean()
            if plot: axes[b].plot(acc[sig], alpha=.5, label=sig)

        for p in peakDict:
            peaks = find_peaks(
                peakDict[p]['dir'] * acc[tapAxis],
                # height=peakDict[p]['Thr'] * .1,
                width=1,
                distance=25,
                prominence=peakDict[p]['Thr'],
                wlen=40,
            )
            if len(peaks[0]) > 0:
                if plot:
                    axes[b].scatter(
                        peaks[0],
                        peakDict[p]['dir'] * peaks[1]['prominences'],
                        # label=f'{p} peaks'
                    )
                peakDict[p]['ind'].extend(peaks[0])
                peakDict[p]['top'].extend(peaks[1]['prominences'])
        # check pos-neg-neg-pos pattern
        peakFound = False
        try:
            # taps longer than 1 sec -> moves
            if len(acc[tapAxis]) > accFs:
                extraMoves[mov_i] = tapblocks[b]
                mov_i += 1
            # check tap-double-sinusoid (+ - - +)
            elif sum(np.logical_and(
                peakDict['neg']['ind'] > peakDict['pos']['ind'][0],
                peakDict['neg']['ind'] < peakDict['pos']['ind'][-1]
            )) >= 2:  # if there are 2 neg peaks between 2 pos peaks
                peakFound = True
                newTaps[tap_i] = tapblocks[b]
                tap_i += 1
            else:  # other pattern -> moves
                extraMoves[mov_i] = tapblocks[b]
                mov_i += 1

        except IndexError:
            extraMoves[mov_i] = tapblocks[b]
            mov_i += 1
            
        if plot:
            # axes[b].set_xticks(np.arange(0, len(x), 100), size=10)
            # axes[b].set_xticklabels(np.arange(i0, i0 + len(x), 100) / 200, size=10)
            # axes[b].set_xlabel('Time (seconds)', size=10)
            axes[b].set_ylim(-2e-6, 2e-6)
            axes[b].set_ylabel('ACC', size=10)
            if b == 0:
                axes[b].legend(fontsize=16, ncol=6, frameon=False,
                bbox_to_anchor=(0.5, 1.3), loc='upper left')
        
            # add peak detect as color
            if peakFound:
                axes[b].fill_between(
                    alpha=.1, color='green',
                    x=np.arange(len(acc['x'])), y1=-2e-6, y2=2e-6, )
            else:
                axes[b].fill_between(
                    alpha=.1, color='red',
                    x=np.arange(len(acc['x'])), y1=-2e-6, y2=2e-6, )
    
    if plot:
        
        fname = f'TapChecker_{run[-6:]}_{side}_scr'
        if smooth: fname += 'smooth'
        plt.savefig(
            os.path.join(plotdir, fname),
            dpi=150, facecolor='w',
        )
        plt.close()

    return newTaps, extraMoves 

    

def tap_saver(
    blocks, savedir, sub
):
    os.makedirs(savedir, exist_ok=True)

    dict_name = f'taps_moves_{sub}'
    np.save(
        os.path.join(savedir, dict_name),
        blocks
    )

    # TODO: add text file with parameters of blocks
    

    # # Load annotation dict
    # video_taps = np.load(os.path.join(deriv_dir, f'{dict_name}.npy'),
    #                      allow_pickle=True).item()

    return f'Tap and Moves blocks-dictionary saved for {sub}'