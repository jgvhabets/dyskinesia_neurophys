'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from datetime import datetime
from itertools import compress, product

# Import own functions
import lfpecog_features.moveDetection_preprocess as movePrep

def pausedTapDetector(
    subdat, tap_move_distance=1, std_svm_thr=1e-7,
    std_find_large_height = 5e-7, std_find_small_height = 2.5e-7,
):
    """
    Detect the moments of finger-raising and -lowering
    during a fingertapping task.
    Function detects the axis with most variation and then
    first detects several large/small pos/neg peaks, then
    the function determines sample-wise in which part of a
    movement or tap the acc-timeseries is, and defines the
    exact moments of finger-raising, finger-lowering, and
    the in between stopping moments. 

    Input:
        - subdat (Class): resulting from subjectData()
        - tap_move_distance: n seconds which have to between
            tap and other movement; too close other movements
            get excluded
    
    Return:
        - outlists (list of lists):
            for tap-lists: each tap-list contains 4
            dopa-times or dopa-indices [start-tap, 1st-peak,
            2nd-peak, end-tap]
            for move-lists: one move-list contains the
            peak-dopa-times or peak-dopa-indices of other
            movements
    """
    # find present acc-sides
    acc_sides = [
        s for s in [
            'left', 'right'
        ] if f'acc_{s}' in vars(subdat).keys()
    ]

    assert len(acc_sides) == 2, print(
        f'\n\t{len(acc_sides)} ACC ATTRIBUTES found in '
        f'{subdat.sub}, Class-Attr keys: {vars(subdat).keys()}'
    )

    out_lists = {}  # store output

    for side in acc_sides:
        print(
            '... detecting movement states for sub '
            f'{subdat.sub} {side} side'
        )
        # empty lists to store final outputs
        for v, m in product(['t', 'i'], ['tap', 'move']): out_lists[f'{side}_{m}_{v}'] = []

        fs = int(getattr(subdat, f'acc_{side}').fs)
        accDf = getattr(subdat, f'acc_{side}').data
        ax_sel = [
            k for k in accDf.keys() if 'ACC' in k
        ]
        if len(ax_sel) == 0:
            ax_sel = [
                k for k in accDf.keys() if k in ['X', 'Y', 'Z']
            ]

        acc_arr = accDf[ax_sel].values

        # PM check for polarity with prep-function

        mainAx_ind = movePrep.find_main_axis(acc_arr)

        times = accDf['dopa_time'].values

        if subdat.sub in ['105']:
            svm_thr = 5e-8  # correct for incorrect ACC range
            find_large_height = 3e-7
            find_small_height = 1e-7
        # elif subdat.sub in ['107', '108', '109']:
        #     svm_thr = 1e-7  # correct for incorrect ACC range
        #     find_large_height = 5e-7
        #     find_small_height = 2.5e-7
        else:
            svm_thr = std_svm_thr
            find_large_height = std_find_large_height
            find_small_height = std_find_small_height
        
        for task in ['tap', 'rest', 'free']:
            # ONLY FOR TAP TASK-BLOCK, other mvoement is defined post-hoc
            # perform per task-block
            taskBlock_inds = movePrep.find_task_blocks(accDf, task)  # gives DF-indices

            tap_i_list, tap_t_list = [], []  # start new list per side (per sub)
            move_i_list, move_t_list = [], []  # start new list per side (per sub)

            for bN, (iB1, iB2) in enumerate(
                zip(taskBlock_inds[0], taskBlock_inds[1])
            ):
                print(f'... in moveDetection {task} block {bN + 1} / {len(taskBlock_inds[0])}')
                sig3ax = acc_arr[iB1:iB2, :]
                sig = sig3ax[:, mainAx_ind]
                if max(sig) > 1:
                    print(f'CORRECTED THRESHOLDS for sub-{subdat.sub}, {side}')
                    if subdat.sub in ['105']:
                        svm_thr = .05  # correct for incorrect ACC range
                        find_large_height = .3
                        find_small_height = .1
                    # elif subdat.sub in ['107', '108', '109']:
                    else:
                        svm_thr = .1  # correct for incorrect ACC range
                        find_large_height = .5
                        find_small_height = .3
                sigInds = np.arange(iB1, iB2)
                sigTimes = times[iB1:iB2]
                svm = movePrep.signalvectormagn(sig3ax)
                smoothsvm = uniform_filter1d(
                    svm, size=int(fs / 4)
                )
                # # Find peaks to help movement detection
                largePos = find_peaks(
                    sig,
                    height=find_large_height,
                    distance=fs,  # 1 s
                )[0]
                smallPeaks = find_peaks(
                    svm,
                    height=find_small_height,
                    distance=fs,
                )[0]
                largeNeg = find_peaks(
                    -1 * sig,  # convert pos/neg for negative peaks
                    height=find_large_height,
                    distance=fs,
                )[0]
                # define peak-timings of TAPS
                if np.logical_or(
                    len(largePos) == 0, len(largeNeg) == 0
                ):
                    # not both large Pos and Neg peaks present
                    if len(largePos) > 0: otherLargePeaks = largePos
                    elif len(largeNeg) > 0: otherLargePeaks = largeNeg
                    else: otherLargePeaks = []

                else:  # both large Pos and Neg present (TAPS)

                    otherLargePeaks = []
                    tap_durs = []
                    for i_posP, posP in enumerate(largePos):
                        # check distance to closest negative peak

                        if min(abs(posP - largeNeg)) > (fs * .5):
                            # large peak without close negative peak
                            otherLargePeaks.append(posP)  # store for other movement
                            continue

                        # negative peak close enough to be a TAP
                    
                        # only include tap (pos and neg-peak) if they are within 0.5 sec
                        negP = largeNeg[np.argmin(abs(posP - largeNeg))]
                        # allocate two large peaks in tap-list
                        if posP > negP: temp_tap = [np.nan, negP, posP, np.nan]
                        if posP < negP: temp_tap = [np.nan, posP, negP, np.nan]

                        # search for beginning and end of tapping movement
                        # based on signal vector activity
                        temp_tap[0], temp_tap[-1] = find_local_act_borders(
                            posP, smoothsvm, svm_thr
                        )
                        if not temp_tap[0] and not temp_tap[-1]:
                            print(f'not found borders for peak {i_posP}')
                            continue
                        # print(f'TAP-duration: {(temp_tap[-1] - temp_tap[0])/fs}')
                        tap_durs.append((temp_tap[-1] - temp_tap[0])/fs)

                        try:
                            tap_t_list.append([sigTimes[t] for t in temp_tap])
                            tap_i_list.append([sigInds[t] for t in temp_tap])
                        
                        except ValueError:
                            print(f'valueError for adding {i_posP}')
                            continue  # skip where nan is left bcs of no border
                        
                        except IndexError:
                            print(f'indexError for adding {i_posP}')
                            continue
                    print(f'added {len(tap_durs)} TAPS (mean {np.nanmean(tap_durs)} seconds) '
                          f'for {task} block {bN + 1}'
                          f' / {len(taskBlock_inds[0])}')
                
                # exclude small Peaks close to Tapping-Peaks
                min_gap = tap_move_distance * fs
                
                if np.logical_and(
                    len(smallPeaks) == 0,
                    len(otherLargePeaks) == 0
                ): continue  # not small/other movements

                sel_peaks = []

                for p in smallPeaks:
                    # ignore moves too close to taps
                    try:
                        min_dist = min(abs(p - largePos))
                        if min_dist > min_gap:
                            sel_peaks.append(p)

                    except ValueError:
                        # if no large peaks in 
                        if largePos == []:
                            sel_peaks.append(p)
                
                for p in otherLargePeaks: sel_peaks.append(p)

                # find movement borders
                temp_moves = []
                for p in sel_peaks:
                    
                    i_s, i_e = find_local_act_borders(
                        p, smoothsvm, svm_thr
                    )
                    if not i_s and not i_e: continue
                    if (i_e - i_s) > ( 5 * fs): continue  # if window is too long (missing data)

                    temp_moves.append([i_s, i_e])
    
                temp_moves = remove_lists_with_NaN(temp_moves)

                try:
                    move_t_list.extend([sigTimes[t] for t in temp_moves])
                    move_i_list.extend([sigInds[t] for t in temp_moves])

                    print(f'added {len(temp_moves)} MOVES for {task} block {bN}')

                except ValueError:
                    continue  # skip where nan is left bcs of no border
                
                except IndexError:
                    print(temp_moves)
                    continue


            out_lists[f'{side}_tap_t'].extend(tap_t_list)
            out_lists[f'{side}_tap_i'].extend(tap_i_list)
            out_lists[f'{side}_move_t'].extend(move_t_list)
            out_lists[f'{side}_move_i'].extend(move_i_list)
        

    return out_lists


def remove_lists_with_NaN(
    list_of_lists
):
    sel = [~np.isnan(l).any() for l in list_of_lists]

    newlist = list(compress(list_of_lists, sel))

    return newlist


def find_local_act_borders(peak_i, svm_sig, svmThr):
    """
    returns indices of borders around activity peak,
    indices correspond to svm_sig-indices
    """
    # find moments below threshold
    idx_low_act = np.where(svm_sig < svmThr)[0]  # low due to tilde conversion
    dists_bools = idx_low_act - peak_i  # find distances to peak-index
    # find closest point on the left
    neg_dists = dists_bools[dists_bools < 0]
    if len(neg_dists) == 0: return False, False
    left_idx = idx_low_act[dists_bools < 0]
    try:
        idx_left_border = left_idx[np.argmin(abs(neg_dists))]
    except ValueError:
        if np.argmin(abs(neg_dists)) == len(left_idx):
            idx_left_border = len(left_idx) - 1
        else: return False, False
    # find closest point on the right
    pos_dists = dists_bools[dists_bools > 0]
    if len(pos_dists) == 0: return False, False
    right_idx = idx_low_act[dists_bools > 0]
    try:
        idx_right_border = right_idx[np.argmin(abs(pos_dists))]
    except ValueError:
        if np.argmin(abs(pos_dists)) == len(right_idx):
            idx_right_border = len(right_idx) - 1
        else: return False, False
    
    return idx_left_border, idx_right_border


def saveAllEphysRestblocks(
    ephysdata, fs, restTimes, dopaIntakeTime, runStart,
    savedir, ephysname, runname, winlen=1024,
):
    """
    Select ephys-data that correspond with detected
    accelerometer-rest moments. Prerequisite is that
    neurophysiology and accelerometer data come from
    identical time period. Sample frequencies can
    differ, but total time-period measured must be equal!
    Epochs are saved per window-length (default is 1024
    samples), this leads to not considering some data
    which falls out of this windows.

    Input:
        - ephysdata (arr): one-dimensional data array
            containing 1 timeseries of neurophys data

        - dopaIntakeTime (str): timepoint of L-Dopa
            intake in format 'YYYY-MM-DD HH-MM'

    """
    # timestamps from start (in sec)
    ephystime = np.arange(0, len(ephysdata), 1 / fs)
    # create empty nd-array to store rest data in epochs
    # of defined window length
    tempdat = np.zeros((1, winlen))
    rowTimes = []  # list to store times corresponding to data rows

    for timeIdx in restTimes[1:-1]:  # skip first and last
        # find acc-Rest-times in ephys-timestamps
        neuInd1 = np.where(ephystime == timeIdx[0])[0][0]
        neuInd2 = np.where(ephystime == timeIdx[1])[0][0]
        dat_sel = ephysdata[neuInd1:neuInd2]
        n_wins = len(dat_sel) // winlen
        dat_sel = np.reshape(
            dat_sel[:n_wins * winlen],
            (n_wins, winlen),
            order='C',  # fills row after row
        )
        tempdat = np.vstack([tempdat, dat_sel])
        # add timestamp of start-time for every included row
        for i in np.arange(n_wins):
            rowTimes.append(timeIdx[0] + (i * winlen / fs))

    rowTimes = np.round(rowTimes, 4)  # round .019999 values
    tempdat = tempdat[1:, :]  # exclude first row of zeros

    dopaIN = datetime.strptime(dopaIntakeTime, '%Y-%m-%d %H:%M')



def specTask_movementClassifier(
    acc_class, task: str = 'free',
    svm_thr=1e-7, verbose=False,    
):
    """
    Function finds specific movement blocks
    (containing ALL movement, no TAP selection)
    for a specific task. (used for movement labeling
    of FREE tasks)

    Args:
        - acc_clas: resulting from accDerivs.get_raw_acc_traces()
        - task: task to select
        - svm_thr: used for activity detection, adjusted
            automatically in specific cases
    
    Returns:
        - move_times, dict containing start and end-lists
            with TIMES (in sec) of movement blocks
    """
    # select out task data
    task_idx = ['task' in c for c in acc_class.colnames]
    task_arr = acc_class.data[:, task_idx].ravel()  # ravel to reduce to 1d for indexing
    # select task moments
    task_sel = task_arr == task

    # select times
    time_idx = ['time' in c for c in acc_class.colnames]
    time_sel = acc_class.data[task_sel, time_idx]

    # select acc data
    acc_idx = ['ACC' in c for c in acc_class.colnames]
    acc_arr = acc_class.data[:, acc_idx]
    acc_sel = acc_arr[task_sel, :]

    # define main axis
    mainAx_ind = movePrep.find_main_axis(acc_arr)

    # calculate and smooth SVM
    svm = movePrep.signalvectormagn(acc_sel)  # svm for task data
    svm = uniform_filter1d(
        svm, size=int(acc_class.fs / 4)
    )
    # find peaks (indices corr to task selection)
    mov_peaks = find_move_moments(
        main_ax_sig=acc_sel[:, mainAx_ind],
        svm=svm, sub=acc_class.sub,
        fs=acc_class.fs,
    )
    if verbose: print(f'# peaks found: {len(mov_peaks)}')
    # get borders of activity around movement peaks (idx corr to svm-indices)
    move_times = {'start': [], 'end': []}
    # prevents double peak processing if border is already set further
    last_border = -30 * 60 * acc_class.fs
    for p in mov_peaks:
        if p < last_border: continue
        i1, i2 = find_local_act_borders(
            p, svm_sig=svm, svmThr=svm_thr,  # svm is already task selected
        )
        if isinstance(i1, bool): continue  # skip incorrect indices
        # convert svm (tasl-selected) indices to time, and apply as mask
        t1, t2 = time_sel[i1], time_sel[i2]
        # check for too big jump bcs of missing data (> 60 seconds)
        if (t2 - t1) > 60: continue
        # add to lists
        move_times['start'].append(time_sel[i1])
        move_times['end'].append(time_sel[i2])
        last_border = i2  # update current border
        
    for k in move_times: move_times[k] = np.array(move_times[k])

    durations = [b - a for a,b in zip(move_times["start"],
                                      move_times["end"])]
    if verbose:
        print(f'...found {len(move_times["start"])} blocks, '
          f'mean duration: {round(np.mean(durations), 1)} seconds')
    
    return move_times


def find_move_moments(
    main_ax_sig, svm, sub, fs,
    large_height = 5e-7,
    small_height = 2.5e-7,
):
    """
    Find three different types of movement peaks
    that are similarly used for TAP detection
    
    - main_ax_sig:
    - svm: (smoothed!, on 250 ms)
    """
    # correct thresholds if necessary
    if max(main_ax_sig) > 1 and sub in ['105']:
        svm_thr = .05  # correct for incorrect ACC range
        large_height = .3
        small_height = .1
    elif max(main_ax_sig) > 1:
        svm_thr = .1  # correct for incorrect ACC range
        large_height = .5
        small_height = .3
    elif sub in ['105']:
        svm_thr = 5e-8  # correct for incorrect ACC range
        large_height = 3e-7
        small_height = 1e-7

    # Find peaks to help movement detection
    largePos = find_peaks(
        main_ax_sig,
        height=large_height,
        distance=fs,  # 1 s
    )[0]
    smallPeaks = find_peaks(
        svm,
        height=small_height,
        distance=fs,
    )[0]
    largeNeg = find_peaks(
        -1 * main_ax_sig,  # convert pos/neg for negative peaks
        height=large_height,
        distance=fs,
    )[0]

    all_peaks = np.sort(np.concatenate([largePos, smallPeaks, largeNeg]))
    all_peaks = np.unique(all_peaks)

    # PEAKS into find_local_act_borders()
    return all_peaks


def get_move_bool_for_timeArray(
    time_arr, move_times: dict,
    MAX_BLOCK_SEC: int = 60,
):
    """
    Args:
        - time_arr: time_arr to match with bool (in SECONDS)
        - move_times: dict containing 'start' and 
            'end' with TIMES of move blocks
        - MAX_BLOCK_SEC: blocks longer than this
            n of seconds are skipped
    
    Returns:
        - move_bool: size corresponds to time_arr,
            "1" for moveblocks, "0" for no movement
    """
    move_bool = np.zeros_like(time_arr)

    for t1, t2 in zip(move_times['start'], move_times['end']):
        if (t2 - t1) > MAX_BLOCK_SEC:
            print(f'very large block  {(t2 - t1) / MAX_BLOCK_SEC}')
        mov_sel = np.logical_and(time_arr > t1,
                                 time_arr < t2)
        move_bool[mov_sel] = 1
    
    return move_bool