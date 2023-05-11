"""
Functions to import and ephys results
"""

# Import public packages and functions
from os.path import join, exists
import numpy as np
from pandas import concat, read_csv, DataFrame

# Import own functions
from utils.utils_fileManagement import (
    get_project_path, load_class_pickle, correct_acc_class
)
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side
)
from lfpecog_analysis.get_acc_task_derivs import get_n_and_length_taps



def get_mic_scores(
    sub,
    task = 'rest',
    data_version = 'v3.1',
    ecogSide_tapAdjust=False,
    winLen_sec = 60,
    part_winOverlap = 0.5,
):
    """
    Get Max Imag Coh scores for sub, data-version and task

    Inputs:
        - sub
        - task
        - data_version
        - ecogSide_tapAdjust: if True, tap will contain
            only tap-windows contra-lateral to ECOG side,
            rest will contain both rest AND ipsi-lat tap to ECoG side
    
    Returns:
        - mic_df of defined task
    """
    task = task.lower()  # prevent typos with capitals

    results_sub_dir = join(get_project_path('results'), 'features', 'mvc', f'sub{sub}')

    if task in ['rest', 'tap']:

        # select out contraECoG taps, add ipsi-ECoG-taps to Rest 
        if ecogSide_tapAdjust:
            
            ecogSided_fname = (
                f'mvc_fts_{sub}_absMIC_{task}_{data_version}'
                f'win{winLen_sec}s_overlap{part_winOverlap}_ecogSided.csv'
            )
            # load if available
            if exists(join(results_sub_dir, ecogSided_fname)):
                mic_df = read_csv(join(results_sub_dir, ecogSided_fname), index_col=0)
                return mic_df

            # create and store if not yet available
            mic_dfs = {}
            for t in ['rest', 'tap']:
                fname = (f'mvc_fts_{sub}_absMIC_{t}_{data_version}'
                    f'win{winLen_sec}s_overlap{part_winOverlap}.csv')
            
                mic_dfs[t] = read_csv(join(results_sub_dir, fname), index_col=0)
            # get new sorted dfs by side
            sided_dfs = {}
            sided_dfs['rest'], sided_dfs['tap'] = sort_resultDf_on_tappingSide(
                mic_dfs['rest'], mic_dfs['tap'], sub=sub,)
            # store both sided dfs
            for t in ['rest', 'tap']:
                ecogSided_fname = (
                    f'mvc_fts_{sub}_absMIC_{t}_{data_version}'
                    f'win{winLen_sec}s_overlap{part_winOverlap}_ecogSided.csv'
                )
                sided_dfs[t].to_csv(
                    join(results_sub_dir, ecogSided_fname),
                    index=True, header=True, sep=',',
                )

            if task == 'rest': return sided_dfs['rest']
            elif task == 'tap': return sided_dfs['tap']

        # use original task and tap labels from recording
        else:    
            mvc_fts_task_file = (f'mvc_fts_{sub}_absMIC_{task}_{data_version}'
                f'win{winLen_sec}s_overlap{part_winOverlap}.csv'
            )
            mic_df = read_csv(join(results_sub_dir, mvc_fts_task_file), index_col=0)

            return mic_df
    
    # get both merged together
    elif np.logical_or(
        'rest' in task and 'tap' in task,
        task == 'both'
    ):
        # combine both tasks
        dfs = {}
        for t in ['rest', 'tap']:  
            task_file = (f'mvc_fts_{sub}_absMIC_{t}_{data_version}'
                f'win{winLen_sec}s_overlap{part_winOverlap}.csv'
            )
            dfs[t] = read_csv(join(results_sub_dir, task_file), index_col=0)

        # concatenate and sort on index
        mic_df = concat([dfs['rest'], dfs['tap']]).sort_index()
    
        return mic_df


def get_peakFreq_in_timeFreq(
    tf_values, times, freqs,
    bin_marge=1, f_min=60, f_max=90,
):
    """

    Inputs:
        - tf_values: 2d array with time freq results
        - times: corr to tf_values
        - freqs: corr to tf_values
        - bin_marge (int): n of bins to add at both sides of freq

    Returns:
        - peak_idx: index of peak freq in freqs
        - peak_f: column name of peak freq
    """
    # check shapes match
    if not np.logical_and(
        tf_values.shape[0] == len(times),
        tf_values.shape[1] == len(freqs)
    ): tf_values = tf_values.T
    # check freqs are no strings
    if type(freqs[0]) == str:
        freqs = [float(f) for f in freqs]
    # create empty lists to store
    max_list, f_list, i_col_list = [], [], []

    for i_col, f in zip(range(len(freqs)), freqs):
        
        if f < f_min or f > f_max: continue
        # take 90th percentile of mean freq-band
        mean_vals = np.mean(tf_values[
            :,
            i_col - bin_marge:i_col + bin_marge + 1
        ], axis=1)

        max_list.append(np.percentile(mean_vals, 90))
        f_list.append(f)
        i_col_list.append(i_col)
    # add minus to array to order descending    
    max_order = np.argsort(-np.array(max_list))
    # take first in order
    peak_idx = np.array(i_col_list)[max_order][0]
    peak_f = np.array(f_list)[max_order][0]

    return peak_idx, peak_f



def get_most_var_freq(
    values, 
    lo_border=60, hi_border=85,
    width=2
):
    """
    values has to be df (n-times x n-freqs)
    """

    # get gamma freq indices in keys
    gamma_f_ind = np.where(
        [hi_border > f > lo_border
        for f in values.keys().astype(float)]
    )[0]
    var_freqs = [
        # scipy.stats.variation(
        np.percentile(
            np.mean(values.values[:, n:n + width], axis=1)
        , 90)
        for n in gamma_f_ind
    ]
    

    var_ord = np.argsort(var_freqs)

    var_freq_i = gamma_f_ind[var_ord][-1]
    
    return var_freq_i


def sort_resultDf_on_tappingSide(
    rest_df, tap_df, sub,
    winLen_sec = 60, data_version='v3.1',
):
    """
    Split a feature df from tapping task
    into contrlat tapping to ECoG-side, and
    add ipsi-lateral tapping to rest

    Input:
        - rest_df: df with ft values in rest
        - tap_df: df with ft values in tapping
        - sub: string code
    
    Returns:
        - rest_df: extended with ipsi-ECoG-tapping
        - new_tap_df: only contra-ECoG tapping
    """
    new_tap_df = DataFrame(columns=tap_df.keys())

    # load Acc-Data from piclked dataClass
    acc = load_class_pickle(join(
            get_project_path('data'),
            'merged_sub_data', data_version,
            f'{sub}_mergedDataClass_{data_version}_noEphys.P'
        ))
    acc = correct_acc_class(acc)

    # SELECT BASED ON UNILATERAL TAP SIDE
    ecog_side = get_ecog_side(sub)
    if ecog_side == 'right':
        i_tap = np.where(acc.colnames == 'left_tap')[0][0]
    elif ecog_side == 'left':
        i_tap = np.where(acc.colnames == 'right_tap')[0][0]

    taps = acc.data[:, i_tap]

    for winStart_t in tap_df.index.values:

        win_idx = np.logical_and(
            acc.times > winStart_t, acc.times < (winStart_t + winLen_sec))
        win_taps = taps[win_idx]

        ntaps, _ = get_n_and_length_taps(win_taps, acc.fs)
        
        # add to tap or rest df
        if ntaps >= 4:
            new_tap_df = concat([new_tap_df, tap_df.loc[winStart_t].to_frame().T],)
        
        else:
            rest_df = concat([rest_df, tap_df.loc[winStart_t].to_frame().T],)
    
    rest_df = rest_df.sort_index()
        
    return rest_df, new_tap_df
    