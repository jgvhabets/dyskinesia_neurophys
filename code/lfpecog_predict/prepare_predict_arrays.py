"""
Functions to help prepare the arrays needed
for classification and prediction of
dyskinesia scores
"""

# import public functions
import numpy as np
from pandas import DataFrame, concat
from itertools import compress, product
from scipy.stats import variation
from os.path import exists, join
from os import listdir

# import own custom functions
from lfpecog_plotting.plot_pred_preparation import (
    boxplot_zscored_LID_features
)
from lfpecog_analysis.ft_processing_helpers import (
    categorical_CDRS
)
from lfpecog_features.bursts_funcs import get_envelop
from utils.utils_fileManagement import (
    get_project_path, load_ft_ext_cfg,
    load_class_pickle, get_avail_ssd_subs
)
from lfpecog_features.feats_spectral_helpers import (
    get_indiv_gammaPeak_range
)


def get_group_arrays_for_prediction(
    feat_dict, label_dict, CDRS_THRESHOLD=.1,
    CDRS_CODING='binary', CATEG_CDRS=False,
    MILD_CDRS=5, SEV_CDRS=10,
    TO_PLOT = False, EXCL_CODE = 99
):
    """
    Gets list of lists-per-subject with arrays
    for prediction

    Returns:
        - X_total, 
        - y_total_binary, 
        - y_total_scale, 
        - sub_ids_total, 
        - ft_times_total
        - ft_names
    """
    assert CDRS_CODING in ['binary', 'categorical'], (
        'CDRS_CODING should be categorical, linear, or binary'
    )
    # create empty list to store individual values for next process part
    X_total = []
    y_total_coded = []
    y_total_scale = []
    sub_ids_total = []
    ft_times_total = []

    for i_s, sub in enumerate(list(feat_dict.keys())):

        ft_names = []
        # add full scaled y-labels
        sub_y_scale = label_dict[sub]
        if CATEG_CDRS:
            sub_y_scale = categorical_CDRS(
                sub_y_scale, preLID_separate=False,
                preLID_minutes=0,
                cutoff_mildModerate=MILD_CDRS,
                cutoff_moderateSevere=SEV_CDRS
            )

        # append sub-codes to sub-id list (for later identifying subjects)
        sub_ids_total.append([sub] * feat_dict[sub].shape[0])  # add subject code, as many times as there are feature rows
        # add subjects ft-times to list (for later plotting)
        ft_times_total.append(feat_dict[sub].index.values)
        ### Create X with standardised Feature-arrays
        sub_X = np.zeros_like((feat_dict[sub]))

        ### Create coded Y-labels based on CDRS (FT_LABELS)
        if CDRS_CODING == 'binary':
            no_LID_sel = np.array(label_dict[sub]) == 0
            LID_sel = np.array(label_dict[sub]) >= CDRS_THRESHOLD
            if sum(no_LID_sel) == 0:
                if sub == '012': no_LID_sel = feat_dict[sub].index.values < 0
                elif sub == '102': no_LID_sel = feat_dict[sub].index.values < 0
                elif sub == '008': no_LID_sel = feat_dict[sub].index.values < 3
                else: raise ValueError(f'for subject {sub}, no NONE-LID moments'
                                       ' found for feature z-scoring')

            # create binary y-labels
            sub_y_coded = []  # y as binary
            for noLID, LID in zip(no_LID_sel, LID_sel):
                if noLID: sub_y_coded.append(0)
                elif LID: sub_y_coded.append(1)
                else: sub_y_coded.append(EXCL_CODE)    
                    

        for n_col, ft in enumerate(feat_dict[sub].keys()):

            ft_names.append(ft)
            values = feat_dict[sub].values[:, n_col]
            # Z-score values based NO-LID mean and std
            noLID_values = values[no_LID_sel]
            m = np.nanmean(noLID_values)
            sd = np.nanstd(noLID_values)
            Z_ALL_values = (values - m) / sd
            sub_X[:, n_col] = Z_ALL_values  # store all feats for pred-exploration
            
        # add subject values to total lists
        print(f'\tfor sub-{sub}, added X-shape: {sub_X.shape}')
        X_total.append(sub_X)
        y_total_coded.append(sub_y_coded)
        y_total_scale.append(sub_y_scale)

    if TO_PLOT:
        boxplot_zscored_LID_features(
            subs_list=list(feat_dict.keys()),
            X_total=X_total,
            y_total_binary=y_total_coded,
            ft_names=ft_names,
            ftLabel_dict=label_dict,
            TO_SAVE_FIG=False,
            figname='LID_ssdFeatures_boxplots_indiv_zScored'
        )
    return X_total, y_total_coded, y_total_scale, sub_ids_total, ft_times_total, ft_names


def merge_group_arrays(X_total, y_total_binary,
                       y_total_scale, sub_ids_total,
                       ft_times_total, ext_acc_arr=False,
                       EXCL_CODE=99,):
    """
    Takes lists with subject-arrays and merges all list
    to single arrays containing all selected windows of
    all subjects.
    All resulting arrays correspond (features, binary/linear
    scores, subject-IDs, timestamps)

    Returns:
        - X_all, y_all_binary, y_all_scale, sub_ids, ft_times_all
    """
    # merge all features and labels per Subject together
    for i, (X_sub, y_sub) in enumerate(zip(X_total, y_total_binary)):
        # loop over list with arrays of feats and labels per subject
        if i == 0:
            # create lists
            X_all = X_sub.copy()
            y_all_binary = list(y_sub).copy()
            y_all_scale = list(y_total_scale[i].copy())
            sub_ids = list(sub_ids_total[i].copy())
            ft_times_all = list(ft_times_total[i].copy())
            # print(sub_ids[0], len(sub_ids_total[i]))

        else:
            # add to lists
            X_all = np.concatenate([X_all, X_sub], axis=0)
            y_all_binary.extend(y_sub)
            y_all_scale.extend(y_total_scale[i])
            sub_ids.extend(sub_ids_total[i])
            ft_times_all.extend(ft_times_total[i])

    y_all_binary = np.atleast_2d(y_all_binary).T
    y_all_scale = np.atleast_2d(y_all_scale).T
    sub_ids = np.atleast_2d(sub_ids).T
    ft_times_all = np.atleast_2d(ft_times_all).T

    # remove all Rows containing NaN Features
    nan_row_sel = np.isnan(X_all).any(axis=1)
    print(f'removed NaN-rows n={sum(nan_row_sel)} out of {len(nan_row_sel)}')
    NAN_SUBS = sub_ids[nan_row_sel]
    
    X_all = X_all[~nan_row_sel]
    y_all_binary = y_all_binary[~nan_row_sel]
    y_all_scale = y_all_scale[~nan_row_sel]
    sub_ids = sub_ids[~nan_row_sel]
    ft_times_all = ft_times_all[~nan_row_sel]
    if isinstance(ext_acc_arr, np.ndarray): ext_acc_arr = ext_acc_arr[~nan_row_sel]

    # remove all rows not belonging to defined two outcome classes
    # (for example: if 0 is CDRS=0, and 1 is CDRS>=3, then CDRS scores 1 and 2 are excluded)
    excl_score_sel = y_all_binary == EXCL_CODE
    print(f'removed rows based on EXCL_CODE: n={sum(excl_score_sel)} out of {len(excl_score_sel)}')
    
    X_all = X_all[~excl_score_sel.ravel()]
    y_all_binary = y_all_binary[~excl_score_sel]
    y_all_scale = y_all_scale[~excl_score_sel]
    sub_ids = sub_ids[~excl_score_sel]
    ft_times_all = ft_times_all[~excl_score_sel]
    if isinstance(ext_acc_arr, np.ndarray): ext_acc_arr = ext_acc_arr[~excl_score_sel.ravel()]

    # X_all contains n-windows, n-features
    # y_all contains y-values (n-windows)
    # sub_ids contains subject-codes corresponding to windows (n-windows)
    print(f'out of n={len(y_all_binary)} samples, n={sum(y_all_binary > 0)} are Dyskinesia'
          f' ({round(sum(y_all_binary > 0) / len(y_all_binary) * 100, 1)} %)')
    
    if isinstance(ext_acc_arr, np.ndarray):
        return X_all, y_all_binary, y_all_scale, sub_ids, ft_times_all, ext_acc_arr
    else:
        return X_all, y_all_binary, y_all_scale, sub_ids, ft_times_all



def get_movement_selected_arrays(
    sub_class,
    FT_MOVE_SELs = {'MOVE_INDEP': ['REST',],
                    'MOVE_DEPEND': ['INVOLUNT', 'VOLUNTARY']},

):
    """
    get arrays selected on movement state for feature
    calculation and prediction. Takes several acc-selected
    states from subclass (ephys_selections_SUB.P).
    creates per subject an array with n-samples long and
    11-columns  (7 bands, cdrs, time, task, move-code),
    containing all non-movement or all movement selections,
    default move-coding: no free, no rest in between taps.

    Arguments:
        - subclass: (ephys_selections_SUB.P)
        - FT_MOVE_SELS: default selection of movement
            dependent vs independent selections

    Returns:
        - sub_arrs (dict), containing first all ephys_sources,
            then MOVE_INDEP/MOVE_DEPEND with lists with all
            arrays from selections ()
    """
    # execute per subject
    sub_arrs = {src: {m: [] for m in FT_MOVE_SELs.keys()}
                for src in sub_class.ephys_sources}

    for src, sel in product(sub_class.ephys_sources,
                            sub_class.incl_selections):
        # select only source selections
        if src not in sel: continue
        if 'lidall' in sel: continue
        # if src != 'lfp_right': continue

        # continue with correct source and movement states
        temp_arr = getattr(sub_class, sel).ephys_2d_arr
        temp_arr = np.concatenate(
            [temp_arr, np.atleast_2d(getattr(sub_class, sel).cdrs_arr).T],
            axis=1)  # add CDRS
        temp_arr = np.concatenate(
            [temp_arr, np.atleast_2d(getattr(sub_class, sel).time_arr).T],
            axis=1)  # add timestamps
        temp_arr = np.concatenate(
            [temp_arr, np.atleast_2d(getattr(sub_class, sel).task_arr).T],
            axis=1)  # add exp-task
        # find movement dep group
        for i_mov, mov in enumerate(FT_MOVE_SELs):  # 0 is mov INDEP, 1 is move DEPEND
            if any([m in sel for m in FT_MOVE_SELs[mov]]):
                # add last array coding for movement dependency
                temp_arr = np.concatenate(
                    [temp_arr, np.atleast_2d([i_mov] * temp_arr.shape[0]).T],
                    axis=1)  # add movement coding
                sub_arrs[src][mov].append(temp_arr)
    
    print(f'return arrays for {sub_class.sub}: {sub_arrs.keys()}')

    return sub_arrs


def get_move_selected_env_arrays(
    sub, LOAD_SAVE: bool = True,
    FT_VERSION: str = 'v6', GAMMA_WIDTH: int = 5,
    INCL_GAMMA_PEAK: bool = True,
    INCL_GAMMA_BROAD: bool = True,
    sfreq: int = 2048,
):
    """
    Takes the lists with arrays selected on
    movement dependency (created in get_movement_selected_arrays())
    and returns freq-band envelop arrays.
    only takes indiv-source-gamma-peak-freq (based on variation).
    Loads sub_classes for extraction from external HD (hardcoded).

    Arguments:
        - sub_class (ephys_elections)
        - sub_arrs (from get_movement_selected_arrays())
        - LOAD_SAVE: if True tries to load and saves if
            newly created
        - GAMMA_WIDTH: freq-bin width for indiv gamma peak finding

    Returns:
        - env_arr (dict): dict containing per ephys-source
            one array with rows: freq-bands, CDRS, timestamps,
            tasks, mov-coding; eavh row n-samples (ALL)
        - env_ranges: dict with freq-band names and freq-ranges
    """
    env_arr = {}  # dict to fill and return

    # get settings for variables
    SETTINGS = load_ft_ext_cfg(FT_VERSION=FT_VERSION)

    # check available data
    data_v = SETTINGS['DATA_VERSION']
    win_len = SETTINGS['WIN_LEN_sec']
    win_overlap = SETTINGS['WIN_OVERLAP_part']
    data_path = join(get_project_path('data'),
                     f'windowed_data_classes_{win_len}s'
                     f'_{win_overlap}overlap', data_v,)
    files = listdir(join(data_path, f'movement_feature_arrays_ft{FT_VERSION}'))

    # get (new) bands names
    bands = SETTINGS['SPECTRAL_BANDS'].copy()
    bands = {'theta' if k == 'delta' else k: v for k, v in bands.items()}
    if INCL_GAMMA_BROAD: env_ranges = {k: v for k, v in bands.items()}
    else: env_ranges = {k: v for k, v in bands.items()[:-3]}
    if INCL_GAMMA_PEAK: env_ranges['gammaPeak'] = 0  # default to guarantee correct empty array build

    # check subject files, LOAD IF AVAILABLE
    if sum([sub in f for f in files]) >= 2 and LOAD_SAVE:
        fbands_out = {}
        # load present data
        for f in [f for f in files if sub in f ]:
            # get data array and define source
            src = f"{f.split('_')[1]}_{f.split('_')[2]}"
            env_arr[src] = np.load(join(data_path, f'movement_feature_arrays_ft{FT_VERSION}',f),
                                   allow_pickle=True)
            # define source freq ranges
            fbands_out[src] = {k: v for k, v in env_ranges.items()}  # take copy of env_ranges as start
            if FT_VERSION == 'v6':
                g_range = f.split('gamma')[1][:4]
                g_range = [int(g_range[:2]), int(g_range[2:])]
            elif FT_VERSION == 'v8':
                g_range = 'tobedefined'
            if INCL_GAMMA_PEAK:
                fbands_out[src]['gammaPeak'] = g_range  # add gamma peak per source
                print(f'...added peakGamma range: {g_range}')
            print(f'- LOADED {src}: {f}')

        return env_arr, fbands_out
 
    # only load piclke with all selections if not loaded
    ext_subclass_path = join(
        get_project_path('data', extern_HD=True),
        'windowed_data_classes_10s_0.5overlap', f'ft_{FT_VERSION}',
        'selected_ephys_classes_all'
    )
    print(f'...({sub}) new extraction, loading selections pickle')
    sub_class = load_class_pickle(join(ext_subclass_path,
                                       f'ephys_selections_{sub}.P'),
                                  convert_float_np64=True)
    # get subject arrays
    sub_arrs = get_movement_selected_arrays(sub_class=sub_class)
    
    # calculate full env arrays per source
    for src in sub_class.ephys_sources:
        f_sel = [sub_class.sub in f and src in f for f in files]  # creates bool
        if sum(f_sel) == 1 and LOAD_SAVE:

            src_file = np.array(files)[f_sel]
            env_arr[src] = np.load(join(data_path, src_file), allow_pickle=True)
            print(f'- LOADED: {src_file}')
            continue
        
        # if not existing or LOAD_SAVE is false
        print(f'- START {src.upper()} (sub-{sub_class.sub})')
        src_arrs = sub_arrs[src]  # take dict with MOV DEP/INDEP arrays per source
        n_all_samples = sum([a.shape[0] for m in src_arrs.values() for a in m])  # sum all arrays up
        env_arr[src] = np.zeros((len(env_ranges), n_all_samples))  # new arr to fill with band-envelops

        src_max_gamma_var = 0  # set at source-beginning for gamma peak finding

        # ravel lists in dicts to single-channel arrays (MOV - DEP+INDEP)
        for i_ch, (band, f_range) in enumerate(bands.items()):
            print(src, band)
            
            if 'gamma' in band:
                # loop over gamma bands and take freq bin with largest variation            
                sig = np.concatenate([a[:, i_ch] for m in src_arrs.values() for a in m])
                assert ~ any(np.isnan(sig)), f'NaNs in sig: {sum(np.isnan(sig))}'

                if INCL_GAMMA_PEAK and band == 'gammaPeak':
                    if FT_VERSION == 'v8':
                        f_range = get_indiv_gammaPeak_range(sub=sub, src=src)
                        print(f'...v8 INDIV GAMMA PEAK: {f_range}')
                elif INCL_GAMMA_PEAK:
                    for f_1 in np.arange(f_range[0],
                                            f_range[1] - (GAMMA_WIDTH - 1),
                                            GAMMA_WIDTH / 2):
                        f_range = [f_1, f_1 + GAMMA_WIDTH]
                    # get envelop within peak range
                    env = get_envelop(sig, fs=sfreq, bandpass_freqs=f_range)
                    var_f = variation(env)
                    if var_f > src_max_gamma_var:
                        src_max_gamma_var = var_f
                        indiv_gamma_range = f_range
                        # add highest gamma to env array (possibly overwritten by next higher gamma)
                        env_arr[src][-1, :] = env  # last row for indiv peak gamma
                
                elif INCL_GAMMA_BROAD:
                    env = get_envelop(sig, fs=sfreq, bandpass_freqs=f_range)
                    env_arr[src][i_ch, :] = env

            elif 'gamma' not in band:
                # for other bands take env over full freq range
                sig = np.concatenate([a[:, i_ch] for m in src_arrs.values() for a in m])
                assert ~ any(np.isnan(sig)), f'NaNs in sig: {sum(np.isnan(sig))}'
                
                env = get_envelop(sig, fs=2048, bandpass_freqs=f_range)
                
                try:
                    env_arr[src][i_ch, :] = env
                except ValueError:
                    if abs(len(sig) - len(env)) <= 6:
                        env_arr[src] = env_arr[src][:, :len(env)]
                    env_arr[src][i_ch, :] = env
        
        # after adding all freq-band envelops, add 'meta' info (order: CDRS, timestamps, tasks, mov-coding)
        meta_arr = np.concatenate([a[:, -4:] for m in src_arrs.values() for a in m]).T
        try:
            env_arr[src] = np.concatenate([env_arr[src], meta_arr], axis=0)
        except ValueError:
            env_arr[src] = np.concatenate(
                [env_arr[src],
                 meta_arr[:, :env_arr[src].shape[1]]], axis=0
            )  # single sample removed in env creation, adjust length

        if LOAD_SAVE:
            f_name = (f'sub{sub}_{src}_movEnvArray_gamma'
                      f'{int(indiv_gamma_range[0])}{int(indiv_gamma_range[1])}.npy')
            np.save(join(data_path,
                         f'movement_feature_arrays_ft{FT_VERSION}',
                         f_name),
                    env_arr[src], allow_pickle=True)
            print(f'...saved {f_name} (arr shape: {env_arr[src].shape})')
    
    if INCL_GAMMA_PEAK: env_ranges['gammaPeak'] = indiv_gamma_range


    return env_arr, env_ranges



if __name__ == '__main__':
    """
    Run (WIN): cd REPO/code: python -m  lfpecog_predict.prepare_predict_arrays
    """
    
    from lfpecog_analysis.psd_analysis_classes import (
        PSD_vs_Move_sub, metaSelected_ephysData
    )
    
    print(f'extract movement selected envelop arrays')

    FT_VERSION='v8'
    SETTINGS = load_ft_ext_cfg(FT_VERSION=FT_VERSION)

    SUBS = get_avail_ssd_subs(DATA_VERSION=SETTINGS["DATA_VERSION"],
                              FT_VERSION=FT_VERSION)
    
    for sub in SUBS:
        print(f'\nextract envelop-arrays for sub-{sub}')
        # get move-selected env arrays
        _, _ = get_move_selected_env_arrays(
            sub=sub, LOAD_SAVE=True,
            FT_VERSION=FT_VERSION,
        )