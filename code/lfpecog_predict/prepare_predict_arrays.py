"""
Functions to help prepare the arrays needed
for classification and prediction of
dyskinesia scores
"""

# import public functions
import numpy as np
from pandas import DataFrame, concat
from itertools import compress

# import own custom functions
from lfpecog_plotting.plot_pred_preparation import boxplot_zscored_LID_features


def get_group_arrays_for_prediction(
    feat_dict, label_dict, CDRS_THRESHOLD: .1,
    TO_PLOT = False,
    EXCL_CODE = 99
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
    # create empty list to store individual values for next process part
    X_total = []
    y_total_binary = []
    y_total_scale = []
    sub_ids_total = []
    ft_times_total = []

    for i_s, sub in enumerate(list(feat_dict.keys())):
        ft_names = []

        ### Create Y-labels based on CDRS (FT_LABELS)
        no_LID_sel = np.array(label_dict[sub]) == 0
        LID_sel = np.array(label_dict[sub]) >= CDRS_THRESHOLD

        # create binary y-labels
        sub_y_bin = []  # y as binary
        for noLID, LID in zip(no_LID_sel, LID_sel):
            if noLID: sub_y_bin.append(0)
            elif LID: sub_y_bin.append(1)
            else: sub_y_bin.append(EXCL_CODE)
        # add full scaled y-labels
        sub_y_scale = label_dict[sub]
        # append sub-codes to sub-id list (for later identifying subjects)
        sub_ids_total.append([sub] * feat_dict[sub].shape[0])  # add subject code, as many times as there are feature rows
        # add subjects ft-times to list (for later plotting)
        ft_times_total.append(feat_dict[sub].index.values)
        ### Create X with standardised Feature-arrays
        sub_X = np.zeros_like((feat_dict[sub]))

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
        X_total.append(sub_X)
        y_total_binary.append(sub_y_bin)
        y_total_scale.append(sub_y_scale)

    if TO_PLOT:
        boxplot_zscored_LID_features(
            subs_list=list(feat_dict.keys()),
            X_total=X_total,
            y_total_binary=y_total_binary,
            ft_names=ft_names,
            ftLabel_dict=label_dict,
            TO_SAVE_FIG=False,
            figname='LID_ssdFeatures_boxplots_indiv_zScored'
        )
    return X_total, y_total_binary, y_total_scale, sub_ids_total, ft_times_total, ft_names


def merge_group_arrays(X_total, y_total_binary,
                       y_total_scale, sub_ids_total,
                       ft_times_total,
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
            X_all = X_sub.copy()
            y_all_binary = y_sub.copy()
            y_all_scale = list(y_total_scale[i].copy())
            sub_ids = list(sub_ids_total[i].copy())
            ft_times_all = list(ft_times_total[i].copy())

        else:
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
    X_all = X_all[~nan_row_sel]
    y_all_binary = y_all_binary[~nan_row_sel]
    y_all_scale = y_all_scale[~nan_row_sel]
    sub_ids = sub_ids[~nan_row_sel]
    ft_times_all = ft_times_all[~nan_row_sel]

    # remove all rows not belonging to defined two outcome classes
    # (for example: if 0 is CDRS=0, and 1 is CDRS>=3, then CDRS scores 1 and 2 are excluded)
    excl_score_sel = y_all_binary == EXCL_CODE

    X_all = X_all[~excl_score_sel.ravel()]
    y_all_binary = y_all_binary[~excl_score_sel]
    y_all_scale = y_all_scale[~excl_score_sel]
    sub_ids = sub_ids[~excl_score_sel]
    ft_times_all = ft_times_all[~excl_score_sel]

    # X_all contains n-windows, n-features
    # y_all contains y-values (n-windows)
    # sub_ids contains subject-codes corresponding to windows (n-windows)
    print(X_all.shape, y_all_binary.shape, y_all_scale.shape,
        sub_ids.shape, ft_times_all.shape)
    print(f'out of n={len(y_all_binary)} samples, n={sum(y_all_binary)} are Dyskinesia'
        f' ({round(sum(y_all_binary) / len(y_all_binary) * 100, 1)} %)')
    
    return X_all, y_all_binary, y_all_scale, sub_ids, ft_times_all