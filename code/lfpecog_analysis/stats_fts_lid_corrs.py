"""
calculate stats for psd comparisons
"""

# import libraries
import numpy as np
import pandas as pd
import json
import os
# from statsmodels.formula.api import mixedlm
import gpboost as gpb
from scipy.stats import norm, pearsonr
from itertools import product

# import own functions
from utils.utils_fileManagement import (get_project_path,
                                        make_object_jsonable)



def replace_gammas_for_maxGamma(df):

    gamma_keys = [k for k in list(df.keys()) if any(
        ['gamma1' in k, 'gamma2' in k, 'gamma3' in k]
    )]
    if len(gamma_keys) == 0: return df

    # check local spectral features
    for source, var in product(['lfp', 'ecog'],
                               ['mean_psd', 'variation']):
        g_keys = [k for k in gamma_keys
                  if source in k and var in k]
        if len(g_keys) > 0:
            new_name = g_keys[0].replace('gamma1', 'gamma')
            df[new_name] = df[g_keys].max(axis=1)

    # check coherences
    for source, var in product(['STN_STN', 'STN_ECOG'],
                               ['sq_coh', 'imag_coh']):
        g_keys = [k for k in gamma_keys
                  if source in k and var in k]
        if len(g_keys) > 0:
            new_name = g_keys[0].replace('gamma1', 'gamma')
            df[new_name] = df[g_keys].max(axis=1)

    df = df.drop(columns=gamma_keys)

    return df


def normalize_feats_scores(df, NORM_CDRS=False,
                           NORM_METH='norm'):

    skip_norm_cols = ['sub', 'LID', 'ACC']
    if not NORM_CDRS: skip_norm_cols.append('CDRS')
    
    for ft in list(df.keys()):
        
        if ft in skip_norm_cols: continue

        if NORM_METH == 'norm':
            df[ft] = df[ft] / np.max(df[ft])  #  normalize for R
        elif NORM_METH == 'std':
            df[ft] = (df[ft]) / np.nanstd(df[ft]) #  standardize for R
        
    return df


def get_R_smResults(result, X, y, ft=None,):
    # print(result.summary())

    # based on covariance param
    # R1 = result.cov_params()[ft][1] / (np.std(X) * np.std(y))
    # based on coeff
    R2 = result.fe_params[1] / (np.std(X) * np.std(y))

    R = result.fe_params[1]

    p = result.pvalues[ft]
        
    # print(f'R (cov): {R1}, R (coef): {R2}')

    return R, p


def get_ft_lid_corrs(feat_dict, label_dict, acc_dict=False,
                     TARGET='CDRS', STAT_METH='lmm',
                     NORM_CDRS=False,
                     RETURN_STAT_DF=False, feats_incl=False,
                     EXCL_NO_DYSK=False,):
    
    if STAT_METH == 'pearson' and isinstance(feats_incl, list):
        corrs = {f: [] for f in feats_incl}

    if STAT_METH == 'lmm':
        # lm_df = pd.DataFrame(columns=)  #  + feats_incl
        noFtCols = ['sub', 'CDRS', 'LID']
        if len(acc_dict) == len(feat_dict):
            noFtCols.append('ACC')
        ftCols = []
        # lm_nan_row = [np.nan] * lm_df.shape[1]

    for i_s, sub in enumerate(list(feat_dict.keys())):
        # check for non-dysk
        if (label_dict[sub] == 0).all():
                if EXCL_NO_DYSK:
                    print(f'no dyskinesia for sub-{sub}')
                    continue
                else:
                    print(f'included sub-{sub} without dyskinesia')
        # loop over feature to include every feature in correct matter
        i_f = 0  # create own counter due to none-feat-cols
        for f_sel in feat_dict[sub].keys():
            if f_sel in ['sub', 'CDRS', 'LID', 'ACC']: continue
            # define ft-values (merge/handle bilaterality of ft-names)
            if np.logical_and('psd' in  f_sel or 'variation' in f_sel,
                              'lfp' in f_sel):
                # merge lfp left and right (once)
                if 'left' in f_sel: continue
                f_sel = f_sel.split('_right_')[1]
                temp_l = feat_dict[sub][f'lfp_left_{f_sel}'].values
                temp_r = feat_dict[sub][f'lfp_right_{f_sel}'].values
                temp_values = np.nanmean([temp_l, temp_r], axis=0)
                if f'lfp_{f_sel}' not in ftCols: ftCols.append(f'lfp_{f_sel}')
            # put ecog left vs right in same column
            elif 'ecog_left' in f_sel or 'ecog_right' in f_sel:
                temp_values = feat_dict[sub][f_sel]
                if 'left' in f_sel: f_sel = f_sel.replace('ecog_left', 'ecog')
                elif 'right' in f_sel: f_sel = f_sel.replace('ecog_right', 'ecog')
                if f_sel not in ftCols: ftCols.append(f_sel)
            else:
                temp_values = feat_dict[sub][f_sel]
                # add to columns
                if f_sel not in ftCols: ftCols.append(f_sel)

            # define rated cdrs-scores
            temp_labels = label_dict[sub].copy()
            # define acc rms
            if 'ACC' in noFtCols: temp_rms = acc_dict[sub].copy()

            # calculate and add corr for subject and feature
            if STAT_METH == 'pearson':
                # check and correct for NaNs (for LMM do this after df created)
                nan_sel = np.isnan(temp_values)
                if nan_sel.any():
                    temp_values = temp_values[~nan_sel]
                    temp_labels = temp_labels[~nan_sel]
                    assert len(temp_labels) == len(temp_values), 'X and Y lengths unequal'
                # calculate statistics
                R, p = pearsonr(temp_values, temp_labels)
                corrs[f_sel].append(R)

            elif STAT_METH == 'lmm':
                # expand arr rows with NaNs to fill for sub at first feature
                if i_s == 0 and i_f == 0:  # create array for first sub
                    sub_row0 = 0
                    sub_rowX = sub_row0 + len(temp_labels)
                    lm_arr = np.array([[np.nan] * (len(noFtCols)+1)] * len(temp_labels))
                    # print(f'{sub}, shape arr after nan adding: {lm_arr.shape}'
                    #         f', sub-rows: {sub_row0} : {sub_rowX}; n: {len(temp_labels)}//{len(temp_values)}')
                elif i_s == 0:  # add column per ft during first subject
                    lm_arr = np.concatenate([lm_arr, np.atleast_2d([np.nan] * lm_arr.shape[0]).T],
                                            axis=1)

                elif i_f == 0:  # for other subjects (add to existing array)
                    sub_row0 = lm_arr.shape[0]
                    sub_rowX = sub_row0 + len(temp_labels)
                    lm_arr = np.concatenate([lm_arr, [[np.nan] * lm_arr.shape[1]] * len(temp_labels)],
                                            axis=0)
                    # print(f'{sub}, shape arr after nan adding: {lm_arr.shape}'
                    #         f', sub-rows: {sub_row0} : {sub_rowX}; n: {len(temp_labels)}//{len(temp_values)}')
                # add sub- and label-rows at first feature
                if i_f == 0:
                    # in order of ['sub', 'CDRS', 'LID'] noFtCol
                    lm_arr[sub_row0:sub_rowX, 0] = [sub] * len(temp_labels)
                    lm_arr[sub_row0:sub_rowX, 1] = temp_labels
                    lm_arr[sub_row0:sub_rowX, 2] = (temp_labels > 0).astype(int)
                    if 'ACC' in noFtCols: lm_arr[sub_row0:sub_rowX, 3] = temp_rms

                # add feature values
                lm_arr[sub_row0:sub_rowX, len(noFtCols) + i_f] = temp_values
                # increase counter
                i_f += 1

    if STAT_METH == 'pearson': return corrs

    if STAT_METH == 'lmm':

        corrs = {}
        # check and correct for NaNs (for LMM do this after df created)
        nan_sel = np.isnan(lm_arr).any(axis=1)  # True if row contains NaN
        if nan_sel.any():
            print(f'{sum(nan_sel)} rows deleted bcs of NaNs present')
            lm_arr = lm_arr[~nan_sel]
        # create df for lmm
        lm_df = pd.DataFrame(data=lm_arr, columns=noFtCols+ftCols)

        # transform gamma columns
        lm_df = replace_gammas_for_maxGamma(lm_df)

        # normalize values
        # if TARGET == 'CDRS': NORM_CDRS = True
        # else: NORM_CDRS = False

        lm_df = normalize_feats_scores(lm_df, NORM_CDRS=NORM_CDRS,
                                       NORM_METH='std')
        
        if TARGET == 'CDRS' and np.max(lm_df['CDRS']) > 10:
            print('\n\nCDRS ZSCORED')
            lm_df['CDRS'] = lm_df['CDRS'] / np.std(lm_df['CDRS'])


        for ft in list(lm_df.keys()):
            
            if ft in ['sub', 'CDRS', 'LID', 'ACC']: continue

            # statsmodels
            model = mixedlm(f"{TARGET} ~ {ft}", lm_df,
                            groups=lm_df["sub"])
            result = model.fit(method='lbfgs')

            R, p = get_R_smResults(result=result, X=lm_df[ft],
                                       y=lm_df[TARGET], ft=ft)
            corrs[ft] = (R, p)
        
        if RETURN_STAT_DF: return corrs, lm_df
        else: return corrs