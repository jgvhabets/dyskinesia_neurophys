"""
calculate stats for psd comparisons
"""

# import libraries
import numpy as np
import pandas as pd
import json
import os
from itertools import compress
# from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt
# import own functions
from utils.utils_fileManagement import (get_project_path,
                                        make_object_jsonable)

# import lfpecog_analysis.process_connectivity as processConn
from statsmodels.regression.mixed_linear_model import MixedLM

### new stat functions (Jan 24)

def calc_lmem_freqCoeffs(temp_values,
                         temp_ids,
                         temp_freqs,
                         ALPHA: float = .05,
                         VALUES_GIVEN_GROUPED=False,
                         GROUP_LABELS=None,
                         STATS_PER_LID_CAT: bool = False,
                         LID_REST_BASELINE: bool = False,
                         STATS_VERSION='',
                         verbose: bool = False,):
    """
    temp_values:
    temp_ids: array corresponding to values
    temp_freqs: array corr to values

    LID_REST_BASELINE: alternative baselining for TAP
        and INVOLUNT in which baseline is the LID-category
        without movement (in rest)

    Returns:
        - dynamic output_list: list depending
            on returning on GRAD and CI
            if STATS_PER_LID_CAT: dict's are given
            with keys: midl/moderate/severe
        - freqs: corr to results
    """
    # select freqs 4 - 35 Hz and 60 - 90 Hz
    f_sel = np.logical_or(
        [f >= 4 and f <= 35 for f in temp_freqs],
        [f >= 60 and f <= 90 for f in temp_freqs]
    )

    # correct alpha for multiple comparisons
    ALPHA = ALPHA / sum(f_sel)
    if STATS_VERSION == '2Hz':
        ALPHA = ALPHA * 2  # half of the freqs-length due to 2-Hz blocks
    RETURN_GRAD, RETURN_CI = False, False

    if VALUES_GIVEN_GROUPED and any(GROUP_LABELS != None):
        stat_values = temp_values
        stat_labels=GROUP_LABELS.astype(int)
        stat_ids = temp_ids

    elif not VALUES_GIVEN_GROUPED:
        print('data not given grouped, execute get_stats_arrays()')
        # organize long arrays with all values, labels and sub-ids
        (stat_values,
         stat_labels,
         stat_ids) = get_stats_arrays(ipsivalues=temp_values,
                                      ipsi_ids=temp_ids)

    # objects to store data in
    if not STATS_PER_LID_CAT:
        
        coeffs_freqs, sign_freqs, grads = [], [], []
    
    elif STATS_PER_LID_CAT:
        
        if np.nanmax(GROUP_LABELS) <= 3:
            CAT_CODING = {1: 'mild', 2: 'moderate', 3: 'severe'}
        
        elif np.nanmax(GROUP_LABELS) == 4:
            CAT_CODING = {1: 'no', 2: 'mild', 3: 'moderate', 4: 'severe'}
        
        elif np.nanmax(GROUP_LABELS) > 4 and LID_REST_BASELINE:
            CAT_CODING = {0: 'no', 1: 'mild', 2: 'moderate', 3: 'severe'}  # LIDREST baseline is coded +10
        
        coeffs_freqs = {c: [] for c in CAT_CODING.values()}
        sign_freqs = {c: [] for c in CAT_CODING.values()}
    
    # calculate coeffs and pvalues per frequency bin
    res_freqs = []  # to prevent double calculating if freq-bin is larger than 1

    for i_f, f in enumerate(temp_freqs):
        # skip irrelevant freqs
        if not f in temp_freqs[f_sel]:
            if verbose: print(f'...skip freq {f} Hz in STATs')
            continue
        if f in res_freqs:
            print(f'...skip {f} Hz (already done before)')
            continue
        
        if not LID_REST_BASELINE:
            assert sum(stat_labels == 0) > 0, f'NO BASELINE found in LMM calc ({f} Hz)'

        ### calculate each category separately
        res_freqs.append(f)
        if STATS_PER_LID_CAT: 
            # loop over LID-categories, always same baseline (coded as 0)
            for CAT in CAT_CODING:  # corr to mild-moderate-severe (0: NO-LID is skipped)
                if verbose: print(f'...start STAT calc for LID CAT: {CAT}, {CAT_CODING[CAT]} ({f} Hz)')
                
                if not LID_REST_BASELINE:
                    cat_sel = np.logical_or(stat_labels == 0,
                                            stat_labels == CAT)
                    cat_labels = stat_labels[cat_sel]
                    cat_values = stat_values[cat_sel, i_f]
                    cat_ids = stat_ids[cat_sel]
                
                # overwrite in alternative case of LID-rest category baselines
                elif LID_REST_BASELINE:
                    cat_sel = np.logical_or(stat_labels == (CAT + 10),  # baseline coded +10
                                            stat_labels == CAT)
                    temp_labels = stat_labels[cat_sel]
                    # set labels always baseline: 0 and CAT: 1
                    cat_labels = np.zeros_like(temp_labels)
                    cat_labels[temp_labels == CAT] = 1
                    cat_values = stat_values[cat_sel, i_f]
                    cat_ids = stat_ids[cat_sel]
                    assert sum(cat_labels == 0) > 0, f'NO (alt, LID-categ) BASELINE found in LMM calc ({f} Hz)'
                    if sum(cat_labels == 1) == 0:
                        print(f'NO VALUES in LMM (LID cat: {CAT}, {f} Hz), DELETED CAT STORING LIST')
                        if CAT_CODING[CAT] in coeffs_freqs.keys():
                            del coeffs_freqs[CAT_CODING[CAT]]
                            del sign_freqs[CAT_CODING[CAT]]
                        continue

                # take mean of both freqs (except for last)
                if STATS_VERSION == '2Hz' and i_f + 1 != stat_values.shape[1]:
                    cat_values = np.mean([cat_values, stat_values[cat_sel, i_f + 1]], axis=0)
                    res_freqs.append(temp_freqs[i_f + 1])
                
                # calculate difference versus 0
                try:
                    result_list = run_mixEff_wGroups(
                        dep_var=cat_values,
                        indep_var=cat_labels,
                        groups=cat_ids,
                        TO_ZSCORE=False,
                        RETURN_GRADIENT=RETURN_GRAD,
                        RETURN_CI=RETURN_CI
                    )
                except:
                    print(cat_values.shape)
                    print(cat_labels.shape)
                    print(cat_ids.shape)
                    print(sum(cat_sel), len(cat_sel))
                    raise ValueError
                # in case of unsuccessful calc 
                if isinstance(result_list, bool):
                    if result_list == False:
                        print(f'\n#### no conversion for {f} (LID cat: {CAT})\n')
                        print(f'sum {sum(cat_sel), {len(cat_sel)}}')
                        print(f'cat values: {cat_values.shape}')
                        fig, ax = plt.subplots(1,1, figsize=(4, 4))
                        ax.scatter([0] * sum(cat_labels == 0),
                                cat_values[cat_labels == 0],)
                        ax.scatter([1] * sum(cat_labels == 1),
                                cat_values[cat_labels == 1],)
                        ax.set_title(f'{f} Hz, LID category: {CAT}')
                        plt.show()
                        break
                        coeffs_freqs.append(0)
                        sign_freqs.append(False)
                        continue
                fixEff_cf, pval = result_list[:2]
                coeffs_freqs[CAT_CODING[CAT]].append(fixEff_cf)
                sign_freqs[CAT_CODING[CAT]].append(pval < ALPHA)
                # add double for 2Hz blocks
                if STATS_VERSION == '2Hz' and i_f + 1 != stat_values.shape[1]:
                    coeffs_freqs[CAT_CODING[CAT]].append(fixEff_cf)
                    sign_freqs[CAT_CODING[CAT]].append(pval < ALPHA)
                
                # check where lengths go off
                assert len(coeffs_freqs[CAT_CODING[CAT]]) == len(np.unique(res_freqs)), (
                    f'cfs-l: {len(coeffs_freqs[CAT_CODING[CAT]])}, '
                    f'res-f-l: {len(np.unique(res_freqs))}\nAFTER FREQ {f} Hz'
                )

        else:  # NONE-CATEGORICAL STATISTICS
            # calculate coeffs for med-effect on values (random slopes for subjects)
            temp_values = stat_values[:, i_f]
            # take mean of both freqs
            if STATS_VERSION == '2Hz' and i_f + 1 != stat_values.shape[1]:
                temp_values = np.mean([temp_values, stat_values[:, i_f + 1]])
                res_freqs.append(temp_freqs[i_f + 1])
            result_list = run_mixEff_wGroups(
                dep_var=temp_values,
                indep_var=stat_labels,
                groups=stat_ids,
                TO_ZSCORE=False,
                RETURN_GRADIENT=RETURN_GRAD,
                RETURN_CI=RETURN_CI
            )
            if isinstance(result_list, bool):
                if result_list == False:
                    print(f'\n#### no conversion for {f}\n')
                    fig, ax = plt.subplots(1,1, figsize=(4, 4))
                    ax.scatter([0] * sum(stat_labels == 0),
                            stat_values[stat_labels == 0, i_f],)
                    ax.scatter([1] * sum(stat_labels == 1),
                            stat_values[stat_labels == 1, i_f],)
                    ax.set_title(f'{f} Hz')
                    plt.show()
                    break
                    coeffs_freqs.append(0)
                    sign_freqs.append(False)
                    continue
            # unpack dynamic result list
            if RETURN_GRAD:
                grad = result_list[-1]
                grads.append(grad)
            if RETURN_CI and RETURN_GRAD: ci = result_list[-2]
            elif RETURN_CI and not RETURN_GRAD: ci = result_list[-1]
            fixEff_cf, pval = result_list[:2]
            # print(f'p-value {f} Hz: {pval}')
            sig_bool = pval < ALPHA
            # add values to lists
            coeffs_freqs.append(fixEff_cf)
            sign_freqs.append(sig_bool)
            # add double for 2Hz blocks
            if STATS_VERSION == '2Hz' and i_f + 1 != stat_values.shape[1]:
                coeffs_freqs.append(fixEff_cf)
                sign_freqs.append(sig_bool)

    if len(STATS_VERSION) > 1:
        # delete double 2 Hz freqs and keep only freq-selected (4-35, 60-90)
        res_freqs = np.sort(np.unique(res_freqs))
        valid_fs = np.isin(res_freqs, temp_freqs[f_sel], assume_unique=True)
        res_freqs = res_freqs[valid_fs]
        for k in coeffs_freqs.keys():
            coeffs_freqs[k] = np.array(coeffs_freqs[k])[valid_fs]
            sign_freqs[k] = np.array(sign_freqs[k])[valid_fs]
        
    if not STATS_PER_LID_CAT:
        coeffs_freqs = np.array(coeffs_freqs)
        sign_freqs = np.array(sign_freqs)


    if RETURN_GRAD: grads = np.array(grads)

    if RETURN_CI: print('fix result_list output with CI')

    if not RETURN_GRAD: return (coeffs_freqs, sign_freqs), np.array(res_freqs)

    elif RETURN_GRAD: return (coeffs_freqs, sign_freqs, grads), np.array(res_freqs)


def run_mixEff_wGroups(dep_var, indep_var,
                       groups, TO_ZSCORE=True,
                       ALPHA=.01,
                       RETURN_CI=False,
                       RETURN_GRADIENT=False,):
    """
    # tests sign effect of LID on ephys
    # Model: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html
    # Results: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html

    Returns:
        - output_list: contains fixed-effect Coeff, pvalue,
            if defined also Conf-Interv, coef-gradient
    """

    # z-score ephys values on group level for scaling
    if TO_ZSCORE:
        dep_var = (dep_var - np.std(dep_var)) / np.mean(dep_var)
    # define model
    lm_model = MixedLM(
        endog=dep_var,  # dependent variable (ephys score)
        exog=indep_var,  # independent variable (i.e., LID presence, movement)
        groups=groups,  # subjects
        exog_re=None,  # (None)  defaults to a random intercept for each group
    )
    # run and fit model
    try:
        lm_results = lm_model.fit()
    except:
        return False
    # extract results
    fixeff_cf = lm_results._results.fe_params[0]
    pval = lm_results._results.pvalues[0]

    output_list = [fixeff_cf, pval]  # to keep output number dynamic

    if RETURN_CI:
        conf_int = lm_results.conf_int(alpha=ALPHA)[0]
        output_list.append(conf_int)
    
    if RETURN_GRADIENT:
        grad = lm_results._results.params_object()
        output_list.append(grad)

    # print(f'fixed effect coeff: {fixeff_cf}')  # fixed-effect coeffs
    # print(f'Confidence Interval (alpha: {ALPHA}): {conf_int} (p = {pval.round(5)})')
    # print(lm_results.summary())

    # return two, three, or four values
    return output_list
    

def get_stats_arrays(ipsivalues, ipsi_ids):
    """
    ipsi_values: expects dict with arrays per category
    ipsi_lids: corresponding
    """
    stat_values, stat_ids = {}, {}
    for lid_code, lid_cats in zip(['lid', 'nolid'],
                                    [[1, 2, 3], [0]]):
        # select all arrays LID vs NO-LID, sorted in groups of category
        # [sub1-cat1, sub2-cat1, .., subX-cat1, sub1-cat2, sub2-cat2, ..]
        temp_values = [a for cat in ipsivalues
                        for a in ipsivalues[cat] if cat in lid_cats]
        # get corresponding list of sub-ids
        temp_ids = [s for cat in ipsi_ids for s in ipsi_ids[cat]
                        if cat in lid_cats]  # get sub-ids w/o LID
        temp_ids = [[s] * a.shape[0] for s, a in
                        zip(temp_ids, temp_values)]  # multiply ids with corr array shapes
        temp_ids = np.array([i for l in temp_ids for i in l])  # unpack list of lists    
        # convert into long arrays
        temp_values = np.array([r for a in temp_values for r in a])
        assert temp_values.shape[0] == temp_ids.shape[0], 'incorrect shapes'
        stat_values[lid_code] = temp_values
        stat_ids[lid_code] = temp_ids
    # create single arrays for stats
    stat_values = np.concatenate([stat_values['nolid'],
                                stat_values['lid']])
    stat_labels = np.concatenate([[0] * len(stat_ids['nolid']),
                                [1] * len(stat_ids['lid'])])
    stat_ids = np.concatenate([stat_ids['nolid'],
                            stat_ids['lid']])

    assert (stat_values.shape[0] == stat_labels.shape[0]
            == stat_ids.shape[0]), 'incorrect lengths'

    return stat_values, stat_labels, stat_ids



# previous stat functions (Oct 23)

def process_mean_stats(
    mean_stats, datatype,
    DATA_VERSION='v4.0', FT_VERSION='v4',
    save_stats=True,
):
    if save_stats:
        store_path = os.path.join(get_project_path('results'), 'stats',
                                  f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                                  f'{datatype}_LMM_noLID_vs_LID')
        if not os.path.exists(store_path):
            os.makedirs(store_path)
            print(f'created STATS PATH: {store_path}')
    
    for i_f, f_hz in enumerate(mean_stats['freqs']):

        df_arr = []
        for i_t, (tup1, tup0, tupSc) in enumerate(
            zip(mean_stats['LID'],
                mean_stats['noLID'],
                mean_stats['CDRS'])
        ):

            assert tup0[0] == tup1[0], ('subs dont match NO: '
                                        f'{tup0[0]} vs LID: {tup1[0]}')
            assert tup1[0] == tupSc[0], ('subs dont match LID: '
                                         f'{tup1[0]} vs CDRS: {tupSc[0]}')
            sub = tup1[0]

            p1 = tup1[1][i_f, :]
            p0 = tup0[1][i_f, :]
            sc = tupSc[1]

            if p1.shape[0] == 0: continue

            # add all powers with LID-bool and sub-match
            df_arr.extend([[sub, 0, 0, p] for p in p0])  # sub, LID-bool, CDRS, power
            df_arr.extend([[sub, 1, s, p] for p, s in zip(p1, sc)])  # sub, LID-bool, CDRS, power

        lm_data = pd.DataFrame(columns=['sub', 'LID', 'CDRS', 'mean_power'],
                               data=df_arr)
        # if save dataframe
        if save_stats:
            lm_data.to_csv(os.path.join(store_path,
                                        f'{datatype}_LMM_LID_PSD_{f_hz}Hz_df.xlsx'))
            print(f'df for {f_hz} Hz saved')


# def get_binary_p_perHz(datatype, save_date='0000',
#                        DATA_VERSION='v4.0', FT_VERSION='v4',
#                        load_data=False,
#                        save_ps=True, return_ps=False,
#                        return_full_dict=False,):
    
#     store_path = os.path.join(get_project_path('results'), 'stats',
#                               f'data_{DATA_VERSION}_ft_{FT_VERSION}',
#                               f'{datatype}_LMM_noLID_vs_LID')
#     assert os.path.exists(store_path), 'incorrect path'

#     freqs = np.arange(4, 91)

#     if load_data and (return_ps or return_full_dict):
#         try:
#             with open(os.path.join(
#                 store_path, f'{datatype}_LMM_results_'
#                 f'pvalues_{save_date}.json'
#             ), 'r') as json_file:
#                 store_json = json.load(json_file)
            
#             if return_full_dict:
#                 return store_json
#             elif return_ps:
#                 p_list = store_json['p_values']
#                 return p_list

    
#         except:
#             print('calculate p-values based on saved data')
    
#     p_list = []

#     for i_f, f_hz in enumerate(freqs):
#         print(f'START {f_hz} Hz')
#         if f_hz > 35 and f_hz < 60:
#             p_list.append(np.nan)
#             continue

#         lm_data = pd.read_csv(os.path.join(store_path,
#                                            f'{datatype}_LMM_LID_PSD_{f_hz}Hz_df.xlsx'),
#                             index_col=0, header=0, )
        
#         model = mixedlm("mean_power ~ LID", lm_data,
#                         groups=lm_data["sub"])
#         result = model.fit(method='lbfgs')

#         # Extract the p-value for the Condition variable
#         p_list.append(result.pvalues['LID'])

#     if save_ps:
#         store_json = {'p_values': p_list, 'freqs': list(freqs)}

#         store_json = make_object_jsonable(store_json)

#         with open(os.path.join(store_path,
#                                f'{datatype}_LMM_results_pvalues_{save_date}.json'),
#                   'w') as json_file:
#             json.dump(store_json, json_file)
    
#     if return_ps: return p_list
#     elif return_full_dict: return store_json