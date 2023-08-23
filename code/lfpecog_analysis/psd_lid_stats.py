"""
calculate stats for psd comparisons
"""

# import libraries
import numpy as np
import pandas as pd
import json
import os
from statsmodels.formula.api import mixedlm

# import own functions
from utils.utils_fileManagement import (get_project_path,
                                        make_object_jsonable)




def process_mean_stats(
    mean_stats, datatype, save_stats=True,
):
    if save_stats:
        store_path = os.path.join(get_project_path('results'),
                                  'stats', f'{datatype}_LMM_noLID_vs_LID')
        assert os.path.exists(store_path), 'incorrect path'
    
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


def get_binary_p_perHz(datatype, save_date='0000', load_data=False,
                       save_ps=True, return_ps=False,
                       return_full_dict=False,):
    
    store_path = os.path.join(get_project_path('results'),
                              'stats', f'{datatype}_LMM_noLID_vs_LID')
    assert os.path.exists(store_path), 'incorrect path'

    freqs = np.arange(4, 91)

    if load_data and (return_ps or return_full_dict):
        try:
            with open(os.path.join(
                store_path, f'{datatype}_LMM_results_'
                f'pvalues_{save_date}.json'
            ), 'r') as json_file:
                store_json = json.load(json_file)
            
            if return_full_dict:
                return store_json
            elif return_ps:
                p_list = store_json['p_values']
                return p_list

    
        except:
            print('calculate p-values based on saved data')
    
    p_list = []

    for i_f, f_hz in enumerate(freqs):
        print(f'START {f_hz} Hz')
        if f_hz > 35 and f_hz < 60:
            p_list.append(np.nan)
            continue

        lm_data = pd.read_csv(os.path.join(store_path,
                                           f'{datatype}_LMM_LID_PSD_{f_hz}Hz_df.xlsx'),
                            index_col=0, header=0, )
        
        model = mixedlm("mean_power ~ LID", lm_data,
                        groups=lm_data["sub"])
        result = model.fit(method='lbfgs')

        # Extract the p-value for the Condition variable
        p_list.append(result.pvalues['LID'])

    if save_ps:
        store_json = {'p_values': p_list, 'freqs': list(freqs)}

        store_json = make_object_jsonable(store_json)

        with open(os.path.join(store_path,
                               f'{datatype}_LMM_results_pvalues_{save_date}.json'),
                  'w') as json_file:
            json.dump(store_json, json_file)
    
    if return_ps: return p_list
    elif return_full_dict: return store_json