"""
Plot SSD features in different states
"""

# import public functions
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
import matplotlib.pyplot as plt
import gpboost as gpb




def get_violin_ft_data(
    X_all, y_all_binary, ft_names,
    sign_test: str = 'GLMM', sub_ids=None,   
):
    """
    create lists for violinplots, include
    significance testing p-values
    """
    assert sign_test.upper() in ['GLMM', 'MWU'], (
        'sign_test hs to be GLMM (gen linear mixed effects model)'
        f', or MWU (mann-whitney-u), {sign_test} was given'
    )
    violin_df = pd.DataFrame(columns=['feature', 'values', 'lid'])

    violin_values = [X_all[:, i_ft] for i_ft in np.arange(X_all.shape[1])]
    violin_values = np.array(violin_values).ravel()

    violin_names = [[f] * X_all.shape[0] for f in ft_names]
    violin_names = np.array(violin_names).ravel()

    violin_y = [y_all_binary for i_ft in np.arange(X_all.shape[1])]
    violin_y = np.array(violin_y).ravel()

    violin_df['feature'] = violin_names
    violin_df['values'] = violin_values
    violin_df['lid'] = violin_y

    if sign_test.upper() == 'MWU':
        violin_ps = [mannwhitneyu(X_all[:, i_f][y_all_binary == 0],
                                        X_all[:, i_f][y_all_binary == 1])[1]
                    for i_f in np.arange(X_all.shape[1])]
    
    elif sign_test.upper() == 'GLMM':
        stats_cols = ['coeff', 'p']
        lmm_results = pd.DataFrame(
            data=np.zeros((len(ft_names), len(stats_cols))),
            columns=stats_cols, index=ft_names
        )

        for i_ft, ft_name in enumerate(ft_names):
            gp_model = gpb.GPModel(
                group_data=sub_ids,
                likelihood="binary",
            )
            gp_model.fit(y=y_all_binary, X=X_all[:, i_ft],
                        params={'std_dev': True})
            coefs = gp_model.get_coef().values.ravel()
            z_values = coefs[0] / coefs[1]
            p_values = 2 * norm.cdf(-np.abs(z_values))            
            lmm_results.at[ft_name, 'coeff'] = coefs[0]
            lmm_results.at[ft_name, 'p'] = p_values

        violin_ps = lmm_results['p'].values

    return violin_df, violin_ps

def readable_ftnames(ft_names):
    temp_ftnames = []
    for f in ft_names:
        if 'ecog' in f: f=f.replace('ecog_right', 'ecog')
        elif 'lfp' in f: f=f.replace('lfp_right', 'lfp')
        if 'narrow_gamma' in f: f=f.replace('narrow_gamma', 'gamma')
        temp_ftnames.append(f.upper())

    return temp_ftnames