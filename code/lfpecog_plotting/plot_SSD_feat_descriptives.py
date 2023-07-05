"""
Plot SSD features in different states
"""

# import public functions
from os.path import join
import numpy as np
import pandas as pd
from itertools import product

from scipy.stats import (
    mannwhitneyu, norm, pearsonr, spearmanr
)
import gpboost as gpb
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

import matplotlib.pyplot as plt
import seaborn as sns
from lfpecog_plotting.plotHelpers import get_colors

def plot_binary_featViolins(
    X_all, y_all, sub_ids, ft_names,
    incl_ft_sources,
    sign_test='glmm', ALPHA = .05,
    SHOW_PLOT=True, SAVE_PLOT=False,
    fig_name=None, fig_path=None,
):
    # select features based on source
    ecog_sel = np.array(['ecog' in f.lower() for f in ft_names])
    if incl_ft_sources == 'STN': sel = ~ecog_sel
    elif incl_ft_sources == 'ECOG': sel = ecog_sel
    elif incl_ft_sources == 'ALL': sel = np.array([True] * len(ecog_sel))

    X_all = X_all[:, sel]
    ft_names = np.array(ft_names)[sel]

    # Create violinplot LID vs no-LID
    violin_df, violin_ps = get_violin_ft_data(
        X_all, y_all, ft_names, sub_ids=sub_ids,
        sign_test=sign_test, binary_LID=True,)

    fsize=24
    # correct alpha for mult. comparisons
    ALPHA /= len(ft_names)

    fig, ax = plt.subplots(1, 1, figsize=(24, 12))

    sns.set_theme(style="whitegrid")
    clrs = list(get_colors().values())

    # Draw a nested violinplot and split the violins for easier comparison
    violin = sns.violinplot(data=violin_df,
                x="feature", y="values",
                hue="lid",
                split=True,
                inner="quartile",
                linewidth=1,
                ax=ax,
                palette={0: clrs[4], 1: clrs[1]})

    # change transparency based on sign differences
    double_ps = []  # violin.collections desribes every half-violin, therefore double the p-values
    for p in violin_ps: double_ps.extend([p, p])

    for body, p_ft in zip(violin.collections, double_ps):
        
        if p_ft < ALPHA: body.set_alpha(1)
        else: body.set_alpha(.3)

    # set properties quartile lines
    for l in violin.lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('k')
        l.set_alpha(0.8)
    for l in violin.lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(2)
        l.set_color('black')
        l.set_alpha(0.8)

    sns.despine(left=True)

    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(len(ft_names)))
    ft_names = readable_ftnames(ft_names)
    ax.set_xticklabels([n for n in ft_names], rotation=60, ha='right',
                    size=fsize)
    for i_tick, p in enumerate(violin_ps):
        if p < ALPHA: ax.get_xticklabels()[i_tick].set_weight('bold')
    ax.set_ylabel('z-scored values (a.u.)', size=fsize + 8)
    ax.set_xlabel('')
    plt.tick_params(axis='both', size=fsize, labelsize=fsize+2)

    h, l = ax.get_legend_handles_labels()
    l = ['no LID', 'LID']
    ax.legend(h, l, fontsize=fsize+8, frameon=False,
            ncol=2, loc='lower right')
    ax.set_title(f'{incl_ft_sources} Feature differences '
                 '(10-s windows): without-LID versus with-LID'
                f'   (Bonf. corrected alpha = {round(ALPHA, 4)})',
                size=fsize + 8, weight='bold')

    plt.tight_layout()

    if SAVE_PLOT:
        plt.savefig(join(fig_path, fig_name), facecolor='w', dpi=300,)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close()


def plot_feats_on_categLID(
    X_all, y_all, sub_ids, ft_names,
    incl_ft_sources,
    sign_test='glmm', ALPHA = .05,
    SHOW_FT_BOXPLOT=True, SAVE_FT_BOXPLOT=False,
    SHOW_CORR_BARPLOT=True, SAVE_CORR_BARPLOT=False,
    fig_path=None, fig_name_ftBox=None,
    fig_name_corrBar=None,
):
    # select features based on source
    ecog_sel = np.array(['ecog' in f.lower() for f in ft_names])
    if incl_ft_sources == 'STN': sel = ~ecog_sel
    elif incl_ft_sources == 'ECOG': sel = ecog_sel
    elif incl_ft_sources == 'ALL': sel = np.array([True] * len(ecog_sel))

    X_all = X_all[:, sel]
    ft_names = np.array(ft_names)[sel]

    # Create violinplot LID vs no-LID
    box_lists, box_ps = get_violin_ft_data(
        X_all, y_all, ft_names, sub_ids=sub_ids,
        sign_test=sign_test, binary_LID=False,)
    box_ps = np.array(box_ps)
    # correct alpha for mult. comparisons
    ALPHA /= len(ft_names)

    fsize=24

    if SAVE_FT_BOXPLOT or SHOW_FT_BOXPLOT:
        fig, ax = plt.subplots(1, 1, figsize=(24, 12))

        color_names = ['lightblue', 'nightblue', 'sand','turquoise', ]
        clrs = [get_colors()[c] for c in color_names] * len(ft_names)

        # Draw a nested violinplot and split the violins for easier comparison
        box = ax.boxplot(x=box_lists,
                        positions=[x + shift for x, shift in 
                                    product(np.arange(len(ft_names)), [-.3, -.1, .1, .3])],
                        widths=.1,
                        patch_artist=True,)

        for i_box, (patch, color) in enumerate(zip(box['boxes'], clrs)):
            # set right color for box-body
            patch.set_facecolor(color)
            # if p-value is LARGER than ALPHA (not sign), reduce color-alpha
            if box_ps[int(i_box / 4), 1] > ALPHA:
                patch.set_alpha(.4)

        ax.set_ylim(-5, 5)
        ax.set_xticks(np.arange(len(ft_names)))
        ft_names = readable_ftnames(ft_names)
        ft_names = [f'{f} ({round(R, 2)})' for f, R in
                    zip(ft_names, box_ps[:, 0])]
        ax.set_xticklabels([n for n in ft_names], rotation=60, ha='right',
                        size=fsize)
        
        for i_tick, p in enumerate(box_ps[:, 1]):
            if p < ALPHA: ax.get_xticklabels()[i_tick].set_weight('bold')
        
        ax.set_ylabel('z-scored values (a.u.)', size=fsize + 8)
        ax.set_xlabel('')
        plt.tick_params(axis='both', size=fsize, labelsize=fsize+2)

        # h, l = ax.get_legend_handles_labels()
        # l = ['no LID', 'LID']
        # ax.legend(h, l, fontsize=fsize+8, frameon=False,
        #         ncol=2, loc='lower right')
        
        ax.set_title(f'{incl_ft_sources} Feature differences '
                    '(10-s windows): categorical-LID'
                    f'   (Bonf. corrected alpha = {round(ALPHA, 4)})',
                    size=fsize + 8, weight='bold')

        plt.tight_layout()


    if SAVE_FT_BOXPLOT:
        plt.savefig(join(fig_path, fig_name_ftBox),
                    facecolor='w', dpi=300,)

    if SHOW_FT_BOXPLOT:
        plt.show()
    else:
        plt.close()



def get_violin_ft_data(
    X_all, y_all, ft_names,
    binary_LID = True,
    sign_test: str = 'GLMM', sub_ids=None,   
):
    """
    create lists for violinplots, include
    significance testing p-values
    """
    assert sign_test.upper() in ['GLMM', 'MWU', 'MIXEDLM'], (
        'sign_test hs to be GLMM (gen linear mixed effects model)'
        f', or MWU (mann-whitney-u), or MIXEDLM (statsmodels), '
        f'{sign_test} was given'
    )
    
    if binary_LID:

        violin_data = pd.DataFrame(columns=['feature', 'values', 'lid'])

        violin_values = [X_all[:, i_ft] for i_ft in np.arange(X_all.shape[1])]
        violin_data['values'] = np.array(violin_values).ravel()
        
        violin_names = [[f] * X_all.shape[0] for f in ft_names]
        violin_data['feature'] = np.array(violin_names).ravel()

        y_all_binary = (y_all > 0).astype(int)
        violin_y = [[y_all_binary] * X_all.shape[1]]
        violin_y = np.array(violin_y).ravel()
        violin_data['lid'] = violin_y > 0

        if sign_test.upper() == 'MWU':
            violin_ps = [mannwhitneyu(X_all[:, i_f][y_all_binary == 0],
                                    X_all[:, i_f][y_all_binary > 0])[1]
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

        elif sign_test.upper() == 'MIXEDLM':
            violin_ps = []

            for i_ft, ft in enumerate(ft_names):

                data_df = pd.DataFrame({
                    'Feature': X_all[:, i_ft],
                    'LID': y_all_binary,
                    'Sub': sub_ids
                })
                model = mixedlm("Feature ~ LID", data_df,
                                groups=data_df["Sub"])
                result = model.fit()
                
                # print(result.summary())
                
                # Extract the p-value for the Condition variable
                p_value = result.pvalues['LID']
                violin_ps.append(p_value)

    elif not binary_LID:
        violin_data = []
        violin_ps = []

        for i_ft in np.arange(X_all.shape[1]):
            # loop over all features
            ft_values = X_all[:, i_ft]

            for score in [0, 1, 2, 3]:
                # select ft-value belonging to LID-score (multiclass)
                score_values = list(ft_values[y_all == score])
                violin_data.append(score_values)
            
            # define correlation between LID-categories and feature-values
            R, p = pearsonr(x=ft_values, y=y_all)
            violin_ps.append([R, p])
            
        
    return violin_data, violin_ps


def readable_ftnames(ft_names):
    temp_ftnames = []
    for f in ft_names:
        if 'right' in f: f=f.replace('right', 'R')
        elif 'left' in f: f=f.replace('left', 'L')

        if 'narrow_gamma' in f: f=f.replace('narrow_gamma', 'gamma')
        
        if 'mean_burst' in f: f=f.replace('mean_burst', 'burst')

        temp_ftnames.append(f.upper())

    return temp_ftnames