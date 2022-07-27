"""
Merge and Visualise Extracted Tapping-Features

takes features extracted over 10 second blocks
"""

# Import public packages and functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, variation
from itertools import compress
from os.path import join

# Import own fucntions
from lfpecog_features.tapping_feat_calc import aggregate_arr_fts


def combineFeatsPerScore(
    ftDict: dict, fts_include, merge_method: str,
):
    """
    Merges and compares features of different
    tapping runs, classifies based on the cor-
    responding UPDRS tapping subscore

    Input:
        - ftDict (dict): containing Classes
            (from tapping_extract_features)
            with features and info per run
        - fts_include (list): define which
            features to include in analysis
        - merge_method (str): method to
            aggregate array-based features
            per run, most be from defined list
    
    Returns:
        feat_dict_out (dict): containing one dict
            per features; with features per
            updrs tapping subscore
            
    """
    merge_method_list = [
        'allin1', 'mean', 'sum', 'stddev', 'coefVar',
        'range', 'trend_slope', 'trend_R', 'median',
    ]
    assert merge_method in merge_method_list, (
        f'merge_method: "{merge_method}" is not '
        f'in {merge_method_list}')

    feat_dict_out = {}
    for ft_sel in fts_include:

        ft_per_score = {}
        for s in np.arange(5): ft_per_score[s] = []

        for i in list(ftDict.keys()):
            
            s = ftDict[i].updrsSubScore
            tempscore = getattr(ftDict[i], ft_sel)

            if type(tempscore) != np.ndarray:  #float or np.float_
                ft_per_score[s].append(tempscore)
            
            elif type(tempscore) == np.ndarray:

                if tempscore.size == 0: continue

                if merge_method == 'allin1':
                    if np.isnan(tempscore).any():
                        tempscore[~np.isnan(tempscore)]
                    ft_per_score[s].extend(tempscore)  # all in one big list

                else:
                    ft_per_score[s].append(
                        aggregate_arr_fts(
                            method=merge_method,
                            arr=tempscore
                        )
                    )

        feat_dict_out[ft_sel] = ft_per_score

    return feat_dict_out



def clean_list_of_lists(dirty_lists):
    """
    Remove nans from a list of lists
    """
    clean_lists = []
    
    for templist in dirty_lists:
            sel = ~np.isnan(templist)
            clean_lists.append(
                list(compress(templist, sel))
            )

    return clean_lists


def plot_boxplot_feats_per_subscore(
    fts_include: list, featDict: dict,
    merge_method: str, plot_title: str='',
    figsave_name: str='', figsave_dir: str='',
    show: bool=False
):
    """
    Plots boxplots of tapping features which
    are ordered per updrs subscore priorly.
    
    Input:
        - fts_include (list): list with feature names
        - fts_per_score (list): list containing
            feature-arrays per
    """
    fig, axes = plt.subplots(
        len(fts_include), 1, figsize=(24, 16)
    )

    for row, ft_sel in enumerate(fts_include):
        fts_scores = featDict[ft_sel]
        boxdata = [fts_scores[s] for s in fts_scores.keys()]
        boxdata = clean_list_of_lists(boxdata)
        axes[row].boxplot(boxdata)

        tick_Ns = [len(l) for l in boxdata]
        xlabels = [
            f'{i}  (n={tick_Ns[i]})' for i in np.arange(0, 5)
        ]
        axes[row].set_xticklabels(
            xlabels, fontsize=18, ha='center',)
        axes[row].set_xlabel('UPDRS tapping sub score', fontsize=18)
        axes[row].set_ylabel(ft_sel, fontsize=18)

    plt.suptitle(
        f'{plot_title} (merged by {merge_method})',
        size=24, x=.5, y=.92, ha='center',
    )
    
    if figsave_name:
        
        plt.savefig(
            join(figsave_dir, figsave_name),
            dpi=150, facecolor='w',
        )
    if show: plt.show()
    if not show: plt.close()