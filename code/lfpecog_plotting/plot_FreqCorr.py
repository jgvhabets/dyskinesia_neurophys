"""
Plot Frequency-Correlation-Plots
"""

# import public functions
from os.path import join, exists
from os import makedirs
import numpy as np
from itertools import compress, product
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from scipy.stats import pearsonr

# import own functions
from lfpecog_plotting.plotHelpers import (
    get_colors, convert_ft_names
)
from lfpecog_analysis.ft_processing_helpers import (
    FeatLidClass
)
from utils.utils_fileManagement import get_project_path


def plot_heatmap_ftSettings_vs_Dyskinesia(
    settingsDict, fig_name,
):
    """
    Input:
        - settingsDict: dict with different settings
            of feature settings and dyskinesia-label
            settings used to create features, should
            contain FT_VERSION, CORR_TARGET, CATEG_CDRS,
            INCL_STN, INCL_ECOG
    """
    classDict = {}

    for setting in settingsDict:
        label = f'ft-{setting["FT_VERSION"]}'
        if setting["CORR_TARGET"] == 'LID': label += '_bin'
        elif (setting["CORR_TARGET"] == 'CDRS' and
              setting["CATEG_CDRS"]): label += '_cat'
        else: label += '_lin'
                
        # create feat label class
        classDict[label] = FeatLidClass(
            FT_VERSION=setting["FT_VERSION"],
            CORR_TARGET=setting['CORR_TARGET'],
            CATEGORICAL_CDRS=setting["CATEG_CDRS"],
            INCL_ECOG=setting["INCL_ECOG"],
            INCL_STN=setting["INCL_STN"],
            TO_CALC_CORR=True,)
    
    # plot STN-only and ECoG-required features separately
    grid_feats = list(list(classDict.values())[0].corrs.keys())

    # select out STN only feats if STN AND ECOG are included
    if np.logical_and(list(classDict.values())[0].INCL_ECOG,
                      list(classDict.values())[0].INCL_STN):
        ecog_sel = ~np.array(['lfp' in f or 'STN_STN' in f
                                for f in grid_feats])
        grid_feats = list(compress(grid_feats, ecog_sel))
        print(f'ECoG requiring feats: {grid_feats}')

    # create grid with NaNs (n-features x n-ft/label-settings)
    R_grid = np.array([[np.nan] * len(classDict)
                       ] * len(grid_feats))
    p_grid = R_grid.copy()

    for i, sett_label in enumerate(classDict):
        ftCorrs = np.array(list(classDict[sett_label].corrs.values()))

        Rs = ftCorrs[:, 0]
        ps = ftCorrs[:, 1]
        # exclude only-STN features if necessary
        if np.logical_and(list(classDict.values())[0].INCL_ECOG,
                          list(classDict.values())[0].INCL_STN):
            Rs = Rs[ecog_sel]
            ps = ps[ecog_sel]
        # add R and p to grid for specific setting column
        R_grid[:, i] = Rs
        p_grid[:, i] = ps

    # plot heatmap
    plot_ftCorr_heatMap(R_grid=R_grid, p_grid=p_grid,
                        grid_feats=grid_feats,
                        sett_labels=list(classDict.keys()),
                        fig_name=fig_name,)

def plot_ftCorr_heatMap(R_grid, p_grid, fig_name,
                        grid_feats, sett_labels,
                        FS = 24,):
    
    # define sign mask
    sig_mask = p_grid < (.05 / len(grid_feats))  # correct alpha (0.05) for mult comparisons
    # create sep sign and non-sig grids for plotting difference sig vs no-sig
    sig_R_grid, nonsig_R_grid = R_grid.copy(), R_grid.copy()
    sig_R_grid[~sig_mask] = np.nan
    nonsig_R_grid[sig_mask] = np.nan

    grid_feats = convert_ft_names(grid_feats, find='delta', replace='theta')

    none_map = ListedColormap(['none'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 12), )
    # full colored for sign Rs
    sig_map = ax.pcolormesh(sig_R_grid, vmin=-1, vmax=1,
                            cmap='RdBu_r', edgecolors='w',)
    # color with hatch stripes
    nonsig_map = ax.pcolormesh(nonsig_R_grid, vmin=-1, vmax=1,
                               cmap='RdBu_r', edgecolors='w',)
    hatch = plt.pcolor(nonsig_R_grid, vmin=-1, vmax=1,
                       hatch='//', cmap=none_map,
                       edgecolor='w', )
    # show colorbar
    cbar = fig.colorbar(sig_map, pad=.01)
    cbar.set_label('LMM - Coefficient (a.u.)', size=FS, weight='bold')
    cbar.ax.tick_params(size=FS, labelsize=FS-2,)

    ax.set_yticks(np.arange(.5, R_grid.shape[0] + .5, 1))
    ax.set_yticklabels(grid_feats)

    ax.set_xticks(np.arange(.5, R_grid.shape[1] + .5, 1))
    ax.set_xticklabels(sett_labels)

    ax.tick_params(top=True, labeltop=True,
                bottom=False, labelbottom=False,
                size=FS, labelsize=FS,)

    plt.tight_layout()

    plt.savefig(join(get_project_path('figures'),
                     'feat_dysk_corrs', fig_name),
                dpi=300, facecolor='w',)
    plt.close()

def calculate_Rs_FreqCorr(
    FreqCorr_dict, mean_per_score = True,
    PLOT_SUB_CORR = False,
):
    """
    requires import matplotlib.pylab as pl

    Inputs:
        - FreqCorr_dict: dict with PSDs and CDRS-scores
            per subject, resulting from:
            plot_PSDs.plot_STN_PSD_vs_LID(), run with
            'CALC_FREQ_CORR' set True
    """
    frqCor_values = {}
    first_sub = list(FreqCorr_dict.keys())[0]
    for state in FreqCorr_dict[first_sub].keys():
        frqCor_values[state] = {'R': [], 'p': []}
        freqs = FreqCorr_dict[first_sub][state][2]
    
    vir_n_colors = pl.cm.viridis(np.linspace(0,1,len(freqs)))

    for sub in FreqCorr_dict.keys():

        if PLOT_SUB_CORR: figsub, axessub = plt.subplots(1,2, figsize=(8, 4))
        
        for i_st, state in enumerate(frqCor_values.keys()):

            arr = FreqCorr_dict[sub][state][0].copy()
            scores = np.array(FreqCorr_dict[sub][state][1]).copy()
            freqs = np.array(FreqCorr_dict[sub][state][2]).copy()

            blank_sel = np.logical_or(freqs <= 35, freqs >= 60)
            freqs = freqs[blank_sel]
            arr = arr[blank_sel, :]

            pres_scores = np.unique(scores)
            means_per_f = {}
            curr_Rs, curr_ps = [], []

            # loop over freq-bins
            for i_f, f in enumerate(freqs):
                means_per_f[f] = {}
                # loop over present scores
                for s in pres_scores:
                    # calculate mean freq-bin-PSD for score
                    score_sel = scores == s
                    if mean_per_score:
                        means_per_f[f][s] = np.nanmean(arr[i_f, score_sel])
                if mean_per_score:
                    temp_scores = list(means_per_f[f].keys())
                    temp_values = list(means_per_f[f].values())
                if not mean_per_score:
                    temp_scores = scores
                    temp_values = arr[i_f, :]
                # calculate CORRELATION BETWEEN SCORE AND MEAN PSDs
                # for every FREQ-BIN separately
                R, p = pearsonr(x=temp_values, y=temp_scores)
                curr_Rs.append(R)
                curr_ps.append(p)


                # plot freq-lines
                if PLOT_SUB_CORR: 
                    if f in [5, 15, 20, 30, 60, 75, 90]:
                        axessub[i_st].plot(temp_scores, temp_values,
                                        label=f'{f} Hz',
                                c=vir_n_colors[i_f], alpha=.5)
                    else:
                        axessub[i_st].plot(temp_scores, temp_values,
                            c=vir_n_colors[i_f], alpha=.5)

            frqCor_values[state]['R'].append(curr_Rs)
            frqCor_values[state]['p'].append(curr_ps)

            if PLOT_SUB_CORR: 
                axessub[i_st].set_title(f'{sub}: {state}')
                axessub[i_st].legend()
        if PLOT_SUB_CORR: plt.show()
        else: plt.close()
            
    return frqCor_values, freqs


def plot_FreqCorr(frqCor_values, freqs,
                  fig_name=None, save_dir=None,
                  use_exact_values=False,):
    """
    did not follow up on plots because single
    frequencies over time do not seem to make
    sense after SSD
    """

    if not exists(save_dir): makedirs(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                            #  sharey='row',
                            )
    fsize=18
    # colors = list(get_colors().values())
    colors = get_colors('Jacoba_107')
    bar_clrs = [colors[1], colors[-4]]
    fill_params = [{'hatch': '//',
                    'facecolor': 'w',
                    'edgecolor': bar_clrs[0],
                    'alpha': .5},
                   {'facecolor': bar_clrs[1],
                    'alpha': .5,}]
    leg_labs = {'bi_match': 'Contralateral STN',
                'bi_nonmatch': 'Ipsilateral STN',
                'STN_match': 'Contralateral STN',
                'ECOG_match': 'Contralateral ECoG',}
    bar_ws = [-.5, .5]  # minus aligns to the left

    for i_ax, ax in enumerate(axes):
        for i, state in enumerate(frqCor_values.keys()):
            # use exact numbers
            if use_exact_values:
                # plot bars with mean correlations
                print(f'{state} color: {bar_clrs[i]}')
                sign = frqCor_values[state]['p'].values < (.05 / len(freqs))
                axes[i_ax].bar(np.array(freqs)[sign],
                               frqCor_values[state]['R'][sign],
                                width=bar_ws[i], align='edge',
                                label=leg_labs[state],
                                color=bar_clrs[i], alpha=.8,)
                axes[i_ax].bar(np.array(freqs)[~sign],
                               frqCor_values[state]['R'][~sign],
                                width=bar_ws[i], align='edge',
                                # label=f'{leg_labs[state]} non-sign',
                                color=bar_clrs[i], alpha=.4,)
                # plot legend
                if i == 2:
                    axes[1].legend()
                
                
                if 'R_stderr' in frqCor_values[state].keys():
                    axes[i_ax].fill_between(freqs,
                                            y1=frqCor_values[state]['R'] - frqCor_values[state]['R_stderr'],
                                            y2=frqCor_values[state]['R'] + frqCor_values[state]['R_stderr'],
                                        **fill_params[i])
                
                # axes[i_ax].set_ylim(-.01, .01)
                axes[0].set_yticks(np.arange(-.01, .011, .005))

            # calculate mean correlations per freq-bin
            else:
                R_arr = np.array(frqCor_values[state]['R'])
                p_arr = np.array(frqCor_values[state]['p'])
                mean_Rs = R_arr.mean(axis=0)
                mean_ps = p_arr.mean(axis=0)
                # plot bars with mean correlations
                axes[i_ax].bar(freqs, mean_Rs,
                            width=bar_ws[i], align='edge',
                            label=leg_labs[state],
                            color=bar_clrs[i], alpha=.8,)
                # calculate variance lines (std-err-mean)
                up_var = mean_Rs + (np.std(R_arr, axis=0) / np.sqrt(R_arr.shape[0]))
                low_var = mean_Rs - (np.std(R_arr, axis=0) / np.sqrt(R_arr.shape[0]))
                axes[i_ax].fill_between(freqs, y1=low_var, y2=up_var,
                                        **fill_params[i])
        if np.nanmax(frqCor_values[state]['R']) > .1:
            axes[i_ax].set_ylim(-1, 1)
            axes[0].set_yticks(np.arange(-1, 1.01, .5))

        axes[i_ax].spines['top'].set_visible(False)
        axes[i_ax].spines['right'].set_visible(False)
        axes[i_ax].tick_params(size=fsize, labelsize=fsize, axis='both',)
        axes[i_ax].axhline(y=0, color='darkgray',)
        for y in [-.5, .5]:
             if np.nanmax(frqCor_values[state]['R']) < .1: y /= 100
             axes[i_ax].axhline(y=y, alpha=.5, lw=1,
                                color='darkgray',)
        for y in [-.25, -.75, .25, .75]:
             if np.nanmax(frqCor_values[state]['R']) < .1: y /= 100
             axes[i_ax].axhline(y=y, alpha=.5, ls='--',
                                lw=1, color='darkgray',)

    axes[0].set_xlim(4, 35)
    
    axes[1].set_xlim(60, 90)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_yticks([])

    axes[0].set_xlabel('Frequency (Hz)', size=fsize,)
    axes[0].set_ylabel('Correlation Coeff. (a.u.)\n(from LMEM)', size=fsize,)

    # # split legend over axes
    hnd, lab = axes[1].get_legend_handles_labels()

    axes[0].legend([hnd[0]], [lab[0]], fontsize=fsize-2, ncol=1, frameon=False,
            loc='center left', bbox_to_anchor=(.1, 1.01))
    axes[1].legend([hnd[1]], [lab[1]], fontsize=fsize-2, ncol=1, frameon=False,
            loc='center left', bbox_to_anchor=(.1, 1.01))

    plt.suptitle('Effect of spectral-band-changes on Clinical Dyskinesia Rating Scores',
                fontsize=fsize, x=.5, y=.95, ha='center',
                weight='bold',)
    plt.tight_layout(h_pad=0)

    plt.savefig(join(save_dir, fig_name), dpi=150,
                facecolor='w',)

    plt.close()