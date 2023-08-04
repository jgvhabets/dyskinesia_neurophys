"""
Plot Frequency-Correlation-Plots
"""

# import public functions
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.stats import pearsonr

# import own functions
from lfpecog_plotting.plotHelpers import get_colors


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
    FreqCorrs_results = {}
    first_sub = list(FreqCorr_dict.keys())[0]
    for state in FreqCorr_dict[first_sub].keys():
        FreqCorrs_results[state] = {'R': [], 'p': []}
        freqs = FreqCorr_dict[first_sub][state][2]
    
    vir_n_colors = pl.cm.viridis(np.linspace(0,1,len(freqs)))

    for sub in FreqCorr_dict.keys():

        if PLOT_SUB_CORR: figsub, axessub = plt.subplots(1,2, figsize=(8, 4))
        
        for i_st, state in enumerate(FreqCorrs_results.keys()):

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

            FreqCorrs_results[state]['R'].append(curr_Rs)
            FreqCorrs_results[state]['p'].append(curr_ps)

            if PLOT_SUB_CORR: 
                axessub[i_st].set_title(f'{sub}: {state}')
                axessub[i_st].legend()
        if PLOT_SUB_CORR: plt.show()
        else: plt.close()
            
    return FreqCorrs_results, freqs


def plot_FreqCorr(
        FreqCorrs_results, freqs,
        fig_name=None, save_dir=None,
):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                            #  sharey='row',
                            )
    fsize=18
    colors = list(get_colors().values())
    bar_clrs = [colors[1], colors[-4]]
    fill_params = {'bi_match': {'hatch': '//',
                                'facecolor': 'w',
                                'edgecolor': bar_clrs[0],
                                'alpha': .5},
                'bi_nonmatch': {'facecolor': bar_clrs[1],
                                'alpha': .5,}}
    leg_labs = {'bi_match': 'Contralateral STN',
                'bi_nonmatch': 'Ipsilateral STN'}
    bar_ws = [-.5, .5]  # minus aligns to the left

    for i_ax, ax in enumerate(axes):
        for i, state in enumerate(FreqCorrs_results.keys()):
            # calculate mean correlations per freq-bin
            R_arr = np.array(FreqCorrs_results[state]['R'])
            p_arr = np.array(FreqCorrs_results[state]['p'])
            mean_Rs = R_arr.mean(axis=0)
            mean_ps = p_arr.mean(axis=0)
            # calculate variance lines (std-err-mean)
            up_var = mean_Rs + (np.std(R_arr, axis=0) / np.sqrt(R_arr.shape[0]))
            low_var = mean_Rs - (np.std(R_arr, axis=0) / np.sqrt(R_arr.shape[0]))
            # plot bars with mean correlations
            axes[i_ax].bar(freqs, mean_Rs,
                           width=bar_ws[i], align='edge',
                           label=leg_labs[state],
                           color=bar_clrs[i], alpha=.8,)
            # plot percentile lines
            axes[i_ax].fill_between(freqs, y1=low_var, y2=up_var,
                                    **fill_params[state])

        axes[i_ax].spines['top'].set_visible(False)
        axes[i_ax].spines['right'].set_visible(False)
        axes[i_ax].tick_params(size=fsize, labelsize=fsize, axis='both',)
        axes[i_ax].axhline(y=0, color='darkgray',)
        for y in [-.5, .5]:
             axes[i_ax].axhline(y=y, alpha=.5, lw=.1,
                                color='darkgray',)
        for y in [-.25, -.75, .25, .75]:
             axes[i_ax].axhline(y=y, alpha=.5, ls='--',
                                lw=1, color='darkgray',)
        axes[i_ax].set_ylim(-1, 1)

        if i_ax == 0:
            axes[i_ax].set_xlim(4, 35)
            axes[i_ax].set_yticks(np.arange(-1, 1.01, .5))
        elif i_ax == 1:
            axes[i_ax].set_xlim(60, 90)
            axes[i_ax].spines['left'].set_visible(False)
            axes[i_ax].set_yticks([])

    axes[0].set_xlabel('Frequency (Hz)', size=fsize,)
    axes[0].set_ylabel('Correlation Coeff.\n(Pearson R)', size=fsize,)
    # split legend over axes
    hnd, lab = axes[1].get_legend_handles_labels()
    axes[0].legend([hnd[0]], [lab[0]], fontsize=fsize-2, ncol=1, frameon=False,
            loc='center left', bbox_to_anchor=(.1, 1.01))
    axes[1].legend([hnd[1]], [lab[1]], fontsize=fsize-2, ncol=1, frameon=False,
            loc='center left', bbox_to_anchor=(.1, 1.01))

    plt.suptitle('STN-powers vs Clinical Dyskinesia Rating Scores'
                '\nduring bilateral dyskinesia',
                fontsize=fsize, x=.5, y=.95, ha='center',
                weight='bold',)
    plt.tight_layout(h_pad=0)

    plt.savefig(join(save_dir, fig_name), dpi=150,
                facecolor='w',)

    plt.close()