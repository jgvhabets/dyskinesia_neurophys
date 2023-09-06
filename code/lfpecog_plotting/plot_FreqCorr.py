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
                print(f'color: {bar_clrs[i]}')
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
    print(hnd)
    print(lab)
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