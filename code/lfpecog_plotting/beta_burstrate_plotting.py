"""
functions to plot burst features
"""

# import public functions
from itertools import compress
import matplotlib.pyplot as plt
from os.path import join

# import own functions
from lfpecog_analysis.get_acc_task_derivs import define_OFF_ON_times
from lfpecog_analysis.load_SSD_features import ssdFeatures
from lfpecog_plotting.plotHelpers import get_colors



def plot_beta_bursts(
    sub, burst_count, SPLIT_OFF_ON,
    TO_SAVE_FIG, TO_PLOT_FIG, FIG_DIR
):

    clrs = list(get_colors().values())  # list with colorcodes
    win_len = burst_count['win_len_sec']
    thresh_origin = burst_count['threshold_origin']

    if SPLIT_OFF_ON:
        # get off/on definitions for subject
        fts = ssdFeatures(sub_list=[sub,])  # load features for CDRS scores

    ncols = len([k for k in burst_count.keys() if 'beta' in k])
    nrows = len(list(burst_count.values())[0])
    
    fig, axes = plt.subplots(
            nrows,  # takes n-datasources as n-rows
            ncols,  # takes n-bands as n-columns
            figsize=(ncols * 4, nrows * 4))

    for burst_type in ['short', 'long']:
        if burst_type == 'short':
            align_bin = 'left'
            clr = clrs[0]
        elif burst_type == 'long':
            align_bin = 'right'
            clr = clrs[2]

        for i_bw, beta_bw in enumerate(['lo_beta', 'hi_beta']):

            for i_d, dType in enumerate(burst_count[beta_bw].keys()):

                if SPLIT_OFF_ON:
                    # get off/on definitions for subject
                    win_sel = {}
                    win_sel['off'], win_sel['on'] = define_OFF_ON_times(
                        feat_times=burst_count[beta_bw][dType]['win_times'],
                        cdrs_scores=getattr(fts, f'sub{sub}').scores.total.copy(),
                        cdrs_times=getattr(fts, f'sub{sub}').scores.times.copy(),
                        incl_tasks='rest',
                        sub=sub,
                        data_version=burst_count['data_version'],
                    )
                    off_bursts = list(compress(
                        burst_count[beta_bw][dType][burst_type],
                        win_sel['off']))
                    on_bursts = list(compress(
                        burst_count[beta_bw][dType][burst_type],
                        win_sel['on']))
                    axes[i_d, i_bw].hist(
                        off_bursts, bins=5,
                        label=f'{burst_type} {beta_bw}: OFF, rest',
                        edgecolor=clr, facecolor='w',
                        hatch='//', alpha=.7, align=align_bin,)
                    axes[i_d, i_bw].hist(
                        on_bursts, bins=5,
                        label=f'{burst_type} {beta_bw}: ON, rest',
                        color=clr,
                        alpha=.4, align=align_bin,)

                else:
                    axes[i_d, i_bw].hist(
                        burst_count[beta_bw][dType][burst_type],
                        bins=5, label=f'{burst_type} {beta_bw} (all)',
                        facecolor=clr, alpha=.5,  align=align_bin,)
                
                axes[i_d, i_bw].set_title(f'sub-{sub}: {dType} {beta_bw} bursts')
            
    axes = axes.flatten()
    for ax in axes:
        ax.legend()
        
        ax.set_xlabel(f'burst-rate vs {thresh_origin}-threshold'
                            '\n(bursts per second)')
        ax.set_ylabel(f'n oberservations\n({win_len} s windows)')
    
    plt.tight_layout()
    
    if TO_SAVE_FIG:
        fig_name = f'betaBursts_histExplore_sub{sub}'
        if SPLIT_OFF_ON: fig_name += '_OffOnRest'
        else: fig_name += '_allWindows'
        
        plt.savefig(
            join(FIG_DIR, fig_name),
            dpi=150, facecolor='w',
        )
    
    if TO_PLOT_FIG:
        plt.show()
    else:
        plt.close()