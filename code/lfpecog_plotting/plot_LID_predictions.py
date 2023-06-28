"""
Standard Predictive Plots
"""

# import public packages
from os.path import join, exists
from os import makedirs
import matplotlib.pyplot as plt
from pandas import Series

from lfpecog_plotting.plotHelpers import get_colors


def plot_sub_binary_preds(
    preds_subs, SUBS, sub_ids,
    y_all_scale, ft_times_all,
    y_all_binary=None,
    PLOT_PROBA = False,
    SAVE_FIG=False, PLOT_FIG=False,
    fig_dir=None, fig_name=None
):

    clrs = list(get_colors().values())

    fig, axes = plt.subplots(
        len(SUBS), 1, figsize=(8, 12),
        # sharex='col',
    )

    fs = 14
    for i_s, sub in enumerate(SUBS):
        handles, labels = [], []

        plot_preds = preds_subs[sub]['pred']
        if PLOT_PROBA: plot_probas = preds_subs[sub]['proba'][:, 1]
        sub_sel = sub_ids == sub
        plot_cdrs = y_all_scale[sub_sel]  # get CDRS as full scale
        plot_fttimes = ft_times_all[sub_sel]
        assert len(plot_preds) == len(plot_cdrs), (
            '# predictions and # scores not equal'
        )

        ymax = max(plot_cdrs)
        if ymax == 0: ymax = 1
        
        # fill moments where LID was predicted
        axes[i_s].fill_between(plot_fttimes,
                            y1=-0, y2=ymax,
                            where=plot_preds == 1, alpha=.4,
                            color=clrs[1],
                            label='LID predicted')
        # fill moments where NO LID was predicted
        axes[i_s].fill_between(plot_fttimes,
                            y1=-0, y2=ymax,
                            color=clrs[4],
                            where=plot_preds == 0, alpha=.4,
                            label='No LID predicted')
        
        # plot probabilities of prediction
        if PLOT_PROBA:
            ax2 = axes[i_s].twinx()  # create second y-axis for probabilities
            ax2.plot(plot_fttimes, plot_probas, lw=.8, color='purple',
                    alpha=.8, label='Predicted probability')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Predicted\nprobability', fontsize=fs, weight='bold',)
            ax2.tick_params(axis='both', labelsize=fs, size=fs,)
            for side in ['top',]:
                ax2.spines[side].set_visible(False)
            hnd, lab = ax2.get_legend_handles_labels()
            handles.extend(list(hnd))
            labels.extend(list(lab))

        # plot CDRS as full scale
        axes[i_s].plot(plot_fttimes, plot_cdrs, lw=3, color='green',
                        label='Real CDRS (unilat.)')

        axes[i_s].set_title(f'sub-{sub}', weight='bold', fontsize=fs)
        axes[i_s].set_xlabel('Time (minutes vs L-Dopa intake)',
                            fontsize=fs, )
        axes[i_s].set_ylabel('Dyskinesia\n(CDRS)',
                            fontsize=fs, weight='bold',)
        hnd, lab = axes[i_s].get_legend_handles_labels()
        handles.extend(list(hnd))
        labels.extend(list(lab))

    axes[0].legend(handles, labels, frameon=False,
                loc='lower center', bbox_to_anchor=(.5, 1.2),
                fancybox=False, shadow=False,
                borderaxespad=1, ncol=3,
                prop={
                    # 'weight': 'bold',
                    'size': fs
                }
    )

    # plt.suptitle('Individual binary Dyskinesia-Predictions vs CDRS',
    #             #  weight='bold',
    #              fontsize=fs+4)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=fs, size=fs,)
        for side in ['top','right']:
            ax.spines[side].set_visible(False)
    plt.tight_layout()

    if SAVE_FIG:
        
        if not exists(fig_dir): makedirs(fig_dir)

        plt.savefig(join(fig_dir, fig_name),
                    facecolor='w', dpi=300,)

    if PLOT_FIG: plt.show()
    else: plt.close()


def plot_sub_gradual_preds(
    preds_subs, SUBS, sub_ids,
    ft_times_all,
    smooth_pred_samples=0,
    PLOT_PROBA = False,
    SAVE_FIG=False, PLOT_FIG=False,
    fig_dir=None, fig_name=None
):

    clrs = list(get_colors().values())

    fig, axes = plt.subplots(
        len(SUBS), 1, figsize=(8, 12),
        sharex='col',
    )

    fs = 14
    for i_s, sub in enumerate(SUBS):
        handles, labels = [], []

        y_preds = Series(preds_subs[sub]['pred'])
        if PLOT_PROBA: plot_probas = preds_subs[sub]['proba'][:, 1]
        sub_sel = sub_ids == sub
        plot_cdrs = preds_subs[sub]['true']
        plot_fttimes = ft_times_all[sub_sel]
        assert len(y_preds) == len(plot_cdrs), (
            '# predictions and # scores not equal'
        )

        if smooth_pred_samples > 0:
            y_preds = y_preds.rolling(window=smooth_pred_samples,
                                      center=True).mean()

        plot_fttimes = plot_fttimes * 60

        axes[i_s].bar(x=plot_fttimes,
                      height=y_preds,
                      width=.5,
                    #   allign='edge',
                      color=clrs[2],
                      edgecolor=clrs[2],
                      label='CDRS predicted')
        
        # # plot probabilities of prediction
        # if PLOT_PROBA:
        #     ax2 = axes[i_s].twinx()  # create second y-axis for probabilities
        #     ax2.plot(plot_fttimes, plot_probas, lw=.8, color='purple',
        #             alpha=.8, label='Predicted probability')
        #     ax2.set_ylim(0, 1)
        #     ax2.set_ylabel('Predicted\nprobability', fontsize=fs, weight='bold',)
        #     ax2.tick_params(axis='both', labelsize=fs, size=fs,)
        #     for side in ['top',]:
        #         ax2.spines[side].set_visible(False)
        #     hnd, lab = ax2.get_legend_handles_labels()
        #     handles.extend(list(hnd))
        #     labels.extend(list(lab))

        # plot CDRS as full scale
        axes[i_s].plot(plot_fttimes, plot_cdrs,
                          lw=1, color=clrs[0],
                          label='Real CDRS (total bilat.)')

        # axes[i_s].set_title(f'sub-{sub}', weight='bold', fontsize=fs)
        
        axes[i_s].set_ylabel(
            # 'Dyskinesia\n(CDRS)',
            f'sub-{sub}',
            fontsize=fs, weight='bold',)
        hnd, lab = axes[i_s].get_legend_handles_labels()
        handles.extend(list(hnd))
        labels.extend(list(lab))

        # axes[i_s].set_ylim(-2, 12)
        axes[i_s].set_ylim(-.5, 3)
        # axes[i_s].set_xticks(plot_fttimes)
        # axes[i_s].set_xticklabels(plot_fttimes)
        # print(sub, ticks)

    axes[0].legend(handles, labels, frameon=False,
                loc='lower center', bbox_to_anchor=(.5, 1.2),
                fancybox=False, shadow=False,
                borderaxespad=1, ncol=3,
                prop={'size': fs}
    )
    axes[-1].set_xlabel('Time (minutes vs L-Dopa intake)',
                            fontsize=fs, weight='bold',)

    # plt.suptitle('Individual binary Dyskinesia-Predictions vs CDRS',
    #             #  weight='bold',
    #              fontsize=fs+4)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=fs, size=fs,)
        for side in ['top','right']:
            ax.spines[side].set_visible(False)
    
    plt.tight_layout(h_pad=.5)

    if SAVE_FIG:
        
        if not exists(fig_dir): makedirs(fig_dir)

        plt.savefig(join(fig_dir, fig_name),
                    facecolor='w', dpi=300,)

    if PLOT_FIG: plt.show()
    else: plt.close()