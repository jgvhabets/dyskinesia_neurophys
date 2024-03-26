"""
Plot activity distribution within prediction results
"""

# import public packages
from os.path import join
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from utils.utils_fileManagement import get_project_path
from lfpecog_plotting.plotHelpers import (
    get_plot_jitter, remove_duplicate_legend
)



def scatter_predErrors(
    dat_dict,
    SAVE_PLOT = True,
    SHOW_PLOT = False,
    fig_name = f'0000_scatterErrors',
    ADD_JITTER = True,
    SHUFFLE_SCATTERS = True,
    ROUND_PREDS: bool = False,
    modelname=False, out_param=' ',
):

    fig, ax = plt.subplots(1,1)

    x_all_scat, y_all_scat, diff_all_scat = [], [], []

    fsize = 14
    vmin, vmax = -3, 3

    for sub in dat_dict:
        sub_dict = dat_dict[sub]

        # GET VALUES
        sub_true = sub_dict['y_true'].copy()
        sub_pred = sub_dict['y_pred'].copy()
        if ROUND_PREDS: sub_pred = np.around(sub_pred)
        pred_diff = sub_pred - sub_true
        x_temp = sub_true
        y_temp = sub_dict['acc'].copy()
        assert len(x_temp) == len(y_temp) == len(pred_diff), 'incorrect lengths'

        # add jitter
        if ADD_JITTER:
            xjitt, yjitt = get_plot_jitter(x_temp, y_temp, ZERO_SPACE=False,)
            x_temp, y_temp = x_temp + xjitt, y_temp + yjitt
        
        x_all_scat.extend(x_temp)
        y_all_scat.extend(y_temp)
        diff_all_scat.extend(pred_diff)

        # print(f'...added for sub-{sub}: n={len(x_temp)}')
    x_all_scat = np.array(x_all_scat)
    y_all_scat = np.array(y_all_scat)
    diff_all_scat = np.array(diff_all_scat)

    if SHUFFLE_SCATTERS:
        np.random.seed(27)
        idx = np.arange(len(x_all_scat))
        np.random.shuffle(idx)
        x_all_scat = x_all_scat[idx]
        y_all_scat = y_all_scat[idx]
        diff_all_scat = diff_all_scat[idx]


    # SCATTER ALL DOTS
    scat = ax.scatter(x_all_scat, y_all_scat, c=diff_all_scat,
                    alpha=.3, cmap='coolwarm', vmin=vmin, vmax=vmax,)

    # scatter correct preds in green
    # incorr_sel = diff_all_scat != 0
    # corr_sel = diff_all_scat == 0
    # print(f'n-correct: {sum(corr_sel)} / {len(corr_sel)} ')

    # incorr_scat = ax.scatter(x_all_scat[incorr_sel], y_all_scat[incorr_sel],
    #                   alpha=.3, c=diff_all_scat[incorr_sel],
    #                   cmap='coolwarm', vmin=vmin, vmax=vmax,)
    # corr_scat = ax.scatter(x_all_scat[corr_sel], y_all_scat[corr_sel],
    #                        alpha=1, edgecolor='forestgreen',
    #                        facecolor='None',)


    cbar = ax.scatter([], [], c=[], cmap='coolwarm', vmin=vmin, vmax=vmax,)

    # PM: plot cbar without transparancy for cbar legend

    ylab = rf"$\bfMovement$" + " " + rf"$\bfpresence$" + "\n(acc, z-scored RMS)"
    xlab = rf"$\bfTrue$" + " " + rf"$\bfdyskinesia$" + " " + rf"$\bfseverity$" + "\n(CDRS, sum)"
    cbar_lab = rf"$\bfPrediction$" + " " + rf"$\bferror$" + "\n(CDRS, points)"

    ax.set_xlabel(xlab, size=fsize,)
    ax.set_ylabel(ylab, size=fsize,)
    ax.tick_params(size=fsize, labelsize=fsize,)
    cbar = plt.colorbar(cbar, ax=ax,)
    cbar.set_label(label=cbar_lab, size=fsize,)
    cbar.set_ticks(np.linspace(vmin, vmax, 5), size=fsize,)
    cbar.set_ticklabels(np.linspace(vmin, vmax, 5), size=fsize,)

    if isinstance(modelname, str):
        ax.set_title(f'Model: {modelname}, {out_param} prediction',
                     size=fsize, weight='bold',)

    plt.tight_layout()

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'final_Q1_2024',
                    'prediction', 'group')

        print(f'...saved {fig_name} in {path}')
        plt.savefig(join(path, fig_name), facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

        

    if SHOW_PLOT:
        plt.show()



def plot_binary_act_distr(
    dat_dict: dict, model: str,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    fig_name='0000_binary_act_preds', 
):

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fsize=14
    hist_bins = np.arange(-2, 4.1, .1)

    for sub in dat_dict:
        
        sub_dict = dat_dict[sub]

        # GET VALUES
        sub_true = sub_dict['y_true'].copy()
        sub_pred = sub_dict['y_pred'].copy()
        sub_acc = sub_dict['acc'].copy()

        for i_ax, (label, sel) in enumerate(
            zip(['negative', 'positive'],
                [sub_pred == 0, sub_pred == 1])
        ):
            # print(f'{sub}, {label} predictions, n = {sum(sel)}')
            sel_preds = sub_pred[sel]
            sel_trues = sub_true[sel]
            sel_acc = sub_acc[sel]
            # split in false and true (correct) predictions
            false_sel = sel_preds == sel_trues
            correct_sel = sel_preds != sel_trues

            false_acc = sel_acc[false_sel]
            false_truelabel = sel_trues[false_sel]
            correct_acc = sel_acc[correct_sel]
            correct_truelabel = sel_trues[correct_sel]

            axes[i_ax].hist(
                false_acc, bins=hist_bins,
                color='orangered', alpha=.5,
                label=f'incorrect',
                density=True,
            )
            axes[i_ax].hist(
                correct_acc, bins=hist_bins,
                color='forestgreen', alpha=.5,
                label=f'correct',
                density=True,
            )
            if label == 'negative':
                title = ("Activity distribution:\n" + rf"$\bfNo-Dyskinesia$"
                        + " " + rf"$\bfprediction$" + f"\n(model: {model})")
            elif label == 'positive':
                title = ("Activity distribution:\n" + rf"$\bfDyskinesia-present$"
                        + " " + rf"$\bfprediction$" + f"\n(model: {model})")
            axes[i_ax].set_title(title, size=fsize,    )
            

    for ax in axes:
        hnd_labs = ax.get_legend_handles_labels()
        hnd, labs = remove_duplicate_legend(hnd_labs)
        ax.legend(hnd, labs, fontsize=fsize, frameon=False,)

        ax.set_xticks(np.arange(-2, 4.1, 2))
        ax.set_xticklabels(np.arange(-2, 4.1, 2).astype(int))
        y1, y2 = ax.get_ylim()
        ax.set_yticks(np.arange(0, y2, 1))
        ax.set_yticklabels(np.arange(0, y2, 1).astype(int))
        ax.tick_params(axis='both', size=fsize, labelsize=fsize, )

        ax.set_xlabel(f'Activitiy (indiv - zscored)', size=fsize,)
        ax.set_ylabel(f'Observations (a.u.)', size=fsize,)


    plt.tight_layout()

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'final_Q1_2024',
                    'prediction', 'group')

        print(f'...saved {fig_name} in {path}')
        plt.savefig(join(path, fig_name), facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

    if SHOW_PLOT:
        plt.show()


def plot_predValues_per_ActBin(
    dat_dict: dict, model: str,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    fig_name='0000_binary_act_preds', 
):

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    fsize=14
    bin_width = .5
    hist_bins = np.arange(-2, 4.1, bin_width)

    bins = {'sens': [], 'spec': []}
    for n in hist_bins:
        bins['sens'].append([])
        bins['spec'].append([])
    
    print(f'lengths: {len(bins["sens"])}, {len(bins["spec"])}')

    for sub in dat_dict:
        
        sub_dict = dat_dict[sub]

        # GET VALUES
        sub_true = sub_dict['y_true'].copy()
        sub_pred = sub_dict['y_pred'].copy()
        sub_acc = sub_dict['acc'].copy()


        for i_bin, bin_start in enumerate(hist_bins):
            bin_sel = np.logical_and(sub_acc >= bin_start,
                                     sub_acc < (bin_start + bin_width))
            if i_bin == 0:
                bin_sel = sub_acc < (bin_start + bin_width)  # no neg border
            elif i_bin == (len(hist_bins) - 1):
                bin_sel = sub_acc >= bin_start  # no pos border
 
            # print(f'...sub-{sub}, bin # {i_bin}: {sum(bin_sel)} samples')
            if sum(bin_sel) == 0: continue

            all_neg = sum(sub_true[bin_sel] == 0)
            truepos = sum(np.logical_and(sub_true[bin_sel] == 1,
                                         sub_pred[bin_sel] == 1))
            trueneg = sum(np.logical_and(sub_true[bin_sel] == 0,
                                         sub_pred[bin_sel] == 0))
            all_pos_preds = sum(sub_pred[bin_sel] == 1)

            # sens = truepos / all_pos_preds
            # spec = trueneg / all_neg

            if all_pos_preds > 0: bins['sens'][i_bin].append(truepos / all_pos_preds)
            if all_neg > 0: bins['spec'][i_bin].append(trueneg / all_neg)

            # print(f'...{sub}, added sns: {sens}, spc: {spec}, based on {sum(bin_sel)} samples')
    
    # print(f'lengths sens list: {[len(l) for l in bins["sens"]]}')
    # print(f'lengths spec list: {[len(l) for l in bins["spec"]]}')

    # print(f'lengths sens list: {bins["sens"]}')
    # print(f'lengths spec list: {bins["spec"]}')

    ax.axhline(xmin=hist_bins[0], xmax=hist_bins[-1], y=.5,
               color='gray', alpha=.2, lw=1, ls='--',)
    for y in [0, 1]: ax.axhline(xmin=hist_bins[0], xmax=hist_bins[-1], y=y,
               color='gray', alpha=.2, lw=1,)
    
    # PLOT BOXES
    box_sens = ax.boxplot(
        bins['sens'],
        positions=hist_bins - (bin_width/6),
        patch_artist=True,
        widths=bin_width/3,
    )
    box_spec = ax.boxplot(
        bins['spec'],
        positions=hist_bins + (bin_width/6),
        patch_artist=True,
        widths=bin_width/3,
    )
    # Color Boxes
    for box in box_spec['boxes']:
        box.set_facecolor('purple')
        box.set_alpha(.5)
    for box in box_sens['boxes']:
        box.set_facecolor('green')
        box.set_alpha(.5)
    
    for box in [box_sens, box_spec]:
        for m in box['medians']: m.set_color('gray')

    # Plot LEGEND
    fig.text(x=.8, y=.15, s='Sensitivity',
        va='bottom', ha='left', size=fsize-2,
        rotation=0,  weight='bold',
        bbox={'facecolor': 'forestgreen', 'alpha': .3},
    )
    fig.text(x=.8, y=.05, s='Specificity',
        va='bottom', ha='left', size=fsize-2,
        rotation=0,  weight='bold',
        bbox={'facecolor': 'purple', 'alpha': .3},
    )
    
    # plot title and labels
    title = (rf"$\bfDyskinesia$" + " "+ rf"$\bfprediction$" + " " rf"$\bfperformance$" + " "
                + rf"$\bfvs.$" + " " + rf"$\bfActivity$" + f"\n(model: {model})")
    ax.set_title(title, size=fsize-2,)
    ax.set_xlabel(rf"$\bfActivity$" + "\n(indiv. z scored)", size=fsize,)
    ax.set_ylabel(rf"$\bfIndividual$" + " " rf"$\bfscores$" + " " + "\n(a.u.)", size=fsize,)            

    # fix ticks and axes
    ax.set_xticks(np.arange(-2, 4.1, 2))
    ax.set_xticklabels(np.arange(-2, 4.1, 2).astype(int))
    ax.set_ylim(-.1, 1.1)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels([0, .5, 1],)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize, )
    for r in ['right', 'top']: ax.spines[r].set_visible(False)


    plt.tight_layout()

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'final_Q1_2024',
                    'prediction', 'group')

        print(f'...saved {fig_name} in {path}')
        plt.savefig(join(path, fig_name), facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

    if SHOW_PLOT:
        plt.show()


    
