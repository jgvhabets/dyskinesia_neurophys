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

    ylab = (rf"$\bfMovement$" + " " + rf"$\bfpresence$"
            + "\n(indiv. z-scored acc-rms)")
    xlab = (rf"$\bfTrue$" + " " + rf"$\bfdyskinesia$" +
            " " + rf"$\bfseverity$" + "\n(total CDRS score)")
    cbar_lab = (rf"$\bfPrediction$" + " " + rf"$\bferror$" +
                "\n(\u0394 CDRS points)")

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
                    'prediction', 'group_v8', 'scale')

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

        ax.set_xlabel(f'Movement presence\n(indiv. z-scored acc-rms)', size=fsize,)
        ax.set_ylabel(f'Observations (a.u.)', size=fsize,)


    plt.tight_layout()

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'final_Q1_2024',
                    'prediction', 'group_v8', 'binary')

        print(f'...saved {fig_name} in {path}')
        plt.savefig(join(path, fig_name), facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

    if SHOW_PLOT:
        plt.show()


def plot_predValues_per_ActBin(
    dat_dict: dict, model: str,
    bin_width = .8,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    SensSpec=True, PredAcc=False,
    fig_name='0000_binary_act_preds', 
):

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    fsize=14
    hist_bins = np.arange(-2, 4.1, bin_width)
    clr_sens = 'mediumseagreen'
    clr_spec = 'goldenrod'

    if SensSpec:
        metric_labels = ['sens', 'spec']
        metric_legends = ['Sensitivity', 'Specificity']
    else:
        metric_labels = ['pos_pv', 'neg_pv']
        metric_legends = ['"DYSKINESIA" predicted',
                          '"NO-DYSKINESIA" predicted']

    bins = {l: [] for l in metric_labels}
    for n in hist_bins:
        for l in metric_labels:
            bins[l].append([])
    
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
            all_pos = sum(sub_true[bin_sel] == 1)
            truepos = sum(np.logical_and(sub_true[bin_sel] == 1,
                                         sub_pred[bin_sel] == 1))
            trueneg = sum(np.logical_and(sub_true[bin_sel] == 0,
                                         sub_pred[bin_sel] == 0))
            all_pos_preds = sum(sub_pred[bin_sel] == 1)
            all_neg_preds = sum(sub_pred[bin_sel] == 0)


            if SensSpec:
                if all_pos > 0: bins['sens'][i_bin].append(truepos / all_pos)
                if all_neg > 0: bins['spec'][i_bin].append(trueneg / all_neg)
            else:
                if all_pos_preds > 0: bins['pos_pv'][i_bin].append(truepos / all_pos_preds)
                if all_neg_preds > 0: bins['neg_pv'][i_bin].append(trueneg / all_neg_preds)
            
            
            # print(f'...{sub}, added sns: {sens}, spc: {spec}, based on {sum(bin_sel)} samples')
    

    ax.axhline(xmin=hist_bins[0], xmax=hist_bins[-1], y=.5,
               color='gray', alpha=.2, lw=1, ls='--',)
    for y in [0, 1]: ax.axhline(xmin=hist_bins[0], xmax=hist_bins[-1], y=y,
               color='gray', alpha=.2, lw=1,)
    
    # PLOT BOXES
    boxes = {}
    for i_m, (m, w) in enumerate(
        zip(metric_labels, [-1 * bin_width/6, bin_width / 6])
    ):
        boxes[m] = ax.boxplot(
            bins[m],
            positions=hist_bins + w,
            patch_artist=True,
            widths=bin_width/3,
        )

    # Color Boxes
    for box_patch, clr in zip(
        boxes.values(),
        [clr_sens, clr_spec]
    ):
        for m in box_patch['medians']: m.set_color('gray')
        for box in box_patch['boxes']:
            box.set_facecolor(clr)
            box.set_edgecolor(clr)
            box.set_linewidth(0)
            box.set_alpha(.5)

    # Plot LEGEND
    if SensSpec: x_leg, y_leg = ([.8, .8], [.15, .05])
    else: x_leg, y_leg = ([.25, .6], [.85, .85])
    
    fig.text(x=x_leg[0], y=y_leg[0], s=metric_legends[0],
        va='bottom', ha='left', size=fsize-4,
        rotation=0,  weight='bold',
        bbox={'facecolor': clr_sens, 'alpha': .3, 'lw': 0,},
    )
    fig.text(x=x_leg[1], y=y_leg[1], s=metric_legends[1],
        va='bottom', ha='left', size=fsize-4,
        rotation=0,  weight='bold',
        bbox={'facecolor': clr_spec, 'alpha': .3, 'lw': 0,},
    )
    
    # plot title and labels
    if SensSpec:
        title = (rf"$\bfDyskinesia$" + " "+ rf"$\bfprediction$" + " " rf"$\bfperformance$" + " "
                    + rf"$\bfand$" + " " + rf"$\bfActivity$" + f"\n(model: {model})")
        ax.set_title(title, size=fsize-2,)
    else:
        ax.set_title(f'Model: {model}\n', size=fsize-2, loc='left',)

    ax.set_xlabel(rf"$\bfMovement$" + " " + rf"$\bfpresence$"
                  + "\n(indiv. z-scored acc-rms)",
                  size=fsize,)
    if SensSpec:
        lab = (rf"$\bfPrediction$" + " " + rf"$\bfmetrics$" + " " + "\n")
    else:
        lab = (rf"$\bfPrediction$" + " " + rf"$\bfaccuracies$" + " " + "\n(%)")
    ax.set_ylabel(lab, size=fsize,)            

    # fix ticks and axes
    ax.set_xticks(np.arange(-2, 4.1, 2))
    ax.set_xticklabels(np.arange(-2, 4.1, 2).astype(int))
    ax.set_ylim(-.1, 1.1)
    ax.set_yticks([0, .5, 1])
    if SensSpec: ax.set_yticklabels([0, .5, 1],)
    elif PredAcc: ax.set_yticklabels(['0 %', '50 %', '100 %'],)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize, )
    for r in ['right', 'top']: ax.spines[r].set_visible(False)


    plt.tight_layout()

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'final_Q1_2024',
                    'prediction', 'group_v8', 'binary')
        if SensSpec: path = join(path, 'SensSpec')
        elif PredAcc: path = join(path, 'PredAcc')

        print(f'...saved {fig_name} in {path}')
        plt.savefig(join(path, fig_name), facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

    if SHOW_PLOT:
        plt.show()


    
