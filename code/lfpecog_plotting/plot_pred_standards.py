"""
Standard Predictive Plots
"""

# import public packages
from os.path import join
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from utils.utils_fileManagement import get_project_path

def plot_confMatrix(
    cm, to_show=True, to_save=False,
    fig_path=None, fig_name=None,
):
    fs=14

    plt.imshow(cm)
    # cbar = plt.colorbar()
    # cbar.set_label('# observations\n', rotation=270, fontsize=fs)
    for x_txt, y_txt in product([0, 1], [0, 1]):
        if cm[x_txt, y_txt] > 450:
            plt.text(x_txt, y_txt, cm[x_txt, y_txt], color='k',
                    ha='center', fontsize=fs, weight='bold',)
        else:    
            plt.text(x_txt, y_txt, cm[x_txt, y_txt], color='w',
                    ha='center', fontsize=fs, weight='bold',)

    plt.xlabel('Predicted Label', weight='bold', fontsize=fs+2)
    plt.xticks([0, 1], labels=['No LID', 'LID'])
    plt.yticks([0, 1], labels=['No LID', 'LID'])
    plt.ylabel('True Label', weight='bold', fontsize=fs+2)

    plt.tick_params(axis='both', labelsize=fs, size=fs)
    plt.tight_layout()

    if to_save:
        plt.savefig(join(fig_path, 'prediction', fig_name),
                    facecolor='w', dpi=300,)
    if to_show: plt.show()
    else: plt.close()


from matplotlib.gridspec import GridSpec


def plot_subPreds_over_time(
    t_minutes=None, y_true=None, y_pred=None, acc=None,
    sub_dict=False, lid_out_param='binary', 
    PLOT_CBARS=True,
    SAVE_PLOT=False, SHOW_PLOT=True, FOR_FIG=False,
    fig_name='pred_results_cmaps_v0', model=False,
):
    """
    Plot timeline with colormaps of true and prediction
    labels
    """
    if isinstance(model, str): assert model in ['cebra', 'lm'], 'wrong model'

    if isinstance(sub_dict, dict):
        t_minutes = sub_dict['times']
        y_true = sub_dict['y_true']
        y_pred = sub_dict['y_pred']
        acc = sub_dict['acc']

    cmap_true = 'Oranges'
    cmap_pred = 'Purples'
    acc_ax = 0
    true_ax = 1
    pred_ax = 2
   
    fsize=16

    if lid_out_param == 'binary': MAX_CDRS = 3
    elif lid_out_param == 'scale': MAX_CDRS = 6

    ### CREATE FIGURE
    h_ratios = [1, 1, 1]
    w_ratios = [1]
    if PLOT_CBARS: w_ratios += [.1, .1]

    # plot colorbars above
    # h_ratios = [.5, .5, .2] + h_ratios
    # acc_ax += 3
    # true_ax += 3
    # pred_ax += 3

    if not PLOT_CBARS:
        gridspec = dict(hspace=0.0,
                        height_ratios=h_ratios,
                        width_ratios=w_ratios,)

        fig, axes = plt.subplots(nrows=len(h_ratios),
                                ncols=len(w_ratios),
                                gridspec_kw=gridspec,
                                figsize=(12, 2),
                                constrained_layout=True,
                                )
    else:
        fig = plt.figure( figsize=(15, 3),)  # layout='constrained',
        gs = GridSpec(len(h_ratios), 20, figure=fig)
        n_gs_labels = 3
        n_gs_cbars = 2
    
    axes_list = []



    ### PLOT ACC ROW
    if not PLOT_CBARS: ax_acc = axes[acc_ax]
    else: ax_acc = fig.add_subplot(gs[acc_ax, n_gs_labels:-n_gs_cbars])

    ax_acc.scatter(t_minutes, acc, s=10,
               color='rosybrown', alpha=.9,)
    ax_acc.set_ylim(-2, 3)
    ax_mark_nans(t_minutes, ax_acc)
    axes_list.append(ax_acc)
    


    ### PLOT true DYSKINESIA ROW HEATMAP
    if not PLOT_CBARS: ax_true = axes[true_ax]
    else: ax_true = fig.add_subplot(gs[true_ax, n_gs_labels:-n_gs_cbars], sharex=ax_acc,)

    heat_true = ax_true.pcolor(
        t_minutes, np.arange(2),
        np.atleast_2d([np.zeros(len(t_minutes)),
                       y_true.astype(float)]),
        vmin=0, vmax=MAX_CDRS, cmap=cmap_true,
    )
    ax_true.set_ylim(0.5, 1)
    ax_mark_nans(t_minutes, ax_true)
    axes_list.append(ax_true)
    


    ### PLOT predicted DYSKINESIA ROW HEATMAP
    if not PLOT_CBARS: pred_ax = axes[pred_ax]
    else: pred_ax = fig.add_subplot(gs[pred_ax, n_gs_labels:-n_gs_cbars], sharex=ax_acc,)

    heat_pred = pred_ax.pcolor(
        t_minutes, np.arange(2),
        np.atleast_2d([np.zeros(len(t_minutes)),
                       y_pred.astype(float)]),
        vmin=0, vmax=MAX_CDRS, cmap=cmap_pred,
    )
    pred_ax.set_ylim(0.5, 1)
    pred_ax.set_xlabel('Time (minutes after L-Dopa intake)',
                        size=fsize+4, weight='bold',)
    ax_mark_nans(t_minutes, pred_ax)
    axes_list.append(pred_ax)
    # set xticks for last row
    xticks = np.arange(0, max(t_minutes), 15)
    pred_ax.set_xticks(xticks, size=fsize,)
    pred_ax.set_xticklabels(xticks.astype(int), size=fsize)
    


    ### plot COLORBAR-TRUE
    ax = fig.add_subplot(gs[:, -2])
    cbar_true = fig.colorbar(heat_true, ax=ax, pad=.12,
                        use_gridspec=True, fraction=1, aspect=5,
                        # orientation='horizontal',
                        )
    axes_list.append(ax)
    ax.set_xticks([],)


    ### plot COLORBAR-PRED
    ax = fig.add_subplot(gs[:, -1])
    cbar_pred = fig.colorbar(heat_pred, ax=ax, pad=.12,
                        use_gridspec=True, fraction=1, aspect=5,
                        # orientation='horizontal',
                        )
    axes_list.append(ax)
    ax.set_xticks([],)

    for cbar in [cbar_true, cbar_pred]:
        cbar.ax.set_xticks([])
        cbar.ax.set_yticks(np.linspace(0, MAX_CDRS, 5), size=fsize,)
        cbar.ax.get_yaxis().labelpad=20

    cbar_true.ax.set_yticklabels([],)
    cbar_pred.ax.set_yticklabels(np.linspace(0, MAX_CDRS, 5), size=fsize,)
    cbar.ax.set_ylabel('CDRS score', rotation=90, size=fsize + 2, weight='bold',)
    
    # Label space ax
    label_ax = fig.add_subplot(gs[:, :n_gs_labels])
    label_ax.set_xticks([],)
    axes_list.append(label_ax)

    
    # Plot xlabel at bottom
    for ax, r in product(axes_list,
                         ['bottom', 'left', 'right', 'top']):
        ax.spines[r].set_visible(False)
        ax.set_yticks([],)
    # hide xticks for other axes
    plt.setp(ax_acc.get_xticklabels(), visible=False)
    plt.setp(ax_true.get_xticklabels(), visible=False)
    

    # Plot row labels
    fig.text(x=.13, y=.82, s='Activity',
             va='center', ha='right', size=fsize,
             rotation=0,  weight='bold',
             bbox={'facecolor': 'rosybrown', 'alpha': .3},
             )
    fig.text(x=.13, y=.57, s='True LID',
             va='center', ha='right', size=fsize,
             rotation=0, weight='bold',
             bbox={'alpha': .3, 'color': 'darkorange'},)
    fig.text(x=.13, y=.32, s='Predicted LID',
             va='center', ha='right', size=fsize,
             rotation=0, weight='bold',
             bbox={'alpha': .3, 'color': 'darkblue'},)
    
    # # Plot TASK
    # fig.text(x=.12, y=.86, s='TASK',
    #          va='center', ha='right', size=fsize-4,
    #          rotation=0,  weight='bold',
    #          bbox={'alpha': .3, 'color': 'darkgreen'},
    #          )

    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                        'final_Q1_2024',
                        'prediction')
        if isinstance(model, str): path = join(path, model, lid_out_param)

        if FOR_FIG:
            fig_name += '.pdf'
            plt.savefig(join(path, fig_name), facecolor='w',)
            
        else:
            print(f'...saved {fig_name} in {path}')
            plt.savefig(join(path, fig_name),
                        facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

        

    if SHOW_PLOT:
        plt.show()



def plot_groupPreds_over_time(
    dat_dict, lid_out_param, 
    PLOT_CBARS=True,
    SAVE_PLOT=False, SHOW_PLOT=True, FOR_FIG=False,
    fig_name='pred_results_cmaps_v0', model=False,
):
    """
    Plot timeline with colormaps of true and prediction
    labels
    """
    if isinstance(model, str): assert model in ['cebra', 'lm'], 'wrong model'
    

    cmap_true = 'Oranges'
    cmap_pred = 'Purples'
    acc_ax = 1
    true_ax = 2
    pred_ax = 3
   
    fsize=16

    if lid_out_param == 'binary': MAX_CDRS = 3
    elif lid_out_param == 'scale': MAX_CDRS = 6

    ### CREATE FIGURE
    base_h_ratios = [.5, .5]  # colorbars
    sub_h_ratios = [.3, .5, .5, .5]
    h_ratios = base_h_ratios.copy()
    for s in dat_dict.keys(): h_ratios.extend(sub_h_ratios)
    w_ratios = [1]

    # plot colorbars above
    # h_ratios = [.5, .5, .2] + h_ratios
    # acc_ax += 3
    # true_ax += 3
    # pred_ax += 3

    gridspec = dict(hspace=0.0,
                    height_ratios=h_ratios,
                    width_ratios=w_ratios,)

    fig, axes = plt.subplots(nrows=len(h_ratios),
                            ncols=len(w_ratios),
                            gridspec_kw=gridspec,
                            figsize=(10, len(dat_dict)),
                            constrained_layout=True,
                            )

        # fig = plt.figure( figsize=(15, 3),)  # layout='constrained',
        # gs = GridSpec(len(h_ratios), 20, figure=fig)
        # n_gs_labels = 3
        # n_gs_cbars = 2
    
    axes_list = []
    fill_axes = []


    ### plot COLORBAR-TRUE
    cmap_arr = np.arange(0, MAX_CDRS + .1, 1)
    cbar_true = axes[0].pcolor(
        cmap_arr, np.arange(2),
        np.atleast_2d([np.zeros(len(cmap_arr)),
                        cmap_arr.astype(float)]),
        vmin=0, vmax=MAX_CDRS, cmap=cmap_true,
    )
    ### plot COLORBAR-PRED
    cbar_pred = axes[1].pcolor(
        cmap_arr, np.arange(2),
        np.atleast_2d([np.zeros(len(cmap_arr)),
                        cmap_arr.astype(float)]),
        vmin=0, vmax=MAX_CDRS, cmap=cmap_pred,
    )

    for cbar_ax in [axes[0], axes[1]]:
        cbar_ax.set_ylim(0.5, 1)
        cbar_ax.set_yticks([], )
        
    axes[0].set_xticklabels([],)
    axes[0].set_xticks([],)
    axes[1].set_xticks(cmap_arr, size=fsize,)
    axes[1].set_xticklabels(cmap_arr.astype(int), size=fsize,)
    axes[1].set_xlabel('CDRS score', rotation=0, size=fsize + 2, weight='bold',)
    
    

    for i_sub, sub in enumerate(dat_dict.keys()):

        t_minutes = dat_dict[sub]['times']
        y_true = dat_dict[sub]['y_true']
        y_pred = dat_dict[sub]['y_pred']
        acc = dat_dict[sub]['acc']


        # fill ax
        fill_axes.append(axes[len(base_h_ratios) + (i_sub * len(sub_h_ratios))])


        ### PLOT ACC ROW
        i_ax_acc = len(base_h_ratios) + (i_sub * len(sub_h_ratios)) + acc_ax
        ax_acc = axes[i_ax_acc]
        ax_acc.scatter(t_minutes, acc, s=5, color='rosybrown', alpha=.9,)
        ax_acc.set_ylim(-2, 3)
        ax_mark_nans(t_minutes, ax_acc)
        axes_list.append(ax_acc)
    


        ### PLOT true DYSKINESIA ROW HEATMAP
        ax_true = axes[i_ax_acc + 1]
        ax_acc.get_shared_x_axes().join(ax_acc, ax_true)
        heat_true = ax_true.pcolor(
            t_minutes, np.arange(2),
            np.atleast_2d([np.zeros(len(t_minutes)),
                        y_true.astype(float)]),
            vmin=0, vmax=MAX_CDRS, cmap=cmap_true,
        )
        ax_true.set_ylim(0.5, 1)
        ax_mark_nans(t_minutes, ax_true)
        axes_list.append(ax_true)
    


        ### PLOT predicted DYSKINESIA ROW HEATMAP
        ax_pred = axes[i_ax_acc + 2]
        ax_acc.get_shared_x_axes().join(ax_acc, ax_pred)

        heat_pred = ax_pred.pcolor(
            t_minutes, np.arange(2),
            np.atleast_2d([np.zeros(len(t_minutes)),
                        y_pred.astype(float)]),
            vmin=0, vmax=MAX_CDRS, cmap=cmap_pred,
        )
        ax_pred.set_ylim(0.5, 1)
        
        ax_mark_nans(t_minutes, ax_pred)
        axes_list.append(ax_pred)
        
        # set xticks for last row
        if i_sub == len(dat_dict) - 1:
            xticks = np.arange(0, max(t_minutes), 15)
            ax_pred.set_xticks(xticks, size=fsize,)
            ax_pred.set_xticklabels(xticks.astype(int), size=fsize)
            ax_pred.set_xlabel('Time (minutes after L-Dopa intake)',
                            size=fsize+4, weight='bold',)
        else:
            plt.setp(ax_pred.get_xticklabels(), visible=False)

        



        # hide xticks for other axes
        # plt.setp(ax_acc.get_xticks(), visible=False)
        # plt.setp(ax_true.get_xticks(), visible=False)
        plt.setp(ax_acc.get_xticklabels(), visible=False)
        plt.setp(ax_true.get_xticklabels(), visible=False)
        
    # # Label space ax
    # label_ax = fig.add_subplot(gs[:, :n_gs_labels])
    # label_ax.set_xticks([],)
    # axes_list.append(label_ax)

    
    # Plot xlabel at bottom
    for ax, r in product(axes_list,
                         ['bottom', 'left', 'right', 'top']):
        ax.spines[r].set_visible(False)
        ax.set_yticks([],)
    
    for fill_ax in fill_axes:
        fill_ax.set_xticks([])
        fill_ax.set_yticks([])
        for r in ['bottom', 'left', 'right', 'top']:
            fill_ax.spines[r].set_visible(False)

    # # Plot row labels
    # fig.text(x=.13, y=.82, s='Activity',
    #          va='center', ha='right', size=fsize,
    #          rotation=0,  weight='bold',
    #          bbox={'facecolor': 'rosybrown', 'alpha': .3},
    #          )
    # fig.text(x=.13, y=.57, s='True LID',
    #          va='center', ha='right', size=fsize,
    #          rotation=0, weight='bold',
    #          bbox={'alpha': .3, 'color': 'darkorange'},)
    # fig.text(x=.13, y=.32, s='Predicted LID',
    #          va='center', ha='right', size=fsize,
    #          rotation=0, weight='bold',
    #          bbox={'alpha': .3, 'color': 'darkblue'},)
    
    

    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                        'final_Q1_2024',
                        'prediction', 'group')

        if FOR_FIG:
            fig_name += '.pdf'
            plt.savefig(join(path, fig_name), facecolor='w',)
            
        else:
            print(f'...saved {fig_name} in {path}')
            plt.savefig(join(path, fig_name),
                        facecolor='w', dpi=300,)
        
        if not SHOW_PLOT: plt.close()

        

    if SHOW_PLOT:
        plt.show()


def ax_mark_nans(t_line, ax):
    (y1, y2) = ax.get_ylim()
    min_range = np.arange(round(min(t_line)),
                        round(max(t_line)), 1)
    min_data = np.unique(np.around(t_line))
    nan_range = ~np.isin(min_range, min_data)
    
    ax.fill_between(min_range, y1=y1, y2=y2, where=nan_range,
                    facecolor='white', edgecolor='lightgray',
                    alpha=1, hatch='//',)
    

