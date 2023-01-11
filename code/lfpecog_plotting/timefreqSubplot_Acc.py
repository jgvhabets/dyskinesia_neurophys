"""
Subplot with Accelerometer data
for plotting features over time
"""

# import publick functions
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

# import own functions
from utils.utils_fileManagement import get_project_path



def subplot_acc(fig, axes, fs, sub, plot_times, winLen_sec,
                data_version, i_plot_ax=-1, plot_task=False,
                colors={'Left': 'darkblue', 'Right': 'green'},
                plot_type='bars',
):
    """
    Create subplot with Accelerometer activity
    
    Inputs:
        - fig, axes: from main plot
        - fs: fontsize
        - sub: subject code e.g. '001'
        - plot_times: array with times of main
            TimeFreq plot, given in seconds
        - winLen_sec: window length of features plotted
        - data_version (of features plotted)
        - i_plot_ax: CDRS subplot default last
        - plot_task: include shading for task type
        - plot_type: visualise Taps and Moves as 'fill' filling
            between lines, or 'bars' as barplot

    
    Returns:
        - fig, axes with added subplot
    """
    if type(axes) == np.ndarray: ax = axes[i_plot_ax]
    else: ax = axes

    assert plot_type in ['fill', 'bars'], (
        'Warning subplot_acc(): inserted plot_type '
        f'"{plot_type}" not "bars" or "fill"'
    )


    from utils.utils_fileManagement import (
        load_class_pickle,
        mergedData,
        correct_acc_class
    )
    # load Acc-detected movement labels
    acc = load_class_pickle(join(
        get_project_path('data'),
        'merged_sub_data', data_version,
        f'{sub}_mergedDataClass_{data_version}_noEphys.P'
    ))
    print(vars(acc).keys())
    print(f'col names PRE: {acc.colnames}')
    acc = correct_acc_class(acc)
    print(f'col names POST: {acc.colnames}')

    acc_plot_lines = {
        'Left': {'tap': [], 'move': []},
        'Right': {'tap': [], 'move': []}}
    acc_plot_counts = {
        'Left': {'tap': [], 'move': []},
        'Right': {'tap': [], 'move': []}}
    if plot_type == 'fill':
        base_y = {'Left': 0.5, 'Right': 1.75}
    elif plot_type == 'bars':
        base_y = {'Left': 0.5, 'Right': 15}
        ymax = max(base_y.values())
    task_list = []
    
    for t in plot_times:
        try:
            i_start = np.where(acc.times == t)[0][0]
        except IndexError:
            i_start = np.argmin(abs(acc.times - t))
        try:
            i_end = np.where(acc.times == (t + winLen_sec))[0][0]
        except IndexError:
            i_end = np.argmin(abs(acc.times - (t + winLen_sec)))

        # get corr window in acc-data
        win = acc.data[i_start:i_end, :]  # window borders in seconds (as plot_times)
        assert len(win) > 0, f'{i_start} -> {i_end}'
        for side in colors.keys():
            # get % in window of type of unilateral movement
            i_tap = np.where(acc.colnames == f'{side.lower()}_tap')[0][0]
            tap_perc = sum(win[:, i_tap]) / len(win) # value between 0 and 1
            tap_y = tap_perc + base_y[side]
            i_move = np.where(acc.colnames == f'{side.lower()}_move')[0][0]
            move_perc = sum(win[:, i_move]) / len(win)  # value between 0 and 1
            move_y = move_perc + tap_y
            # create full timeseries with tap and move borders
            acc_plot_lines[side]['tap'].append(tap_y)
            acc_plot_lines[side]['move'].append(move_y)
            ### histo-counts
            n_taps = sum(np.diff(win[:, i_tap]) == 1)  # n-times from 0 to 1
            n_moves = sum(np.diff(win[:, i_move]) == 1)  # n-times from 0 to 1
            acc_plot_counts[side]['tap'].append(n_taps)
            acc_plot_counts[side]['move'].append(n_moves)

        # include most prevalent task label
        if plot_task:  # prevent double labels
            icol = np.where(acc.colnames == 'task')[0][0]
            task_ids, counts = np.unique(
                win[:, icol], return_counts=True
            )
            order = np.argsort(~ counts)  # inverse for largest first
            task_present = task_ids[order][0]
            task_list.append(task_present)

        # print(f'...CHECK: t: {t}, t-start: {acc.times[i_start]}, t-end: {acc.times[i_end]}')

    # Plot activity for both sides
    for n_s, side in enumerate(colors.keys()):
        len_x_ax = len(acc_plot_lines[side]['tap'])
        x_ax = np.arange(.5, .5 + len_x_ax)
        y_base_arr = np.array([base_y[side]] * len_x_ax)
        if plot_type == 'fill':
            # fill between y-base and tap-values
            ax.fill_between(x_ax,
                            y_base_arr,
                            acc_plot_lines[side]['tap'],
                            edgecolor=colors[side], alpha=.5,
                            facecolor='w', hatch='///',
                            label=f'Tap {side}',)
            # fill between tapvalues and move-values
            ax.fill_between(x_ax,
                            acc_plot_lines[side]['tap'],
                            acc_plot_lines[side]['move'],
                            color=colors[side], alpha=.2,
                            label=f'Other Move {side}',)
        elif plot_type == 'bars':
            w = 1
            h1 = np.array(acc_plot_counts[side]['tap'])
            h2= np.array(acc_plot_counts[side]['move'])
            ax.bar(x=x_ax, height=h1, bottom=y_base_arr,
                   width=w, align='center',
                   alpha=.5, edgecolor=colors[side],
                   hatch='///', facecolor='w',
                   label=f'Taps {side}',)
            # cap heights at maximum, to prevent plotting into other plots
            cap_idx = np.where(h1 + h2 > ymax)[0]
            if len(cap_idx) > 0:
                for i in cap_idx:
                    h2[i] = ymax - h1[i]  # all combis h1 + h2 are max ymax
                    # plot arrow on capped bars
                    ax.annotate('', xy=(x_ax[i], ymax * (n_s + 1)), 
                        xytext=(x_ax[i], ymax * (n_s + 1) - 2),
                        rotation=90, ha='center', va='center',
                        arrowprops=dict(arrowstyle="->"))

            ax.bar(x=x_ax, height=h2,
                bottom=h1 + y_base_arr,
                width=w, align='center',
                alpha=.2, color=colors[side],
                label=f'Other Moves {side}',)

    # PLOT JUMP IN TIME INDICATORS (gray line where temporal interruption is)
    y_top, y_bot = ax.get_xlim()
    for i_pre, x_t in enumerate(plot_times[1:]):
        # if epochs are more than 5 minutes separated
        if x_t - plot_times[i_pre] >= 120:
            ax.axvline(i_pre + 1, ymin=y_bot, ymax=y_top,
                        color='lightgray', lw=5, alpha=.8,)

    # set subplot settings
    if plot_type == 'fill':
        ax.set_ylabel('Movement (ACC)', size=fs + 6)
        ax.set_ylim(0, max(base_y.values()) + 1)  # 1 above the highest baseline
    elif plot_type == 'bars':
        ax.set_ylabel('Movement\n(n acc-detections)', size=fs + 6)
        ax.set_ylim(0, max(base_y.values()) * 2)  # twice max side-baseline
    ax.set_yticks(list(base_y.values()), size=fs)
    ax.set_yticklabels(list(base_y.keys()), size=fs)

    ax.set_xlim(0, len(plot_times))  # align time axis with main plot (based on time-freq values)
    xtickhop = 6
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    ax.set_xticks([])
    ax.set_xticklabels([])
    # ax.set_xticks(np.linspace(.5, len(plot_times) - .5, len(xticklabs)))
    # ax.set_xticklabels(np.around(xticklabs / 60, 1))
    ax.tick_params(axis='both', size=fs, labelsize=fs)
    for ax_side in ['left', 'right', 'top', 'bottom']:
        getattr(ax.spines, ax_side).set_visible(False)
    ncols = 4
    if plot_task:
        ax.fill_between(np.arange(.5, .5 + len_x_ax),
                        [min(base_y.values())] * len_x_ax,
                        [ax.get_ylim()[1]] * len_x_ax,
                        where=np.array(task_list) == 'tap',
                        facecolor='w', edgecolor='gray',
                        hatch='X', alpha=.3,)
        ncols += 1
    ax.legend(frameon=False, ncol=ncols, fontsize=fs + 2,
                bbox_to_anchor=(.5, -.1), loc='center',)


    return fig, axes

    