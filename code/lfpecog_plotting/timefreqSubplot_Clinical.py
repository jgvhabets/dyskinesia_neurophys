"""
Subplot with Clinical data
for plotting features over time
"""

# import publick functions
import numpy as np
import matplotlib.pyplot as plt

# import own functions
import lfpecog_preproc.preproc_import_scores_annotations as importClin


def subplot_cdrs(
    fig, axes, fs, sub, plot_times, i_plot_ax=0,
    colors = {'total': 'darkblue', 'left': 'lightblue', 'right': 'purple'}
):
    """
    Create subplot with Levodopa Induced Dyskinesia scores
    rated with the Clinical Dyskinesia Rating Scale
    
    Inputs:
        - fig, axes: from main plot
        - fs: fontsize
        - sub: subject code e.g. '001'
        - plot_times: array with times of main
            TimeFreq plot, given in seconds
        - i_plot_ax: CDRS subplot default first    
    
    Returns:
        - fig, axes with added subplot
    """
    ax = axes[i_plot_ax]
    ax.set_xlim(0, len(plot_times))  # align time axis with main plot (based on time-freq values)
    
    # Plot CDRS every 10 minutes
    try:
        scores, _, _ = importClin.run_import_clinInfo(sub=sub)
        # check if scores are present
        if type(scores) == type(None):
            print(f'None CDRS-scores loaded for sub {sub}')
            return fig, axes

        # get and plot CDRS values (scores in min, plot_times in sec)
        y_scores = {}
        y_scores['total'] = scores['CDRS_total']
        y_scores['left'] = scores['CDRS_total_left']
        y_scores['right'] = scores['CDRS_total_right']
        x_times = [np.argmin(abs(m - plot_times))
                   for m in scores['dopa_time'] * 60]
        for k in y_scores.keys():
            ax.plot(
                x_times, y_scores[k],
                marker='o', alpha=.6,
                color=colors[k], lw=5, label=f'{k} sum')
        
    except FileNotFoundError:
        print(f'No clin scores found for sub {sub}')
    
    # PLOT LID-timings (observed Start and Peak moments)
    try:
        lid_timings = importClin.get_seconds_of_LID_start()[sub]
        lid_clrs = {'start': 'green', 'peak': 'orange'}
        for timing in lid_clrs:
            lid_t = getattr(lid_timings, f"t_{timing}")
            # print(timing, lid_i, lid_t)
            lid_i = np.argmin(abs(plot_times - lid_t))
            ax.axvline(lid_i, ymin=0, ymax=20, ls='--', lw=5,
                    color=lid_clrs[timing], alpha=.5,
                    label=timing,)
    except AttributeError:
        print(f'\tSub-{sub} LID timings not available (in mvc-plotting)')


    # PLOT JUMP IN TIME INDICATORS (gray line where temporal interruption is)
    for i_pre, x_t in enumerate(plot_times[1:]):
        # if epochs are more than 5 minutes separated
        if x_t - plot_times[i_pre] >= 120:
            ax.axvline(i_pre + 1, ymin=0, ymax=20,
                       color='lightgray', lw=5, alpha=.8,)

    # set subplot settings
    ax.set_ylim(0, 10)
    ax.set_ylabel('LID (CDRS)', size=fs + 6)
    ax.set_yticks(np.arange(0, 21, 5), size=fs)
    ax.set_yticklabels(np.arange(0, 21, 5), size=fs)
    xtickhop = 6
    xticklabs = np.array(plot_times[::xtickhop], dtype=float)
    ax.set_xticks([])
    ax.set_xticklabels([])
    # ax.set_xticks(np.linspace(.5, len(plot_times) - .5, len(xticklabs)))
    # ax.set_xticklabels(np.around(xticklabs / 60, 1))
    ax.tick_params(axis='both', size=fs, labelsize=fs)
    for side in ['right', 'top', 'bottom']:
        getattr(ax.spines, side).set_visible(False)

    # set legend
    ax.legend(frameon=False, ncol=3, fontsize=fs,
              bbox_to_anchor=[.5, 0], loc='upper center',
    )

    return fig, axes

