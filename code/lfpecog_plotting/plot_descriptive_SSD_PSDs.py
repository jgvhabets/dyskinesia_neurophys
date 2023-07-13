"""
Plot overview PSDs for Paper
"""

# import public functions
from os.path import join
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
import matplotlib as mpl

# import own functions





def plot_PSD_vs_DopaTime(all_timefreqs,
                         sel_subs=None,
                         LOG_POWER=True,
                         ZSCORE_FREQS=True,
                         SMOOTH_PLOT_FREQS=0,
                         BASELINE_CORRECT=False,
                         BREAK_X_AX=False,
                         plt_ax_to_return=False,
                         fsize=12,):
    """
    Plot group-level PSDs (based on SSD data), plot
    mean PSDs for temporal course after LDopa-intake.

    Input:
        - all_timefreqs: (results from ssd_TimeFreq.get_all_ssd_timeFreqs)
            contains tf_values (shape n-freq x n-times), times, freqs.
        - sel_subs: if given, selection of subjects to include
        - LOG_POWER: plot powers logarithmic
        - ZSCORE_FREQS: plot PSDs as z-Scores per freq-bin
        - SMOOTH_PLOT_FREQS: if > 0, window (n freqs) to
            smoothen PSDs values
        - BASELINE_CORRECT: plot difference of time-windows
            versus pre-Dopa-intake
        - BREAK_X_AX: break x-axis between beta and gamma
        - plt_ax_to_return: if plotted as subplot in another plot,
            give the defined axis here
        - fsize: fontsize, defaults to 12
    """
    timings = np.arange(0, 76, 15)  # create timings between 0 and 61/76 with 10/15 min steps
    psds_to_plot = {t: [] for t in timings}

    # first timing 0, is regharded as < 0
    # last timing (70) is regarded as > 60
    if sel_subs: subs = [sub for sub in all_timefreqs.keys() if sub in sel_subs]
    else: subs = all_timefreqs.keys()
    
    for sub in subs:

        for src in all_timefreqs[sub].keys():

            if 'ecog' in src: continue
            # get data for this sub and ephys-source
            tf_values = all_timefreqs[sub][src].values
            if LOG_POWER: tf_values = np.log(tf_values)
            if ZSCORE_FREQS:
                for f in np.arange(tf_values.shape[0]):
                    tf_values[f] = (tf_values[f] - np.mean(tf_values[f])
                                 ) / np.std(tf_values[f])

            tf_times = all_timefreqs[sub][src].times / 60
            tf_freqs = all_timefreqs[sub][src].freqs

            for timing in timings:

                if timing == 0:
                    sel = tf_times < 0
                elif timing == timings[-1]:    
                    sel = tf_times > timing
                else:
                    sel = np.logical_and(tf_times > timing - 10,
                                        tf_times < timing)
                if sum(sel) == 0: continue

                mean_psd = list(np.mean(tf_values[:, sel], axis=1))
                psds_to_plot[timing].append(mean_psd)

    if not plt_ax_to_return:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = plt_ax_to_return
    
    cmap = mpl.cm.get_cmap('winter')  # 'winter' / 'gist_yarg', 'cool
    gradient_colors = np.linspace(0, 1, len(timings))
    gradient_colors = [cmap(g) for g in gradient_colors]
    
    if BASELINE_CORRECT:
        gradient_colors[1:len(timings)] = [cmap(g) for g in np.linspace(0, 1, len(timings)-1)]
    
    for i, timing in enumerate(timings):
        psds = np.array(psds_to_plot[timing])
        psd_mean = np.mean(psds, axis=0)
        # blank freqs irrelevant after SSD
        blank_sel = np.logical_and(tf_freqs > 35, tf_freqs < 60)
        psd_mean[blank_sel] = np.nan
        # smoothen signal for plot
        if SMOOTH_PLOT_FREQS > 0:
            psd_mean = Series(psd_mean).rolling(
                window=SMOOTH_PLOT_FREQS, center=True
            ).mean().values

        if BASELINE_CORRECT and i==0:
            BL = psd_mean
            continue
        elif BASELINE_CORRECT:
            psd_mean = psd_mean - BL

        # adjust label
        if timing == 0:
            label = 'pre L-DOPA intake'
        elif timing == timings[-1]:
            label = f'post {timings[-2]} min'
        else:
            label = f'{timings[i-1]} - {timing} min'
        
        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            x_break = (30, 60)
            nan_pad = 5
            del_sel = np.logical_and(tf_freqs > x_break[0],
                                     tf_freqs < x_break[1])
            del_sel = np.logical_or(del_sel, np.isnan(psd_mean))
            psd_mean = np.delete(psd_mean, del_sel,)
            plt_freqs = np.delete(tf_freqs.copy(), del_sel,).astype(float)

            i_sel = np.argmin(abs(plt_freqs - x_break[0]))

            psd_mean = np.insert(psd_mean, i_sel + 1,
                                 values=[np.nan] * nan_pad,)
            plt_freqs = np.insert(plt_freqs, i_sel + 1,
                                  values=[np.nan] * nan_pad,)

            xticks = np.arange(len(psd_mean))
            xlabels = [''] * len(xticks)
            low_ticks = plt_freqs[plt_freqs < x_break[0]]
            xlabels[:len(low_ticks)] = low_ticks
            high_ticks = plt_freqs[plt_freqs > x_break[1]]
            xlabels[len(xlabels) - len(high_ticks):] = high_ticks

        if not BREAK_X_AX: x_axis = tf_freqs
        elif BREAK_X_AX: x_axis = xticks

        # PLOT LINE
        ax.plot(x_axis, psd_mean, label=label,
                color=gradient_colors[i],
                lw=5, alpha=.5,)

    if BREAK_X_AX:
        ax.set_xticks(xticks[::8], size=fsize)
        ax.set_xticklabels(xlabels[::8], fontsize=fsize)
        if SMOOTH_PLOT_FREQS <= 0: yfill = [.4, -.6]
        elif SMOOTH_PLOT_FREQS <= 8: yfill = [.15, -.3]
        elif SMOOTH_PLOT_FREQS <= 10: yfill = [.1, -.25]
        ax.fill_betweenx(y=yfill, x1=i_sel, x2=i_sel+nan_pad,
                         facecolor='gray', edgecolor='gray', alpha=.3,)
    else:
        ax.set_xticks(np.linspace(x_axis[0], x_axis[-1], 5))
        ax.set_xticklabels(np.linspace(x_axis[0], x_axis[-1], 5))

    ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
              color='gray', lw=1, alpha=.5,)
    
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Power (a.u.)'
    if BASELINE_CORRECT: ylabel = 'Power change vs pre - L-DOPA' + ylabel[5:]
    ax.set_ylabel(ylabel, size=fsize,)
    ax.legend(frameon=False, fontsize=fsize, ncol=2)

    if plt_ax_to_return: return ax
    else: plt.show()


