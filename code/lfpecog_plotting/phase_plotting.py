"""
Phase plotting functions
"""

# import packages and functions
import matplotlib.pyplot as plt
import numpy as np


def plot_rose_axis(
    radians, bin_size_degr = 20,
    n_bins=None,
    polar_ax=None, 

):
    """
    Inputs:
        - ax: axis to plot on
        - radians: list or array with radians to plot,
            if given as degrees, function tries to correct
            and warn for this
        - bin_size_degree: bin size in degrees, defaults to 20
        - n_bins: number of bins. Default this is the
            result of bin_size_degr. Only when n_bins
            is defined, this variable is used.
        - polar_ax: if given, this subplot has to be polar
            (projection='polar'), or polar=True. If non-
            polar ax is given an AssertionError is given.
            If not given: the rose plot will be plotted
            in separate plot
            Ex.:
            ax1 = plt.subplot(221)
            ax1.plot(sig)
            ax2 = plt.subplot(222, projection='polar')
            ax2 = plot_rose_axis(radians=rad_diff, polar_ax=ax2)
            plt.tight_layout()
            plt.show()
    """
    if not np.logical_and(radians >= 0,
                          radians <= 2 * np.pi).all():
        print('radians assumed to be in degrees and corrected')
        radians = np.deg2rad(radians)

    if isinstance(n_bins, float): n_bins = int(n_bins)
    elif isinstance(n_bins, int): n_bins = n_bins
    else: n_bins = int(360 / bin_size_degr)

    if polar_ax:
        assert "PolarAxesSubplot" in str(type(polar_ax)), (
            "given axis 'polar_ax' has to be 'projection='polar''"
        )
        ax = polar_ax
    else:
        plt.figure(figsize=(4,4))
        ax = plt.subplot(1, 1, 1, projection='polar')

    bins = np.linspace(0.0, 2 * np.pi, n_bins + 1)
    counts, bin_edges = np.histogram(radians, bins=bins)

    width = 2 * np.pi / n_bins
    bars = ax.bar(bins[:n_bins], counts, width=width, bottom=0.0)
    for bar in bars:
        bar.set_alpha(0.7)
    max_c = max(counts)
    ticks = np.linspace(0, max_c, 5)
    plt.yticks(ticks, labels=[])

    if polar_ax: return ax
    else:
        plt.show()


def plot_pac_comodulogram(
    pac_arr_2d,
    pha_bin_tuples,
    amp_bin_tuples,
    fsize=14,
    plot_title=None,
    plot_on_ax=None,
    fig_axis=None,
    vmax=None
):
    if plot_on_ax:
        ax = plot_on_ax
        fig = fig_axis
    else: fig, ax = plt.subplots(1,1, figsize=(6, 6))

    im = ax.imshow(pac_arr_2d, cmap='Reds',
                   vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax)


    ticks, ticklabels = get_pac_bins_ticks(
        pha_bin_tuples, n_ticks=3)
    ax.set_xticks(ticks,)
    ax.set_xticklabels(ticklabels, rotation=0)
    ticks, ticklabels = get_pac_bins_ticks(
        amp_bin_tuples, n_ticks=5)
    ax.set_yticks(ticks,)
    ax.set_yticklabels(ticklabels,)

    ax.set_xlabel('Phase freq. (Hz)', size=fsize)
    ax.set_ylabel('Ampltitude freq. (Hz)', size=fsize)

    ax.invert_yaxis()

    if plot_title: ax.set_title(plot_title, size=fsize,
                             weight='bold',)

    ax.tick_params(axis='both', size=fsize, labelsize=fsize)
    
    if plot_on_ax:
        return ax
    else:
        plt.tight_layout()
        plt.show()


def get_pac_bins_ticks(pac_bins, n_ticks=3):

    ticks = np.linspace(0, len(pac_bins) - 1, n_ticks)

    ticklabels = np.linspace(pac_bins[0][0],
                             pac_bins[-1][1],
                             n_ticks)
    
    return ticks, ticklabels.astype(int)