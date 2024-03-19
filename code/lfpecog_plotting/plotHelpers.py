"""
Function to assisst making pretty Plots
"""
# Import publoc packages and functions
# import numpy as np

from os.path import join
import json
from numpy.random import uniform


def get_plot_jitter(x_temp, y_temp, jit_width=.5, ZERO_SPACE=True):
    """
    if ZERO_SPACE: x_temp 0's are shifted to -1's
    """
    # jitter X, CDRS scores
    x_jitter = uniform(
        low=-jit_width, high=jit_width, size=len(x_temp)
    )
    # move to -1 x-axis and increase jitter for no dyskinesia
    if ZERO_SPACE:
        nolid_sel = x_temp == 0  # select current samples with CDRS == 0
        x_temp[nolid_sel] -= 1   # set base at -1 on x-axis
        x_jitter[nolid_sel] = uniform(
            low=-jit_width * 3, high=jit_width * 3, size=sum(nolid_sel)
        )  # double the jitter
    
    # jitter Y, movement z-scores
    y_jitter = uniform(
        low=-jit_width, high=jit_width, size=len(y_temp)
    )

    return x_jitter, y_jitter


def remove_duplicate_legend(
    legend_handles_labels
):
    """
    Input:
        - legend_handles_labels: output of
            plt.gca().get_legend_handles_labels()
    
    Returns:
        - handles, labels: without duplicates
            use as: plt.legend(handles, labels)
    
    Example:
        plt.legend(handles, labels, ncol=4, frameon=False,
                   loc='upper center', bbox_to_anchor=(0.5, -0.2),
                   fancybox=False,)
    """
    handles = legend_handles_labels[0]
    labels = legend_handles_labels[1]
    
    singles = dict(zip(labels, handles))

    handles = singles.values()
    labels = singles.keys()

    return handles, labels


def convert_ft_names(names, find: str='delta',
                     replace: str='theta'):

    new_names = []

    for n in names:
        if find in n:
            new_names.append(n.replace(find, replace))
        else:
            new_names.append(n)
    
    return new_names


def get_colors(scheme='PaulTol'):
    """
    if scheme is 'access_colors_PaulTol' Paul Tol's
        colorscheme for accessible colors is returned,
    if scheme is 'contrast_duo', two contrasting colors
        are returned
    
        colors:
        {'nightblue': '#332288',
        'darkgreen': '#117733',
        'turquoise': '#44AA99',
        'lightblue': '#88CCEE',
        'sand': '#DDCC77',
        'softred': '#CC6677',
        'lila': '#AA4499',
        'purplered': '#882255'}
    """
    assert scheme in ['PaulTol', 'Jacoba'], (
        'scheme entry should be "PaulTol", or "Jacoba"'
    )
    cmap_json = join('lfpecog_plotting',
                     'color_schemes.json')

    with open(cmap_json, 'r') as json_data:
        schemes = json.load(json_data)

    cmap = schemes[scheme]

    return cmap