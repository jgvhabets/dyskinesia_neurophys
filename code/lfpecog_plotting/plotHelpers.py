"""
Function to assisst making pretty Plots
"""
# Import publoc packages and functions
# import numpy as np

from os.path import join
import json

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