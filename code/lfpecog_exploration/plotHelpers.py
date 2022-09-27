"""
Function to assisst making pretty Plots
"""
# Import publoc packages and functions
# import numpy as np


def remove_duplicate_legend(
    legend_handles_labels
):
    """
    Input:
        - legend_handles_labels: output of
            plt.gca().get_legend_handles_labels()
    
    Returns:
        - handles, labels: without duplicates
    """
    handles = legend_handles_labels[0]
    labels = legend_handles_labels[1]
    
    singles = dict(zip(labels, handles))

    handles = singles.values()
    labels = singles.keys()

    return handles, labels


