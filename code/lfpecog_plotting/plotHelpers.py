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


