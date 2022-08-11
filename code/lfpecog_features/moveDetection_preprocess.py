"""
Helper functions for paused tap and behavorial
state detection
"""

# Import public packages
import numpy as np
import pandas as pd
from scipy.stats import variation

# Import own functions

def find_task_blocks(
    subdat, task: str,
):
    """
    Finds the start and end indices of
    task periods in a subjectData()-
    dataframe.
    Takes only dataframe with one datatype.
    E.g. subData.acc_left, or subData.lfp_left

    - subdat: subjectData Class
    - task: str name of task [free/tap/rests]
    """
    list_of_tasks = ['tap', 'rest', 'free']
    assert task in list_of_tasks, print(
        f'Given task "{task} is not in {list_of_tasks}'
    )
    taskdat = subdat['task'].values

    iStart, iEnd = [], []
    
    state = 'NO'

    for i in subdat.index:
        
        if state == 'NO':
            
            if taskdat[i] == task:

                iStart.append(i)
                state = 'YES'
                
        elif state == 'YES':

            if taskdat[i] != task:

                iEnd.append(i)
                state = 'NO'
    
    if state == 'YES': iEnd.append(i)

    return [iStart, iEnd]


def find_main_axis(dat_arr):
    """
    Select acc-axis which recorded tapping the most

    Input:
        - dat_arr (arr): triaxial acc signals
    Returns:
        - main_ax_index (int): [0, 1, or 2], axis
            with most tapping activity detected
    """
    if dat_arr.shape[0] > dat_arr.shape[1]:
        dat_arr = dat_arr.T

    # ax_vars = [variation(dat_arr[ax, :]) for ax in [0, 1, 2]]
    ax_prcts = [np.percentile(dat_arr[ax, :], 99) for ax in [0, 1, 2]]

    # main_ax_index = np.argmax(abs(np.array(ax_vars)))
    main_ax_index = np.argmax(abs(np.array(ax_prcts)))

    return main_ax_index


def signalvectormagn(acc_arr):
    """
    Input:
        - acc_arr (array): triaxial array
            with x-y-z axes (3 x n_samples)
    
    Returns:
        - svm (array): uniaxial array wih
            signal vector magn (1, n_samples)
    """
    if acc_arr.shape[0] != 3: acc_arr = acc_arr.T
    assert acc_arr.shape[0] == 3, ('Array must'
    'be tri-axial (x, y, z from accelerometer')
  
    svm = np.sqrt(
        acc_arr[0] ** 2 +
        acc_arr[1] ** 2 +
        acc_arr[2] ** 2
    )

    return svm