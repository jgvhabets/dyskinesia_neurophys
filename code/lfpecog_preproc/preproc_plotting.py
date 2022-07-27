"""
Function to plot preprocessing progress
in channels versus time
"""
from curses.ascii import FS
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

def plot_groupChannels(
    ch_names: list, groupData, groupName:str,
    Fs, runInfo, moment:str
):
    moment_strings = {
        'raw': {
            'title': f'Raw {groupName.upper()} signals ({Fs} Hz) BEFORE preprocessing',
            'fname': 'after_raw_import'
        },
        'post-preprocess': {
            'title': f'{groupName.upper()} signals ({Fs} Hz) AFTER preprocessing',
            'fname': 'after_preprocessing',
        },
        'post-artefact-removal': {
            'title': f'{groupName.upper()} signals ({Fs} Hz) AFTER artefact removal',
            'fname': 'after_artfct_removal',
        }
    }
    fig, axes = plt.subplots(
        len(ch_names) - 2, 1, figsize=(16, 24),
        sharex=True,
    )

    print(f'channel names in plotting {runInfo.store_str}: {ch_names}')
    print(f'\tMOMENT {moment}; plot time rows: {groupData[:2, :5]}')
    for n in range(len(ch_names) - 2):

        axes[n].plot(
            groupData[1, :],
            groupData[n + 2, :]
        )
        # axes[n].set_xticks(groupData)
        axes[n].set_ylabel(
            ch_names[n + 2], fontsize=18, rotation=45,
        )
    
    min_LDtime = groupData[1, :] / 60
    axes[-1].set_xticks(np.linspace(
        groupData[1, 0], groupData[1, -1], 5
    ))
    axes[-1].set_xticklabels(np.around(np.linspace(
        min_LDtime[0], min_LDtime[-1], 5
    ), 1), fontsize=18)
    axes[-1].set_xlabel(
        f'{ch_names[1]} (in minutes after LT)', fontsize=18,
    )

    fig.suptitle(
        f'{moment_strings[moment]["title"]} ({runInfo.store_str})',
        size=20, x=.5, y=.95, ha='center',
    )

    fname = f'{runInfo.store_str}_{groupName}_{moment_strings[moment]["fname"]}'
    plt.savefig(
        join(runInfo.fig_path, fname),
        dpi=150, facecolor='w',
    )
    plt.close()


def dict_plotting(
    dataDict: dict, Fs_dict:dict,
    chNameDict: dict, runInfo, moment,
):
    """
    Calls plotting function for all groups
    in given dictionaries
    """
    for group in dataDict:

        plot_groupChannels(
            ch_names=chNameDict[group],
            groupData=dataDict[group],
            Fs=Fs_dict[group],
            groupName=group,
            runInfo=runInfo,
            moment=moment,
        )