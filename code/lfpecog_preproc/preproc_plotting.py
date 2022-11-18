"""
Function to plot preprocessing progress
in channels versus time
"""
from itertools import compress
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs



def plot_groupChannels(
    ch_names: list, groupData, groupName:str,
    Fs, runInfo, moment: str, artf_data=None,
):
    moment_strings = {
        'raw': {
            'title': f'Raw {groupName.upper()} signals ({Fs} Hz) BEFORE preprocessing',
            'fname': 'post_raw_import'
        },
        'pre-artefact-removal': {
            'title': f'{groupName.upper()} signals ({Fs} Hz) BEFORE artefact removal',
            'fname': 'pre_artfct_removal',
        },
        'post-artefact-removal': {
            'title': f'{groupName.upper()} signals ({Fs} Hz) AFTER artefact removal',
            'fname': 'post_artfct_removal',
        },
        'post-reref': {
            'title': f'{groupName.upper()} signals ({Fs} Hz) AFTER preprocessing',
            'fname': 'post_rereferencing',
        }
    }

    save_path = join(runInfo.fig_path, runInfo.store_str)
    if not exists(save_path): makedirs(save_path)
        
    timerow_sel = ['time' in name for name in ch_names]
    timerowNames = list(compress(ch_names, timerow_sel))
    nTimeRows = len(timerowNames)
    x_ind = ch_names.index('dopa_time')

    fig, axes = plt.subplots(
        len(ch_names) - nTimeRows, 1, figsize=(16, 24),
        sharex=True,
    )

    for n in range(len(ch_names) - nTimeRows):

        axes[n].plot(
            groupData[x_ind, :],
            groupData[n + nTimeRows, :]
        )

        if moment == 'post-artefact-removal':

            y1, y2 = axes[n].get_ylim()
            axes[n].fill_between(
                x=groupData[x_ind, :],
                y1=y1, y2=y2,
                where=np.isnan(groupData[n + nTimeRows, :]),
                color='red',
                alpha=.4,
                label='Artefact cleaned',
            )

            try:
                artf_data.shape

                axes[n].plot(
                    artf_data[x_ind, :],
                    artf_data[n + nTimeRows, :],
                    color='k', alpha=.6,
                    label=(
                        'Removed as "artefact" (# StdDev cutoff: '
                        f'{runInfo.mainSettings["ephys"]["artf_sd"]})'
                    ),
                )
            except:
                print('No artefact to plot next to clean data')
            
            axes[n].set_ylim(y1, y2)

        axes[n].set_ylabel(
            ch_names[n + 2], fontsize=18, rotation=30,
        )
    
    axes[-1].legend(
        bbox_to_anchor=[.1, -.15],
        loc='upper left', frameon=False,
        fontsize=20, ncol=2,
    )

    min_LDtime = groupData[x_ind, :] / 60
    axes[-1].set_xticks(np.linspace(
        groupData[x_ind, 0], groupData[x_ind, -1], 5
    ))
    axes[-1].set_xticklabels(np.around(np.linspace(
        min_LDtime[0], min_LDtime[-1], 5
    ), 1), fontsize=18)
    axes[-1].set_xlabel(
        f'{ch_names[x_ind]} (in minutes after LT)', fontsize=18,
    )

    fig.suptitle(
        f'{moment_strings[moment]["title"]}\n({runInfo.store_str})',
        size=20, x=.5, y=.95, ha='center',
    )

    fname = (f'{runInfo.store_str}_{groupName}_'
             f'{moment_strings[moment]["fname"]}_v3')
    print(f'\n\n\tFIGURE TO BE SAVED: {fname}')
    plt.savefig(
        join(save_path, fname),
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