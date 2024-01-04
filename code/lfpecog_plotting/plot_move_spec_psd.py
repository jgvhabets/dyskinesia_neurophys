"""
Plots PSDs that are millisecond precise
on movement selection
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from utils.utils_fileManagement import get_project_path
from lfpecog_plotting.plot_descriptive_SSD_PSDs import break_x_axis_psds_ticks
from lfpecog_analysis.specific_ephys_selection import get_hemisphere_movement_location

cond_colors = {
    'nolid': 'green',
    'nolidbelow30': 'limegreen',
    'nolidover30': 'darkgreen',
    'alllid': 'blue', 'mildlid': 'orange',
    'moderatelid': 'red', 'severelid': 'purple',
}

def plot_moveLidSpec_PSDs(
    psd_arrs, psd_freqs, PLOT_MOVE_TYPE,
    SAVE_PLOT: bool = True,

):
    """
    Plots first keys as rows (LFP-ECoG), second keys as cols (CONTRA/IPSI lat)
    """
    # add extra dimension for rest (no movement differences on columns)
    if PLOT_MOVE_TYPE == 'REST':
        psd_arrs = {'rest': psd_arrs}

    # LFP-row0, ECoG-row1; IPSI/CONTRA/BOTH in columns
    ax_row_keys = list(psd_arrs.keys())
    ax_col_keys = list(psd_arrs[ax_row_keys[0]].keys())

    fig, axes = plt.subplots(len(ax_row_keys), len(ax_col_keys),
                             figsize=(len(ax_col_keys)*8, len(ax_row_keys)*6),
                             sharey='row', sharex='col',)
    
    if len(axes.shape) == 1: axes = np.atleast_2d(axes)  # make axes array 2d (i.e., 3x1)

    XTICKS = [4, 12, 20, 28, 60, 70, 80, 89]  # 89 will be labeled 90
    ls = 'solid'

    for axrow, src in enumerate(ax_row_keys):

        for axcol, mov in enumerate(ax_col_keys):
            
            for lid in psd_arrs[src][mov].keys():
                # get and average correct PSDs
                temp_psds = np.array(psd_arrs[src][mov][lid])
                n_subs = len(temp_psds)
                if n_subs == 0:
                    print(f'no psds for {src, mov, lid}')
                    continue
                m = np.mean(temp_psds, axis=0)
                sd = np.std(temp_psds, axis=0)
                sem = np.std(temp_psds, axis=0) / np.sqrt(n_subs)
                
                # break and nan-pad x-axis
                m, sem, x_plot, xlabs = break_x_axis_psds_ticks(
                    tf_freqs=psd_freqs,
                    PSD=m, PSD_sd=sem,
                    x_break = (35, 60), nan_pad = 5
                )
                axes[axrow, axcol].plot(
                    x_plot, m, # PSDs[cond].freqs, m,
                    color=cond_colors[lid], alpha=.8, ls=ls,
                    label=f'{lid} (n={n_subs})',

                )
                axes[axrow, axcol].fill_between(
                    x_plot, y1=m-sem, y2=m+sem, # PSDs[cond].freqs, 
                    color=cond_colors[lid], alpha=.3,
                )
                # add title
                if PLOT_MOVE_TYPE != 'REST':
                    axes[axrow, axcol].set_title(
                        f'{src.upper()}: {mov}-lateral', weight='bold',
                    )
                else:
                    axes[axrow, axcol].set_title(
                        f'{mov.upper()} during REST (no movement)', weight='bold',
                    )

    xtick_sel = np.where([x in XTICKS for x in xlabs])[0]
    xticks = x_plot[xtick_sel]
    xlabs = np.array(xlabs)[xtick_sel]
    xlabs[-1] = 90  # mark last tick with 90 (instead of 89)

    for i_ax, ax in enumerate(axes.ravel()):
        ax.axhline(0, xmin=0, xmax=1, color='gray', alpha=.3,)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs)
        ax.set_ylim(-75, 275)
        ax.legend()
    # set axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Spectral Power, %-change\n'
                    '(vs Med-OFF, no LID)')

    plt.tight_layout()

    if SAVE_PLOT:
        if PLOT_MOVE_TYPE == 'REST':
            FIG_NAME = f'PSDs_{PLOT_MOVE_TYPE}_lidCategs'
        else:
            FIG_NAME = f'PSDs_{PLOT_MOVE_TYPE}contraIpsi_lidCategs'
        FIG_PATH = os.path.join(get_project_path('figures'),
                                'ft_exploration',
                                'data_v4.0_ft_v6',
                                'PSDs_state_specific')
        
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)

    plt.show()


def prep_MOVEMENT_spec_psds(PLOT_MOVE, PSD_DICT, BASELINE):
    """

    Arguments:
        - PLOT_MOVE: TAP or INVONUT
        - PSD_DICT: dict with conditions as keys and containing
            get_selectedEphys() classes per condition (from
            psd_analysis_classes.py)
        - BASELINE: get_selectedEphys() class

    Returns:
        - psd_arrs: dicts of lists with selected and processed
            psd-arrays, organized in LFP/ECoG (axrows), and
            CONTRA/IPSI-lateral movements (axcols)
    """
    assert PLOT_MOVE in ['INVOLUNT', 'TAP']

    if PLOT_MOVE == 'TAP': ATTR_CODE = 'TAP'
    elif PLOT_MOVE == 'INVOLUNT': ATTR_CODE = 'DYSKMOVE'

    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['no', 'mild', 'moderate', 'severe']
    move_axes = [f'{PLOT_MOVE} CONTRA', f'{PLOT_MOVE} IPSI']
    if PLOT_MOVE == 'INVOLUNT': move_axes.append('INVOLUNT BILAT')

    psd_arrs = {'lfp': {d: [] for d in move_axes},
                'ecog': {d: [] for d in move_axes}}  # store all psds for move axes
    
    # replace single large lists for dicts with lid-states
    for s, m in product(psd_arrs.keys(), move_axes):
        psd_arrs[s][m] = {f'{l}lid': [] for l in lid_states}


    # categorize, average, baseline-corr all conditions
    for SRC, cond in product(sources, PSD_DICT.keys()):
        
        if not ATTR_CODE in cond.upper(): continue
        # loop and add subjects
        for attr in vars(PSD_DICT[cond]).keys():
            # skip irelevant attr
            if not ATTR_CODE in attr.upper(): continue
            if SRC not in attr: continue
            sub = attr.split(f'{SRC}_')[1][:3]
            if not sub.startswith('0') and not sub.startswith('1'): continue

            # get psd arr and correct for baseline
            temp_psd = getattr(PSD_DICT[cond], attr)
            try:
                bl = getattr(BASELINE, f'{SRC}_{sub}_baseline')
            except:
                print(f'### WARNING: no baseline found for {sub}: {SRC}, skipped')
                continue
            temp_psd = ((temp_psd - bl) / bl) * 100

            # add to LFP or ECOG and correct move-dict
                # define hemisphere location to movement
            HEM_LOC, MOV_TYPE = get_hemisphere_movement_location(
                SRC=SRC, cond=cond, sub=sub,
                ecog_sides=PSD_DICT[cond].ecog_sides,
            )
            # psd_arrs[SRC.split('_')[0]][f'{MOV_TYPE} {HEM_LOC}'].append(temp_psd)
            # include LID-state:
            temp_lid = attr.split('_')[-1]
            psd_arrs[
                SRC.split('_')[0]][f'{MOV_TYPE} {HEM_LOC}'
            ][temp_lid].append(temp_psd)
            
            # print(f'added {attr} to {SRC}, {MOV_TYPE} {HEM_LOC} {temp_lid}')

    psd_freqs = PSD_DICT[cond].freqs

    return psd_arrs, psd_freqs


def prep_REST_spec_psds(PSD_DICT, BASELINE,
                        PLOT_MOVE: str = 'REST'):
    """

    Arguments:
        - PSD_DICT: dict with conditions as keys and containing
            get_selectedEphys() classes per condition (from
            psd_analysis_classes.py)
        - BASELINE: get_selectedEphys() class

    Returns:
        - psd_arrs: dicts of lists with selected and processed
            psd-arrays, organized in LFP/ECoG (axrows), and
            CONTRA/IPSI-lateral movements (axcols)
    """
    
    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['nolidbelow30', 'nolidover30',
                  'mildlid', 'moderatelid', 'severelid']

    psd_arrs = {s: {l: [] for l in lid_states} for s in sources}  # store all psds for move axes
    

    # categorize, average, baseline-corr all conditions
    for SRC, cond in product(sources, PSD_DICT.keys()):        
        # loop and add subjects
        for attr in vars(PSD_DICT[cond]).keys():
            # skip irelevant attr
            if SRC not in attr or 'rest' not in attr.lower(): continue
            sub = attr.split(f'{SRC}_')[1][:3]
            if not sub.startswith('0') and not sub.startswith('1'): continue

            # get psd arr and correct for baseline
            temp_psd = getattr(PSD_DICT[cond], attr)
            try:
                bl = getattr(BASELINE, f'{SRC}_{sub}_baseline')
            except:
                print(f'### WARNING: no baseline found for {sub}: {SRC}, skipped')
                continue
            temp_psd = ((temp_psd - bl) / bl) * 100

            # include LID-state:
            temp_lid = attr.split('_')[-1]
            psd_arrs[SRC][temp_lid].append(temp_psd)
            
            # print(f'added {attr} to {SRC}, {temp_lid}')
    
    psd_freqs = PSD_DICT[cond].freqs
    
    return psd_arrs, psd_freqs