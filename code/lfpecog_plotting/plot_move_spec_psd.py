"""
Plots PSDs that are millisecond precise
on movement selection
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from pandas import read_csv

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


def prep_and_plot_moveSpecPsd(
    PSD_DICT, BASELINE, PLOT_CONDITION,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    INCL_STATS: bool = False,
    STAT_PER_LID_CAT: bool = True,
    ALPHA: float = 0.05,
    ALT_MOVELID_BASELINE: bool = False,
    STATS_VERSION: str = '',
    STAT_LID_COMPARE: str = 'binary',
    MERGE_REST_STNS: bool = False,
    MERGE_FREE_STNS: bool = False,
    MERGED_DYSK: bool = False,
    ADD_TO_FIG_NAME = False,
):
    # define correct prep function
    if PLOT_CONDITION == 'REST':
        prep_func = prep_REST_spec_psds
        MERGE_STN = MERGE_REST_STNS
    elif PLOT_CONDITION in ['TAP', 'INVOLUNT', 'FREE']:
        prep_func = prep_MOVEMENT_spec_psds
        if PLOT_CONDITION == 'TAP':
            MERGE_STN = False
        elif PLOT_CONDITION == 'FREE':
            MERGE_STN = MERGE_FREE_STNS
            PLOT_CONDITION = ['FREENOMOVE', 'FREEMOVE']
        elif PLOT_CONDITION == 'INVOLUNT':
            MERGE_STN = MERGED_DYSK  # new plotting dysk_move
   
    # execute prep function
    if isinstance(PLOT_CONDITION, str):
        psd_arrs, psd_freqs, psd_subs = prep_func(
            PLOT_MOVE=PLOT_CONDITION,
            PSD_DICT=PSD_DICT,
            BASELINE=BASELINE,
            MERGED_STNS=MERGE_STN,
            RETURN_IDS=True,
        )
    elif isinstance(PLOT_CONDITION, list):
        psd_arrs, psd_subs = {}, {}
        for FREE_COND in PLOT_CONDITION:
            temp_arrs, psd_freqs, temp_subs = prep_func(
                PLOT_MOVE=FREE_COND,
                PSD_DICT=PSD_DICT,
                BASELINE=BASELINE,
                MERGED_STNS=MERGE_STN,
                RETURN_IDS=True,
            )
            psd_arrs[FREE_COND] = temp_arrs
            psd_subs[FREE_COND] = temp_subs

    # plot
    plot_moveLidSpec_PSDs(
        psd_arrs, psd_freqs=psd_freqs, psd_subs=psd_subs,
        PLOT_MOVE_TYPE=PLOT_CONDITION,
        SAVE_PLOT=SAVE_PLOT,
        SHOW_PLOT=SHOW_PLOT,
        INCL_STATS=INCL_STATS,
        STAT_PER_LID_CAT=STAT_PER_LID_CAT,
        ALT_MOVELID_BASELINE=ALT_MOVELID_BASELINE,
        ALPHA=ALPHA,
        STATS_VERSION=STATS_VERSION,
        STAT_LID_COMPARE=STAT_LID_COMPARE,
        MERGED_STNS=MERGE_STN,
        ADD_TO_FIG_NAME=ADD_TO_FIG_NAME,
    )


def plot_moveLidSpec_PSDs(
    psd_arrs, psd_freqs, psd_subs, PLOT_MOVE_TYPE,
    SAVE_PLOT: bool = False,
    SHOW_PLOT: bool = True,
    INCL_STATS: bool = False,
    STAT_PER_LID_CAT: bool = True,
    ALT_MOVELID_BASELINE: bool = False,
    ALPHA: float = 0.05,
    STATS_VERSION: str = '',
    STAT_LID_COMPARE: str = 'binary',
    STAT_DATA_EXT_PATH: bool = True,
    MERGED_STNS: bool = False,
    ADD_TO_FIG_NAME = False,  # additional for quick renaming (debugging)
):
    """
    Plots first keys as rows (LFP-ECoG), second keys as cols (CONTRA/IPSI lat)
    """
    print(f'######### START PLOTTING {PLOT_MOVE_TYPE}')
    # add extra dimension for rest (no movement differences on columns)
    # (movement has double hierarchy already (lfp/ecog, sides))
    if PLOT_MOVE_TYPE == 'REST':
        psd_arrs = {'rest': psd_arrs}
    elif PLOT_MOVE_TYPE == 'INVOLUNT' and MERGED_STNS:
        psd_arrs = {'dysk': psd_arrs}  # one row (dysk), two columns (lfp, ecog)
    elif 'FREE' in PLOT_MOVE_TYPE or any(['FREE' in s for s in PLOT_MOVE_TYPE]):
        # ravel dicts with arrays
        lfp = {m: psd_arrs[m]['lfp'][m] for m in PLOT_MOVE_TYPE}
        ecog = {m: psd_arrs[m]['ecog'][m] for m in PLOT_MOVE_TYPE}
        psd_arrs = {'lfp': lfp, 'ecog': ecog}
        # also ravel subs
        lfp = {m: psd_subs[m]['lfp'][m] for m in PLOT_MOVE_TYPE}
        ecog = {m: psd_subs[m]['ecog'][m] for m in PLOT_MOVE_TYPE}
        psd_subs = {'lfp': lfp, 'ecog': ecog}
        # set defaults
        PLOT_MOVE_TYPE = 'FREE'
        MERGED_STNS = True
        
    # prevent circular import
    from lfpecog_analysis.prep_stats_movLidspecPsd import get_stat_folder
    stat_dir = get_stat_folder(STAT_LID_COMPARE=STAT_LID_COMPARE,
                                STAT_PER_LID_CAT=STAT_PER_LID_CAT,
                                STAT_DATA_EXT_PATH=STAT_DATA_EXT_PATH,
                                STATS_VERSION=STATS_VERSION,
                                ALPHA=ALPHA,)
    
    # LFP-row0, ECoG-row1; IPSI/CONTRA/BOTH in columns
    AX_WIDTH, AX_HEIGHT = 8, 6
    ax_row_keys = list(psd_arrs.keys())
    ax_col_keys = list(psd_arrs[ax_row_keys[0]].keys())
    if ax_col_keys == ['ecog', 'lfp']: ax_col_keys = ['lfp', 'ecog']  # for order in Figure
    

    fig, axes = plt.subplots(len(ax_row_keys), len(ax_col_keys),
                             figsize=(len(ax_col_keys)*AX_WIDTH,
                                      len(ax_row_keys)*AX_HEIGHT),
                             sharey='row', sharex='col',)
    
    if len(axes.shape) == 1: axes = np.atleast_2d(axes)  # make axes array 2d (i.e., 3x1)

    XTICKS = [4, 12, 20, 28, 34, 60, 70, 80, 89]  # 89 will be labeled 90
    ls = 'solid'
    FS = 16  # fontsize

    for axrow, src in enumerate(ax_row_keys):
        # DURING REST: src is here also rest (only 1 row); DURING MOVE: lfp/ecog
        # during merged DYSK -> src is free

        for axcol, mov in enumerate(ax_col_keys):
            # DURING REST: mov is source; during MOVE: contra/ipsi/bi; during FREE: MOVE/ NOMOVE
            # during merged DYSK -> mov is lfp/ecog

            # get STAT dataframe
            if INCL_STATS:
                
                if PLOT_MOVE_TYPE.lower() == 'rest':
                    dfname = f'PsdStateStats_1secWins_REST_{mov}.csv'  # mov is source
                
                elif PLOT_MOVE_TYPE == 'FREE':
                    dfname = f'PsdStateStats_1secWins_{mov}_{src}_MERGED.csv'
                
                elif PLOT_MOVE_TYPE == 'INVOLUNT' and MERGED_STNS:
                    dfname = f'PsdStateStats_1secWins_{PLOT_MOVE_TYPE}_{mov}_MERGED.csv'
                
                else:  # tap and involunt (2x2 or 2x3)
                    if mov.lower() in ['tap_contra', 'tap_ipsi']:
                        mov_side = mov.split('_')[1]
                        dfname = f'PsdStateStats_1secWins_{PLOT_MOVE_TYPE}_{src}_{mov_side}.csv'
                
                    # print(f'plotmovetype: {PLOT_MOVE_TYPE}')
                    # print(f'src: {src}')
                    # print(f'mov: {mov}')
                    else:
                        dfname = f'PsdStateStats_1secWins_{PLOT_MOVE_TYPE}_{src}_{mov}.csv'
                
                # in case of alternative df naming for move with LID categories
                if ALT_MOVELID_BASELINE: dfname = 'alt_' + dfname

                stat_df = os.path.join(stat_dir, dfname)
                if os.path.exists(stat_df):
                    print(f'...stats df loading: {stat_df}')
                    stat_df = read_csv(stat_df, header=0, index_col=0)
                else:
                    raise FileNotFoundError(f'STAT DF not existing ({stat_df}), PM: '
                                            'create with prep_specStats.get_stats_MOVE_psds()')
            
            # LOOP OVER LID-CATEGORIES/ GROUPS
            for lid in psd_arrs[src][mov].keys():
                # get and average correct PSDs
                # check whether psds are means or not 1-sec windows
                if all([len(a.shape) == 2 for a in psd_arrs[src][mov][lid]]):  # given as 1s windows
                    temp_psds = np.array([np.mean(a, axis=0) for a in psd_arrs[src][mov][lid]])
                else:  # given as subject mean PSDs
                    temp_psds = np.array(psd_arrs[src][mov][lid])
                # define subs (and correct for double hemisferes due to merging (left/right; moveUni/Bilat))
                if MERGED_STNS and PLOT_MOVE_TYPE == 'INVOLUNT':
                    n_subs = len(np.unique(psd_subs[mov][lid]))
                elif MERGED_STNS and PLOT_MOVE_TYPE == 'FREE':
                    n_subs = len(np.unique(psd_subs[src][mov][lid]))
                else:
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
                lab = lid.replace('lid', ' LID')  #  add spaces for readability
                lab = lab.replace('below30', ' < 30min')
                lab = lab.replace('over30', ' > 30min')
                # plot mean line of LID severity
                axes[axrow, axcol].plot(
                    x_plot, m,
                    color=cond_colors[lid], alpha=.8, ls=ls,
                    label=f"{lab} (n={n_subs})",
                )
                # plot variance shades (LID severity)
                axes[axrow, axcol].fill_between(
                    x_plot, y1=m-sem, y2=m+sem, # PSDs[cond].freqs, 
                    color=cond_colors[lid], alpha=.3,
                )

            # plot significancies shades (once per AX)
            if INCL_STATS:

                if not STAT_PER_LID_CAT:
                    axes = plot_summary_stats(
                        stat_df=stat_df, axes=axes,
                        PLOT_MOVE_TYPE=PLOT_MOVE_TYPE,
                        STAT_LID_COMPARE=STAT_LID_COMPARE,
                        axrow=axrow, axcol=axcol
                    )
                if STAT_PER_LID_CAT:
                    axes = plot_stats_categs(
                        stat_df, axes, PLOT_MOVE_TYPE,
                        axrow, axcol,
                        ALT_MOVLID_BL=ALT_MOVELID_BASELINE,
                    )
                    

            # add title (once per AX)
            if PLOT_MOVE_TYPE == 'INVOLUNT' and MERGED_STNS:
                ax_title = (f'{mov.upper()} during dyskinetic (invol.) movement')

            elif PLOT_MOVE_TYPE == 'FREE':
                ax_title = (f'{src.upper()}: unscripted FREE-task: {mov.split("FREE")[1]}MENT')
            
            elif PLOT_MOVE_TYPE != 'REST':
                ax_title = (f'{src.upper()} during '
                            f'{mov.split("_")[1]}-lateral'
                            f' {mov.split("_")[0]}-movement')
                ax_title = ax_title.replace('INVOLUNT', 'DYSK')
                ax_title = ax_title.replace('BILAT', 'BI')
            
            else:
                ax_title = (f'{mov.upper()} during REST (no movement)')
            
            axes[axrow, axcol].set_title(ax_title, weight='bold',
                                            size=FS,)


    xtick_sel = np.where([x in XTICKS for x in xlabs])[0]
    xticks = x_plot[xtick_sel]
    xlabs = np.array(xlabs)[xtick_sel]
    xlabs[-1] = 90  # mark last tick with 90 (instead of 89)

    for i_ax, ax in enumerate(axes.ravel()):
        ax.axhline(0, xmin=0, xmax=1, color='gray', alpha=.3,)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs, fontsize=FS,)
        ax.set_ylim(-75, 250)
        if AX_WIDTH == 8: leg_fs = FS - 2
        elif AX_WIDTH == 6: leg_fs = FS - 4
        ax.legend(fontsize=leg_fs, frameon=False, ncol=2,)
        ax.tick_params(size=FS, labelsize=FS,)
        for s in ['right', 'top']: ax.spines[s].set_visible(False)
        ax.text(x=xticks[len(xticks) // 2] + 2, y=-73, s='//', size=16, color='k')

    # set axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)', size=FS, weight='bold',)
    for ax in axes[:, 0]:
        ax.set_ylabel('Spectral Power, %-change\n'
                      '(vs Med-OFF, no LID)', size=FS, weight='bold',)

    plt.tight_layout()

    if SAVE_PLOT:
        FIG_NAME = f'PSDs_{PLOT_MOVE_TYPE}'
        if MERGED_STNS: FIG_NAME += '_mergedSTNs'
        if INCL_STATS:
            FIG_NAME += f'_stats{STAT_LID_COMPARE}'
            if len(STATS_VERSION) > 1: FIG_NAME += STATS_VERSION
        if ADD_TO_FIG_NAME:
            FIG_NAME = ADD_TO_FIG_NAME + FIG_NAME
        FIG_PATH = os.path.join(get_project_path('figures'),
                                'ft_exploration',
                                'data_v4.0_ft_v6',
                                'PSDs_state_specific')
        
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)
        print(f'saved plot {FIG_NAME} in {FIG_PATH}!')

    if SHOW_PLOT: plt.show()
    else: plt.close()




def prep_MOVEMENT_spec_psds(PLOT_MOVE, PSD_DICT, BASELINE,
                            RETURN_IDS: bool = False,
                            RETURN_STATE_ARRAYS: bool = False,
                            MERGED_STNS: bool = False,):
    """

    Arguments:
        - PLOT_MOVE: TAP or INVONUT
        - PSD_DICT: dict with conditions as keys and containing
            get_selectedEphys() classes per condition (from
            psd_analysis_classes.py)
        - BASELINE: get_selectedEphys() class
        - MERGE_REST_STNS: to keep arguments parallel with rest

    Returns:
        - psd_arrs: dicts of lists with selected and processed
            psd-arrays, organized in LFP/ECoG (axrows), and
            CONTRA/IPSI-lateral movements (axcols)
    """
    print(f'start prep MOVEMENT spec psdsPLOTMOVE: {PLOT_MOVE}')
    assert PLOT_MOVE in ['INVOLUNT', 'TAP', 'FREE', 'FREEMOVE', 'FREENOMOVE']

    if PLOT_MOVE == 'INVOLUNT': ATTR_CODE = 'DYSKMOVE'
    else: ATTR_CODE = PLOT_MOVE

    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['no', 'mild', 'moderate', 'severe']
    move_axes = [f'{PLOT_MOVE}_CONTRA', f'{PLOT_MOVE}_IPSI']
    
    if PLOT_MOVE == 'INVOLUNT' and not MERGED_STNS:
        move_axes.append('INVOLUNT_BILAT')
    
    if 'FREE' in PLOT_MOVE: move_axes = [PLOT_MOVE]
    if MERGED_STNS: sources = ['lfp', 'ecog']

    # create dicts to store psds
    psd_arrs = {'lfp': {d: [] for d in move_axes},
                'ecog': {d: [] for d in move_axes}}  # store all psds for move axes
    # store IDs parallel to psds
    if RETURN_IDS: sub_arrs = {'lfp': {d: [] for d in move_axes},
                               'ecog': {d: [] for d in move_axes}}

    # replace single large lists for dicts with lid-states
    for s, m in product(psd_arrs.keys(), move_axes):
        psd_arrs[s][m] = {f'{l}lid': [] for l in lid_states}
        if RETURN_IDS: sub_arrs[s][m] = {f'{l}lid': [] for l in lid_states}
    
    if PLOT_MOVE == 'INVOLUNT' and MERGED_STNS:
        psd_arrs = {'lfp': {f'{l}lid': [] for l in lid_states},
                    'ecog': {f'{l}lid': [] for l in lid_states}}
        if RETURN_IDS:
            sub_arrs['lfp'] = {f'{l}lid': [] for l in lid_states}
            sub_arrs['ecog'] = {f'{l}lid': [] for l in lid_states}

    # categorize, average, baseline-corr all conditions
    for SRC, cond in product(sources, PSD_DICT.keys()):
        # print(f'...{SRC}, {cond}, {ATTR_CODE}')
        if not ATTR_CODE in cond.upper(): continue
        # loop and add subjects
        for attr in vars(PSD_DICT[cond]).keys():
            # print(f'.......{attr}')
            # skip irelevant attr
            if not ATTR_CODE in attr.upper(): continue
            if SRC not in attr: continue
            # print(f'...selected attr (prep move spec psds): {attr}')
            # select on sub
            if 'lfp' in SRC: sub = attr.split('_')[2]  # exclude lfp side
            elif SRC == 'ecog': sub = attr.split('_')[1]  # exclude lfp side
            if not sub.startswith('0') and not sub.startswith('1'): continue
            # get psd arr and correct for baseline
            temp_psd = getattr(PSD_DICT[cond], attr)
            # print(f'...SHAPE temp psd {temp_psd.shape}')
            if SRC == 'lfp' and MERGED_STNS:
                try:
                    if 'FREE' in PLOT_MOVE: bl_attr = f'lfp_{attr.split("_")[1]}_{sub}_baseline'
                    else: bl_attr = f'lfp_right_{sub}_baseline'  # for merged taps/dysk
                    bl = getattr(BASELINE, bl_attr)
                    if len(bl.shape) == 2: bl = bl.mean(axis=0)
                    if not 'FREE' in PLOT_MOVE:  # for merged stns takes mean baseline
                        bl2 = getattr(BASELINE, f'lfp_left_{sub}_baseline')
                        if len(bl2.shape) == 2: bl2 = bl2.mean(axis=0)
                        bl = np.mean([bl, bl2], axis=0)
                except:
                    print(f'### WARNING no merged-STN-baseline sub {sub}')
                    continue
            else:
                try:
                    bl = getattr(BASELINE, f'{SRC}_{sub}_baseline')
                    # take mean baseline if 1-sec windows are given
                    if len(bl.shape) == 2: bl = bl.mean(axis=0)
                except:
                    print(f'### WARNING: no baseline found for {sub}: {SRC}, skipped')
                    continue

            temp_psd = ((temp_psd - bl) / bl) * 100
            if RETURN_STATE_ARRAYS: print(f'...arr shape {attr}: {temp_psd.shape}')

            # add to LFP or ECOG and correct move-dict
                # define hemisphere location to movement
            if not (PLOT_MOVE == 'INVOLUNT' and MERGED_STNS) and (
                'FREE' not in PLOT_MOVE
            ):
                HEM_LOC, MOV_TYPE = get_hemisphere_movement_location(
                    SRC=SRC, cond=cond, sub=sub,
                    ecog_sides=PSD_DICT[cond].ecog_sides,
                )

            # include LID-state:
            temp_lid = attr.split('_')[-1]
            if PLOT_MOVE == 'INVOLUNT' and MERGED_STNS:
                psd_arrs[SRC.split('_')[0]][temp_lid].append(temp_psd)
                # add subject ID to list
                if RETURN_IDS: sub_arrs[
                    SRC.split('_')[0]][temp_lid
                ].append(sub)                
            
            elif PLOT_MOVE in ['INVOLUNT', 'TAP']:
                psd_arrs[
                    SRC.split('_')[0]][f'{MOV_TYPE}_{HEM_LOC}'
                ][temp_lid].append(temp_psd)
                # add subject ID to list
                if RETURN_IDS: sub_arrs[
                    SRC.split('_')[0]][f'{MOV_TYPE}_{HEM_LOC}'
                ][temp_lid].append(sub)
                    
            elif 'FREE' in PLOT_MOVE:
                # print(f'...add to psd_arrs {PLOT_MOVE}, {temp_lid} --> {temp_psd.shape}')
                psd_arrs[SRC.split('_')[0]][PLOT_MOVE][temp_lid].append(temp_psd)
                # add subject ID to list
                if RETURN_IDS:
                    sub_arrs[SRC.split('_')[0]][PLOT_MOVE][temp_lid].append(sub)
            
    psd_freqs = PSD_DICT[cond].freqs

    if not RETURN_IDS: return psd_arrs, psd_freqs

    elif RETURN_IDS: return psd_arrs, psd_freqs, sub_arrs


def prep_REST_spec_psds(PSD_DICT, BASELINE,
                        PLOT_MOVE: str = 'REST',
                        RETURN_IDS: bool = False,
                        MERGED_STNS: bool = False,):
    """

    Arguments:
        - PSD_DICT: dict with conditions as keys and containing
            get_selectedEphys() classes per condition (from
            psd_analysis_classes.py)
        - BASELINE: get_selectedEphys() class
        - PLOT_MOVE: included to keep function in parallel
            with MOVEMENT prep_spec_psds function
        - RETURN_IDS: give parallel output with subject IDs
            corresponding to psd_arr (for mix eff models) 

    Returns:
        - psd_arrs: dicts of lists with selected and processed
            psd-arrays, organized in LFP/ECoG (axrows), and
            CONTRA/IPSI-lateral movements (axcols)
    """
    
    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['nolidbelow30', 'nolidover30',
                  'mildlid', 'moderatelid', 'severelid']

    psd_arrs = {s: {l: [] for l in lid_states} for s in sources}  # store all psds for move axes
    
    if RETURN_IDS: sub_arrs = {s: {l: [] for l in lid_states} for s in sources}  # store IDs parallel to psds

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
                # take mean baseline if 1-sec windows are given
                if len(bl.shape) == 2: bl = bl.mean(axis=0)
            except:
                print(f'### WARNING: no baseline found for {sub}: {SRC}, skipped')
                continue
            temp_psd = ((temp_psd - bl) / bl) * 100

            # include LID-state:
            temp_lid = attr.split('_')[-1]
            psd_arrs[SRC][temp_lid].append(temp_psd)
            
            # add subject ID to list
            if RETURN_IDS: sub_arrs[SRC][temp_lid].append(sub)

    # merge both STNs if defined
    if MERGED_STNS:
        stn_arrs = {s: [] for s in psd_arrs['lfp_left'].keys()}
        if RETURN_IDS: stn_subs = {s: [] for s in sub_arrs['lfp_left'].keys()}
        for side in ['left', 'right']:
            for s in psd_arrs[f'lfp_{side}'].keys():
                stn_arrs[s].extend(psd_arrs[f'lfp_{side}'][s])
                if RETURN_IDS: stn_subs[s].extend(sub_arrs[f'lfp_{side}'][s])
                    
        psd_arrs['lfp'] = stn_arrs
        for s in ['left', 'right']: del(psd_arrs[f'lfp_{s}'])
        # merge sub ids
        if RETURN_IDS:
            sub_arrs['lfp'] = stn_subs
            for s in ['left', 'right']: del(sub_arrs[f'lfp_{s}'])
    
    psd_freqs = PSD_DICT[cond].freqs
    
    if not RETURN_IDS: return psd_arrs, psd_freqs

    elif RETURN_IDS: return psd_arrs, psd_freqs, sub_arrs




def plot_rest_lmm(
    stat_dfs,
    FIGNAME=False, SAVE_FIG: bool = False,
    a = .05,
):
    """
    Plot STAT significancies for REST PSDs
    """
    # define LID split for analysis
    split_labels = ['no-LID (<30) vs all-LID',
                    'no-LID (all) vs all-LID']
        
    fig, axes = plt.subplots(1, len(stat_dfs),
                             figsize=(len(stat_dfs) * 4, 4))

    for i_src, stat_df in enumerate(stat_dfs.values()):
        SRC = list(stat_dfs.keys())[i_src]
        # plot coeffs
        axes[i_src].plot(stat_df.index, stat_df.iloc[:, 0], ls='-', c='blue', alpha=.5,)
        axes[i_src].plot(stat_df.index, stat_df.iloc[:, 2], ls='--', c='darkblue', alpha=.8,)

        axes[i_src].axhline(y=0, color='gray', lw=1, alpha=.5,)
        ymin, ymax = axes[i_src].get_ylim()

        # shade significancies
        axes[i_src].fill_between(
            y1=ymin, y2=ymax, x=stat_df.index,
            where=stat_df.iloc[:, 1], facecolor='None',
            edgecolor='gray', hatch='//',
            label=split_labels[0],
        )
        axes[i_src].fill_between(
            y1=ymin, y2=ymax, x=stat_df.index,
            where=np.logical_and(stat_df.iloc[:, 1], stat_df.iloc[:, 3]),
            color='gray', alpha=.3,
            label=split_labels[1],
        )

        axes[i_src].legend()

        axes[i_src].set_title(
            f'{SRC} in REST (alpha = {round(a / len(stat_df), 4)})',
            weight='bold'
        )
        axes[i_src].set_xlabel('Freq (Hz)')
        axes[i_src].set_ylabel('GLMM Coefficient (a.u.)')

    plt.tight_layout()

    if SAVE_FIG:
        # if len(stat_dfs) == 2:
        #     FIGNAME += '_mergedSTNs'
        FIG_DIR = os.path.join(get_project_path('figures'),
                                 'ft_exploration',
                                 'data_v4.0_ft_v6',
                                 'PSDs_state_specific', 'only_stats')
        print(f'...saving plot {FIGNAME} in {FIG_DIR}')
        plt.savefig(os.path.join(FIG_DIR, FIGNAME),
                    dpi=300, facecolor='w',)
        plt.close()
    else:
        plt.show()


def plot_move_lmm(MOV, stat_dfs,
                  FIGNAME=False, SAVE_FIG: bool = False,):
    
    print(f'PLOTTING {MOV}')
    
    if MOV == 'TAP': ncols = 2
    else: ncols = 3

    fig, axes = plt.subplots(2, ncols, figsize=(12, 8))
    axes = axes.flatten()

    for i_src, stat_df in enumerate(stat_dfs.values()):
        print(f'...PLOT {list(stat_dfs.keys())[i_src]}')

        axes[i_src].plot(stat_df.index, stat_df.iloc[:, 0],
                         ls='-', c='blue', alpha=.5,)

        axes[i_src].axhline(y=0, color='gray', lw=1, alpha=.5,)

        ymin, ymax = axes[i_src].get_ylim()
        axes[i_src].fill_between(
            y1=ymin, y2=ymax, x=stat_df.index,
            where=stat_df.iloc[:, 1], facecolor='None',
            edgecolor='gray', hatch='//',)

        axes[i_src].set_title(f'{MOV}: {list(stat_dfs.keys())[i_src]}')
        axes[i_src].set_xlabel('Freq (Hz)')
        axes[i_src].set_ylabel('GLMM Coefficient (a.u.)')

    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(os.path.join(get_project_path('figures'),
                                 'ft_exploration',
                                 'data_v4.0_ft_v6',
                                 'PSDs_state_specific',
                                 FIGNAME),
                    dpi=300, facecolor='w',)
        plt.close()
    else:
        plt.show()


def plot_summary_stats(stat_df, axes, PLOT_MOVE_TYPE,
                       STAT_LID_COMPARE, axrow, axcol):
    if PLOT_MOVE_TYPE == 'REST':    
        stat_values = [stat_df.iloc[:, 1].values,
                        np.logical_and(stat_df.iloc[:, 1],
                                        stat_df.iloc[:, 3]).values]
    else:
        stat_values = [stat_df.iloc[:, 1].values,]
    if STAT_LID_COMPARE == 'binary':
        split_labels = ['no-LID (<30) vs all-LID',
                        'no-LID (all) vs all-LID']
    elif STAT_LID_COMPARE == 'linear':
        split_labels = ['sign linear LID coeff',]

    
    ymin, ymax = axes[axrow, axcol].get_ylim()
    # break and nan-pad x-axis
    sign_bool, sign_x, sign_xlabs = break_x_axis_psds_ticks(
        tf_freqs=stat_df.index,
        PSD=stat_values[-1],
        x_break = (35, 60), nan_pad = 5
    )
    axes[axrow, axcol].fill_between(
        y1=ymin, y2=ymax, x=sign_x,
        where=sign_bool,
        color='gray', alpha=.2,
        label=split_labels[-1],
    )
    if PLOT_MOVE_TYPE == 'REST':
        # break and nan-pad x-axis
        sign_bool, sign_x, _ = break_x_axis_psds_ticks(
            tf_freqs=stat_df.index,
            PSD=stat_values[0], # sign-bool vs <30 AND all
            x_break = (35, 60), nan_pad = 5
        )
        axes[axrow, axcol].fill_between(
            y1=ymin, y2=ymax, x=sign_x,
            where=sign_bool,  # get only sign-bool vs <30
            facecolor='None', edgecolor='gray', hatch='//',
            label=split_labels[0], alpha=.5,
        )

    return axes


def plot_stats_categs(stat_df, axes, PLOT_MOVE_TYPE,
                      axrow, axcol, lw=8,
                      ALT_MOVLID_BL: bool = False,):
    sign_bools = {}

    if PLOT_MOVE_TYPE == 'REST':
        # get groups
        stat_groups = [k.split('_')[1].split(' vs ')
                       for k in stat_df.keys()]
        groups_lid = np.unique([g[1] for g in stat_groups])  # tested groups
        Y_BASE = 175
    else:
        groups_lid = list(np.unique([g.split("_")[1] for g in stat_df.keys()]))
        Y_BASE = 200
        if 'nolid' in groups_lid:  # put nolid in first position
            groups_lid.remove('nolid')
            groups_lid = ['nolid'] + groups_lid

    for i_cat, cat in enumerate(groups_lid):  # loop over CATEGs to show
        if ALT_MOVLID_BL and cat == 'nolid':
            print(f'dont show significance (ALT LID-cat BASELINING) for cat {cat}')
            continue
        # signs for full and only compared to <30
        if PLOT_MOVE_TYPE == 'REST':
            sign_bools['full'] = stat_df[f'sig_no-LID vs {cat}']
            sign_bools['u30'] = stat_df[f'sig_no-LID (<30) vs {cat}']
            sign_bools['u30'] = np.logical_and(sign_bools['u30'], ~sign_bools['full'])
            cat = cat.split("-")[1]
        else:
            sign_bools['full'] = stat_df[f'sig_{cat}']
            cat = cat.split('lid')[0]

        # break and nan-pad x-axis
        for k in sign_bools:
            sign_bools[k], sign_x, _ = break_x_axis_psds_ticks(
                tf_freqs=sign_bools[k].index,
                PSD=sign_bools[k].values,
                x_break = (35, 60), nan_pad = 5
            )
        # first plot baseline for color background
        axes[axrow, axcol].fill_between(
            sign_x, y1=Y_BASE - ((i_cat + 1) * lw),
            y2=Y_BASE - (i_cat * lw),
            color=cond_colors[f'{cat}lid'],
            alpha=.05,
        )
        # plot dashed for only sign vs below30
        if 'u30' in list(sign_bools.keys()):
            axes[axrow, axcol].fill_between(
                sign_x, where=sign_bools['u30'] == 1,
                y1=Y_BASE - ((i_cat + 1) * lw),
                y2=Y_BASE - (i_cat * lw),
                hatch='//', facecolor='None',
                edgecolor=cond_colors[f'{cat}lid'], alpha=.8,
            )
        # plot full color for full signs
        axes[axrow, axcol].fill_between(
            sign_x, where=sign_bools['full'] == 1,
            y1=Y_BASE - ((i_cat + 1) * lw),
            y2=Y_BASE - (i_cat * lw),
            color=cond_colors[f'{cat}lid'], alpha=.5,
        )




    return axes