"""
script to prepare data for statistics
on the PSDs of movement- and LID-specific
selections

REest and movement moments have different
functions due to different hierarchy in
array-dictionaries, rest is split in LFP-L,
LFP-R, ECoG; movements are split in LFP, ECoG,
AND IPSI, CONTRA movements
"""

# import public packes
import os
import numpy as np
import pandas as pd
from itertools import product

# import own functions
from lfpecog_plotting.plot_move_spec_psd import (
    prep_REST_spec_psds, prep_MOVEMENT_spec_psds,
    plot_rest_lmm, plot_move_lmm
)
from utils.utils_fileManagement import get_project_path
import lfpecog_analysis.psd_lid_stats as psd_Stats
from lfpecog_analysis.psd_analysis_classes import (
    get_allSpecStates_Psds
)


def get_REST_stat_grouped_data(
    PSD_DICT, BL_class,
    STAT_SPLIT: str,
    BIN_or_LIN: str = 'binary',
    SRC: str = 'lfp_left',
    MERGE_STNs: bool = False,
    PSD_1s_windows: bool = False,
):
    # DEFINE STAT GROUPS

    # binary groups
    if STAT_SPLIT == 'no-LID vs all-LID' and BIN_or_LIN == 'binary':
        group_states = {0: ['nolidbelow30', 'nolidover30'],
                        1: ['mildlid', 'moderatelid', 'severelid']}
        
    elif STAT_SPLIT == 'no-LID (<30) vs all-LID' and BIN_or_LIN == 'binary':
        group_states = {0: ['nolidbelow30'],
                        1: ['mildlid', 'moderatelid', 'severelid']}
    
    # linear stat groups
    elif STAT_SPLIT == 'no-LID vs all-LID' and BIN_or_LIN == 'linear':
        group_states = {0: ['nolidbelow30', 'nolidover30'],
                        1: ['mildlid'], 2: ['moderatelid'], 3: ['severelid']}
    
    elif STAT_SPLIT == 'no-LID (<30) vs all-LID' and BIN_or_LIN == 'linear':
        group_states = {0: ['nolidbelow30'], 1: ['nolidover30'],
                        2: ['mildlid'], 3: ['moderatelid'], 4: ['severelid']}

    # binary groups PER LID-CATEGORY
    elif STAT_SPLIT == 'no-LID (<30) vs LID-categs':
        group_states = {0: ['nolidbelow30'],
                        1: ['mildlid'], 2: ['moderatelid'], 3: ['severelid']}
    
    elif STAT_SPLIT == 'no-LID vs LID-categs':
        group_states = {0: ['nolidbelow30', 'nolidover30'],
                        1: ['mildlid'], 2: ['moderatelid'], 3: ['severelid']}
    
    # BASELINE FOR MOVEMENT STATISTICS
    elif STAT_SPLIT == 'move_baseline':
        group_states = {0: ['nolidbelow30', 'nolidover30'],}
    
    # dictionaries to store grouped results
    group_subs = {i: [] for i in group_states.keys()}
    group_psds = {i: [] for i in group_states.keys()}

    # GET PSD ARRAYS
    (psd_arrs,
     psd_freqs,
     psd_subs) = prep_REST_spec_psds(
        PSD_DICT=PSD_DICT,
        BASELINE=BL_class,
        RETURN_IDS=True,
        MERGE_REST_STNS=MERGE_STNs,)

    src_psd = psd_arrs[SRC]
    src_subs = psd_subs[SRC]


    # put data from diff LID-states in correct groups
    for group in group_states:  # loop over different states [0, 1, 2, 3 ...]
        for lid_stat in group_states[group]:  # loop over exact LID states within group [i.e., mildlid, nolidbelow30]

            temp_psd = np.array(src_psd[lid_stat], dtype='object')
            temp_subs = src_subs[lid_stat]
            assert temp_psd.shape[0] == len(temp_subs), (
                f'group-{group}, {lid_stat}: psd_arrs and psd_subs DONT MATCH'
                f' psd: {temp_psd.shape}, subs: {len(temp_subs)}'
            )
            # add to, or create at start
            try:
                group_psds[group] = np.concatenate([group_psds[group], temp_psd])
                group_subs[group].extend(temp_subs)
            except:
                group_psds[group] = temp_psd.copy()  # prevent coupled changing of original object
                group_subs[group] = temp_subs.copy()  # prevent coupled changing of original object
        
        # unpack nested arrays
        if PSD_1s_windows:
            group_subs[group] = [
                [group_subs[group][i]] * group_psds[group][i].shape[0]
                 for i in np.arange(len(group_psds[group]))
            ]
            group_subs[group] = [s for l in group_subs[group] for s in l]
            group_psds[group] = np.array([row for arr in group_psds[group]
                                          for row in arr])
        
        # check whether resulting psd array is n-subs x n-freqs
        assert group_psds[group].shape == (len(group_subs[group]),
                                           len(psd_freqs)), (
                f'psd_arrs ({group_psds[group].shape}) DONT MATCH '
                f'subs: {len(group_subs[group])} and/or freqs: {len(psd_freqs)}'
        )

    stat_values = np.concatenate([group_psds[i] for i in group_states.keys()])
    stat_labels = np.concatenate([[i] * len(group_subs[i]) for i in group_states.keys()])
    stat_ids = np.concatenate([np.array(group_subs[i]) for i in group_states.keys()])

    return stat_values, stat_labels, stat_ids, psd_freqs


def get_MOVE_stat_grouped_data(
    PSD_DICT, BL_class,
    MOVE_COND: str,
    STAT_SPLIT: str = 'linear',
    SRC: str = 'lfp',
    PSD_1s_windows: bool = False,
):
    """
    creates two dicts with stat-data-arrays,
        stat_values, stat_labels, stat_ids, psd_freqs
    for CONTRA and IPSI movements (TAP or INVOLUNT)

    psd_freqs (arr) is not doubled

    """
    # DEFINE STAT GROUPS
    if STAT_SPLIT == 'binary':
        group_states = {0: ['nolid'],
                        1: ['mildlid', 'moderatelid', 'severelid']}
    elif STAT_SPLIT == 'linear':
        group_states = {0: ['nolid'], 1: ['mildlid'], 
                        2: ['moderatelid'], 3: ['severelid']}
        
    # binary groups PER LID-CATEGORY
    elif STAT_SPLIT == 'categs':
        group_states = {0: ['nolid'],
                        1: ['mildlid'], 2: ['moderatelid'], 3: ['severelid']}
    
    # GET PSD ARRAYS (move states are dependent on move_cond)
    (psd_arrs,
     psd_freqs,
     psd_subs) = prep_MOVEMENT_spec_psds(
        PLOT_MOVE=MOVE_COND,
        PSD_DICT=PSD_DICT,
        BASELINE=BL_class,
        RETURN_IDS=True)
    src_psd = psd_arrs[SRC]  # contains INVOLUNT/TAP_CONTRA/IPSI/BILAT
    src_subs = psd_subs[SRC]

    MOVE_STATES = list(src_psd.keys())  # contains both TAP_CONTRA e.g.
    
    group_subs = {m : {i: [] for i in group_states.keys()}
                  for m in MOVE_STATES}
    group_psds = {m : {i: [] for i in group_states.keys()}
                  for m in MOVE_STATES}

    # put data from diff LID-states in correct groups
    
    for MOV, group in product(MOVE_STATES, group_states):
        # loops over LID states within the lists (i.e. mildlid)
        for lid_stat in group_states[group]:
            # print(f'...lid label: {lid_stat}')

            temp_psd = np.array(src_psd[MOV][lid_stat], dtype='object',)  # array with lists/arrays
            temp_subs = src_subs[MOV][lid_stat]

            assert temp_psd.shape[0] == len(temp_subs), (
                f'group-{group}, {MOV}, {lid_stat}: psd_arrs and psd_subs DONT MATCH'
                f' psd: {temp_psd.shape}, subs: {len(temp_subs)}'
            )
            # add to, or create at start
            try:
                group_psds[MOV][group] = np.concatenate([group_psds[MOV][group], temp_psd])
                group_subs[MOV][group].extend(temp_subs)
            except:
                group_psds[MOV][group] = temp_psd.copy()  # prevent coupled changing of original object
                group_subs[MOV][group] = temp_subs.copy()  # prevent coupled changing of original object
        
        # check whether resulting psd array is n-subs x n-freqs
        if len(group_psds[MOV][group]) == 0:
            del group_psds[MOV][group]
            continue

        # unpack nested arrays
        if PSD_1s_windows:
            group_subs[MOV][group] = [
                [group_subs[MOV][group][i]] * group_psds[MOV][group][i].shape[0]
                 for i in np.arange(len(group_psds[MOV][group]))
            ]
            group_subs[MOV][group] = [
                s for l in group_subs[MOV][group] for s in l
            ]
            group_psds[MOV][group] = np.array(
                [row for arr in group_psds[MOV][group] for row in arr]
            )

        assert group_psds[MOV][group].shape == (
            len(group_subs[MOV][group]), len(psd_freqs)
        ), (f'psd_arrs ({group_psds[MOV][group].shape}) DONT MATCH '
            f'subs: {len(group_subs[MOV][group])} and/or freqs: {len(psd_freqs)}')

    # dicts to return
    stat_values, stat_labels, stat_ids = {}, {}, {}
    for MOV in MOVE_STATES:  # loop over TAP / INVOLUNT
        stat_values[MOV] = np.concatenate([group_psds[MOV][i] for i in group_psds[MOV].keys()])
        stat_labels[MOV] = np.concatenate([[i] * len(group_subs[MOV][i]) for i in group_states.keys()])
        stat_ids[MOV] = np.concatenate([np.array(group_subs[MOV][i]) for i in group_states.keys()])

    return stat_values, stat_labels, stat_ids, psd_freqs


def get_stats_REST_psds(
    STAT_DATA_EXT_PATH = True,
    STAT_LID_COMPARE = 'binary',
    PLOT_STATS = False,
    MERGE_STNs: bool = False,
    STATS_VERSION: str = '',
    STAT_PER_LID_CAT: bool = False,
):
    CLASSES_LOADED = False  # bool to only load classes once during creation
    
    stat_dir = get_stat_folder(STAT_LID_COMPARE=STAT_LID_COMPARE,
                               STAT_PER_LID_CAT=STAT_PER_LID_CAT,
                               STAT_DATA_EXT_PATH=STAT_DATA_EXT_PATH,
                               STATS_VERSION=STATS_VERSION,)

    # define LID split for analysis
    if STAT_PER_LID_CAT:
        allowed_splits = ['no-LID (<30) vs LID-categs',
                          'no-LID vs LID-categs']
        bool_dicts, coeff_dicts = [], []
    else:
        allowed_splits = ['no-LID (<30) vs all-LID',
                          'no-LID vs all-LID']
    
    sources = ['lfp_left', 'lfp_right', 'ecog']
    if MERGE_STNs: sources = ['lfp', 'ecog']

    stat_dfs = {}

    for i_src, SRC in enumerate(sources):
        print(f'#### START REST stat psds extraction: {SRC}')
        # for binary LID comparisons
        df_name = f'PsdStateStats_1secWins_REST_{SRC}.csv'
        stat_path = os.path.join(stat_dir, df_name)

        if os.path.exists(stat_path):
            print(f'... loading existing csv {stat_path}')
            stat_dfs[SRC] = pd.read_csv(stat_path, header=0, index_col=0)
            continue

        # CALCULATE statistics
        print(f'... calculating non-existing csv {stat_path}')
        if not CLASSES_LOADED:
            PSDs, BLs = get_allSpecStates_Psds(RETURN_PSD_1sec=True)
            CLASSES_LOADED = True
            
        sign_bools, lmm_coeffs = [], []

        for SPLIT in allowed_splits:
            # get grouped stat data
            (
                stat_values, stat_labels, stat_ids, value_freqs
            ) = get_REST_stat_grouped_data(
                STAT_SPLIT=SPLIT, SRC=SRC,
                BIN_or_LIN=STAT_LID_COMPARE,
                PSD_DICT=PSDs, BL_class=BLs,
                PSD_1s_windows=True,
                MERGE_STNs=MERGE_STNs,
            )

            # get STATS (coeffs, sign-bools) based on grouped data
            (
                (coeffs_freqs, sign_freqs), stat_freqs
            ) = psd_Stats.calc_lmem_freqCoeffs(
                temp_values=stat_values,
                temp_ids=stat_ids,
                temp_freqs=value_freqs,
                VALUES_GIVEN_GROUPED=True,
                GROUP_LABELS=stat_labels,
                STATS_PER_LID_CAT=STAT_PER_LID_CAT,
            )
            if not STAT_PER_LID_CAT:
                sign_bools.append(sign_freqs)
                lmm_coeffs.append(coeffs_freqs)
            else:
                bool_dicts.append(sign_freqs)
                coeff_dicts.append(coeffs_freqs)
            

        # SAVE STATS in dataframes
        if not STAT_PER_LID_CAT:
            # lists to arrays
            sign_bools = np.array(sign_bools)
            lmm_coeffs = np.array(lmm_coeffs)

            colnames = [[f'coef_{s}', f'sig_{s}'] for s in allowed_splits]
            stat_df = pd.DataFrame(
                index=stat_freqs,
                columns=[c for s in colnames for c in s])

            stat_df.iloc[:, 0] = lmm_coeffs[0]
            stat_df.iloc[:, 2] = lmm_coeffs[1]
            stat_df.iloc[:, 1] = sign_bools[0]
            stat_df.iloc[:, 3] = sign_bools[1]
            print(f'\n...SAVING STAT DF: {df_name} (in {stat_dir})')
            stat_df.to_csv(os.path.join(stat_dir, df_name),
                            header=True, index=True)
            stat_dfs[SRC] = stat_df
            
            if PLOT_STATS:
                FIG_NAME = f"REST_lmm_sigs_{STAT_LID_COMPARE}"
                if MERGE_STNs: FIG_NAME += '_mergedSTNs'
                print(f'\n...call plotting {FIG_NAME} ')
                plot_rest_lmm(stat_dfs, FIGNAME=FIG_NAME, SAVE_FIG=True,)
            
        elif STAT_PER_LID_CAT:
            stat_df = pd.DataFrame(index=stat_freqs,)
            for i_split, split in enumerate(allowed_splits):
                for cat in bool_dicts[0].keys():
                    # create coef and sig columns corr with split and category
                    stat_df[
                        f'coef_{split.replace("categs", cat)}'
                    ] = coeff_dicts[i_split][cat]
                    stat_df[
                        f'sig_{split.replace("categs", cat)}'
                    ] = bool_dicts[i_split][cat]

            print(f'\n...SAVING STAT DF: {df_name} with keys: '
                  f'{stat_df.keys()} (in {stat_dir})')
            stat_df.to_csv(os.path.join(stat_dir, df_name),
                            header=True, index=True)


def get_stat_folder(STAT_LID_COMPARE: str,
                    STAT_PER_LID_CAT: bool = False,
                    STAT_DATA_EXT_PATH: bool = True,
                    STATS_VERSION: str = '',):
    
    if STAT_DATA_EXT_PATH:
        stat_dir = ('D://Research/CHARITE/projects/'
                    'dyskinesia_neurophys/data/'
                    'windowed_data_classes_10s_0.5overlap/psdStateStats')
    elif not STAT_DATA_EXT_PATH:
        stat_dir = (get_project_path('data'),
                    'windowed_data_classes_10s_0.5overlap/psdStateStats')

    if len (STATS_VERSION) > 1: stat_dir += f'_{STATS_VERSION}'

    assert STAT_LID_COMPARE in ['binary', 'linear', 'categs'], (
        f'STAT_LID_COMPARE ({STAT_LID_COMPARE}) should be linear / binary / categs'
    )
    if STAT_PER_LID_CAT:
        stat_dir += f'_lidCategs'
    else:      
        stat_dir += f'_lid{STAT_LID_COMPARE.capitalize()}'  # add Linear or Binary

    return stat_dir


def get_stats_MOVE_psds(
    STAT_DATA_EXT_PATH = True,
    STAT_LID_COMPARE = 'linear',
    PLOT_STATS = False,
    STATS_VERSION: str = '',
    STAT_PER_LID_CAT: bool = False,
    REST_BASELINE: bool = True,
):
    """
    Get statistical difference between movement
    states and OTHER MOVEMENT STATES, or diff
    between movements and BASELINE REST (no movement)

        - REST_BASELINE: if True all move categs are compared
            against all-REST-noLID, if False, all move-LID-categs
            are compared against same move-no-LID
    """
    SOURCES = ['lfp', 'ecog']
    SIDES = ['CONTRA', 'IPSI', 'BILAT']

    CLASSES_LOADED = False  # bool to only load classes once during creation

    if STAT_PER_LID_CAT: STAT_LID_COMPARE = 'categs'  # default with one category vs baseline

    stat_dir = get_stat_folder(STAT_LID_COMPARE=STAT_LID_COMPARE,
                               STAT_PER_LID_CAT=STAT_PER_LID_CAT,
                               STAT_DATA_EXT_PATH=STAT_DATA_EXT_PATH,
                               STATS_VERSION=STATS_VERSION,)

    
    for MOV in ['TAP', 'INVOLUNT']:
    
        if STAT_LID_COMPARE == 'binary' and MOV == 'INVOLUNT':
            print(f'SKIP binary comparison for dyskinetic movement')
            continue

        print(f'\n######### START {MOV}  get_stats_MOVE_psds()\n')

        stat_dfs = {}

        for i_src, (SRC, SIDE) in enumerate(
            product(SOURCES, SIDES)
        ):
            if MOV == 'TAP' and SIDE == 'BILAT': continue
            print(f'({MOV}) START-{i_src}: {SRC} x {SIDE}')
            df_name = f'PsdStateStats_1secWins_{MOV}_{SRC}_{SIDE}.csv'
            stat_path = os.path.join(stat_dir, df_name)

            if os.path.exists(stat_path):
                print(f'...load {df_name}')
                stat_dfs[f'{SRC}_{SIDE}'] = pd.read_csv(stat_path,
                                                        header=0, index_col=0)
                continue

            ## calculate grouped stat data

            # load PSD classes (1sec arrays) only once
            if not CLASSES_LOADED:
                print('...load get_allSpecStates_Psds() to calc stat-psds')
                PSDs, BLs = get_allSpecStates_Psds(RETURN_PSD_1sec=True)
                CLASSES_LOADED = True
            
            # get stat-data (if STAT_LID_COMPARE == categs, stat_labels are coded 0: no, 1: mild, 2: moderate, 3: severe
            (
                stat_values, stat_labels, stat_ids, value_freqs
            ) = get_MOVE_stat_grouped_data(
                MOVE_COND=MOV, SRC=SRC,
                STAT_SPLIT=STAT_LID_COMPARE,
                PSD_DICT=PSDs, BL_class=BLs,  # USE 1-SEC CLASSES FOR STATISTICS
                PSD_1s_windows=True,
            )
            print(f'keys of received stat_values: {stat_values.keys()}, search {SIDE}')
            # adjust SIDE to keys, i.e., 'INVOLUNT_CONTRA', 'INVOLUNT_IPSI', 'INVOLUNT_BILAT'
            if f'{MOV}_{SIDE}' in list(stat_values.keys()):
                SIDE = f'{MOV}_{SIDE}'
            else:
                print(f'no matching SIDE VALUES for')
                continue

            stat_values = stat_values[SIDE]
            stat_labels = stat_labels[SIDE]
            stat_ids = stat_ids[SIDE]

            # COMPARE WITH all REST WITHOUT MOVEMENT (labels all 0)
            if REST_BASELINE:
                print('...load rest psds as baseline for movement')
                (
                    bl_values, bl_labels, bl_ids, value_freqs
                ) = get_REST_stat_grouped_data(
                    STAT_SPLIT='move_baseline',
                    SRC=SRC,
                    PSD_DICT=PSDs, BL_class=BLs,
                    PSD_1s_windows=True,
                    MERGE_STNs=True,  # merge STNs to get src lfp
                )
                if MOV == 'TAP':
                    stat_labels += 1  # increase labels to create 0 for baseline
                # add baseline values to stat values
                stat_values = np.concatenate([stat_values, bl_values])
                stat_labels = np.concatenate([stat_labels, bl_labels])
                stat_ids = np.concatenate([stat_ids, bl_ids])
                
            # get STATS (coeffs, sign-bools) based on grouped data
            # internal dealing with LID categories if necessary
            (
                (coeffs_freqs, sign_freqs), stat_freqs
            ) = psd_Stats.calc_lmem_freqCoeffs(
                temp_values=stat_values,
                temp_ids=stat_ids,
                temp_freqs=value_freqs,
                VALUES_GIVEN_GROUPED=True,
                GROUP_LABELS=stat_labels,
                STATS_PER_LID_CAT=STAT_PER_LID_CAT
            )
            # CALC and SAVE STATS in dataframes separate per MOV / SRC / SIDE
            if not STAT_PER_LID_CAT:
                stat_df = pd.DataFrame(
                    index=stat_freqs,
                    columns=[f'coef_{STAT_LID_COMPARE}',
                             f'sig_{STAT_LID_COMPARE}'])
                stat_df.iloc[:, 0] = coeffs_freqs
                stat_df.iloc[:, 1] = sign_freqs

                stat_df.to_csv(stat_path, header=True, index=True)
                stat_dfs[f'{SRC}_{SIDE}'] = stat_df
            
            elif STAT_PER_LID_CAT:
                stat_df = pd.DataFrame(index=stat_freqs,)
                for cat in coeffs_freqs.keys():
                    # create coef and sig columns corr with split and category
                    stat_df[f'coef_{cat}lid'] = coeffs_freqs[cat]
                    stat_df[f'sig_{cat}lid'] = sign_freqs[cat]

                print(f'\n...SAVING STAT DF: {df_name} with keys: '
                      f'{stat_df.keys()} (in {stat_dir})')
                stat_df.to_csv(os.path.join(stat_dir, df_name),
                                header=True, index=True)

        # PLOT
        if PLOT_STATS and not STAT_PER_LID_CAT:
            plot_move_lmm(MOV, stat_dfs, SAVE_FIG=True,
                          FIGNAME=f'1901_LMMcoeffs_{MOV}_{STAT_LID_COMPARE}')