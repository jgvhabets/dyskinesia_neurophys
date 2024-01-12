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
import numpy as np
from itertools import product

# import own functions
from lfpecog_plotting.plot_move_spec_psd import (
    prep_REST_spec_psds,
    prep_MOVEMENT_spec_psds
)

def get_REST_stat_grouped_data(
    PSD_DICT, BL_class,
    STAT_SPLIT: str,
    SRC: str = 'lfp_left',
    PSD_1s_windows: bool = False,
):
    # DEFINE STAT GROUPS
    if STAT_SPLIT == 'no-LID vs all-LID':
        group_states = {0: ['nolidbelow30', 'nolidover30'],
                        1: ['mildlid', 'moderatelid', 'severelid']}
    elif STAT_SPLIT == 'no-LID (<30) vs all-LID':
        group_states = {0: ['nolidbelow30'],
                        1: ['mildlid', 'moderatelid', 'severelid']}

    group_subs = {i: [] for i in group_states.keys()}
    group_psds = {i: [] for i in group_states.keys()}

    # GET PSD ARRAYS
    (psd_arrs,
     psd_freqs,
     psd_subs) = prep_REST_spec_psds(
        PSD_DICT=PSD_DICT,
        BASELINE=BL_class,
        RETURN_IDS=True)
    src_psd = psd_arrs[SRC]
    src_subs = psd_subs[SRC]

    # put data from diff LID-states in correct groups
    for group in group_states:
        for lid_stat in group_states[group]:

            temp_psd = np.array(src_psd[lid_stat])
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
        print(f'...created PSD-group {group}: {group_psds[group].shape} correct')

    stat_values = np.concatenate([group_psds[i] for i in group_states.keys()])
    stat_labels = np.concatenate([[i] * len(group_subs[i]) for i in group_states.keys()])
    stat_ids = np.concatenate([np.array(group_subs[i]) for i in group_states.keys()])

    return stat_values, stat_labels, stat_ids, psd_freqs


def get_MOVE_stat_grouped_data(
    PSD_DICT, BL_class,
    MOVE_COND: str,
    STAT_SPLIT: str = 'no-LID vs all-LID',
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
    if STAT_SPLIT == 'no-LID vs all-LID':
        group_states = {0: ['nolid'],
                        1: ['mildlid', 'moderatelid', 'severelid']}
    elif STAT_SPLIT == 'LID linear':
        group_states = {0: ['nolid'], 1: ['mildlid'], 
                        2: ['moderatelid'], 3: ['severelid']}

    # GET PSD ARRAYS (move states are dependent on move_cond)
    (psd_arrs,
     psd_freqs,
     psd_subs) = prep_MOVEMENT_spec_psds(
        PLOT_MOVE=MOVE_COND,
        PSD_DICT=PSD_DICT,
        BASELINE=BL_class,
        RETURN_IDS=True)
    src_psd = psd_arrs[SRC]  # contains CONTRA and IPSI
    src_subs = psd_subs[SRC]

    MOVE_STATES = list(src_psd.keys())
    
    group_subs = {m : {i: [] for i in group_states.keys()}
                  for m in MOVE_STATES}
    group_psds = {m : {i: [] for i in group_states.keys()}
                  for m in MOVE_STATES}

    # put data from diff LID-states in correct groups
    for MOV, group in product(MOVE_STATES, group_states):
        print(f'START group {group} for movement: {MOV}')
        for lid_stat in group_states[group]:
            # print(f'...lid label: {lid_stat}')

            temp_psd = np.array(src_psd[MOV][lid_stat])
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
            print(f'DELETE empty psd for {MOV} group: {group}')
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
        # print(f'...created PSD-group {group}: {group_psds[MOV][group].shape} correct')

    # dicts to return
    stat_values, stat_labels, stat_ids = {}, {}, {}
    for MOV in MOVE_STATES:
        stat_values[MOV] = np.concatenate([group_psds[MOV][i] for i in group_psds[MOV].keys()])
        stat_labels[MOV] = np.concatenate([[i] * len(group_subs[MOV][i]) for i in group_states.keys()])
        stat_ids[MOV] = np.concatenate([np.array(group_subs[MOV][i]) for i in group_states.keys()])

    return stat_values, stat_labels, stat_ids, psd_freqs


