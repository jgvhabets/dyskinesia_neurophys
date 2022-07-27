"""
Functions to re-reference ECoG and LFP signals.
Some relevant literature regarding rereferencing:
Liu ea, J Neural Eng 2015:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5485665/;
Dubey, J Neurosc 2019:
https://www.jneurosci.org/content/39/22/4299

"""

# Import packages and functions
from array import array
from itertools import compress
from collections import namedtuple
from dataclasses import dataclass
from typing import Any
import numpy as np

# Import own functions
from tapping_feat_calc import nan_array


def LeadLevels():
    '''
    Creates object (namedtuple) with segmented-contact
    levels of all Leads implanted in patients
    participating in this project.
    From anatomical point of view: Zero starts caudal/
    inferior/lower, conform default lead-nomenclature.

    Arguments: None

    Returns:
        - Leads (namedtuple containing level-dicts)
    '''
    LeadLevels =  namedtuple(
        'LeadLevels', 
        'BS_VC_X MT_SS BS_VC'  # extent when necessary
    )
    Leads = LeadLevels(
        
        {  # first: BS_VC_X Boston Sc Verc Cart X
            0: [0, 1, 2],  # lowest/deepest in brain
            1: [3, 4, 5],
            2: [6, 7, 8],
            3: [9, 10, 11],
            4: [12, 13, 14],
            5: [15, ]  # highest, closest to skull
        },
        
        {  # second: MT_SS: MedTronic SenseSight
            0: [0, ],  # lowest
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, ]   # highest
        },
        
        {  # third: BS_VC: Boston Sc Vercise Cart
            0: [0, ],  # lowest
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, ]   # highest
        },
    
    )

    return Leads


@dataclass
class Segm_Lead_Setup:
    """
    Class which stores levels and contacts of different
    DBS-electrode types and brands.
    
    Required Arguments:
        - code: abbreviation of lead-type to identify, is given
        in the RunInfo Class at start of preprocessing flow
        - side: L or R
        - chs_clean: list with channel names after artefact removal
    """
    code: str
    side: str
    chs_clean: list
    
    def __post_init__(self):
        Leads = LeadLevels()  # create Leads-tuple

        fullnames = {
            'BS_VC_X': 'BS Vercise Cartesia X',
            'MT_SS': 'MT SenseSight',
            'BS_VC': 'BS Vercise Cartesia',
        }

        self.name = fullnames[self.code]
        self.levels_num = getattr(Leads, self.code)
        self.n_levels = len(self.levels_num)
        self.n_contacts = sum(
            [len(self.levels_num[l]) for l in self.levels_num]
        )  # sums the length of every level-list in the dict

        # lists of all contact-names per level (regardless clean)
        contacts = [f'LFP_{self.side}_{str(n).zfill(2)}_'
            for n in np.arange(1, self.n_contacts + 1)]

        self.levels_str = {}

        for l in self.levels_num:  # loops over keys (=levels)
            nums = self.levels_num[l]  # takes num-list
            # creates list per level with string-names
            self.levels_str[l] = [contacts[c] for c in nums]

        # exclude channels not in clean-channel list
        combi_str = '\t'.join(self.chs_clean)  # all strings to one
        self.level_rows = {}

        startrow = 1  # row 0 is time
        
        for l in self.levels_str:  # loops over levels
        
            to_del = []  # channels to remove per level
            for c in self.levels_str[l]:
                # if ch-name is not in clean-channel combi-str
                if c not in combi_str:
                    to_del.append(c)

            for c in  to_del: self.levels_str[l].remove(c)

            # create dict's with rows of data
            self.level_rows[l] = list(np.arange(
                startrow, startrow + len(self.levels_str[l])
            ))
            startrow += len(self.levels_str[l])

            assert (len(self.level_rows[l]) == 
                len(self.levels_str[l])), 'Wrong level-dicts'
            # keeps only clean channels in these dict's


def reref_common_average(
    data, group, ch_names, timeRows, report='',
):
    """
    Function to perform Common Average Rereferencing.
    Default in ECoG signal processing.

    Arguments:
        - data (array): 3d array containing time- and signal-
        rows
        - ch_names (list): strings of clean channel names,
        incl time-row-names, excl removed channels
    
    Returns:
        - newdata (array): identical shape array as input,
        signals re-referenced by extracting the mean over
        all signals for each channel
    """
    newdata = nan_array(list(data.shape))  # was empty_like
    newnames = [n[:9] for n in ch_names]
    
    if report:
        report += (
            f'\n\n\tCommon Average Reref ({group})\n'
            f'\nRaw-Names: {ch_names}\n\n'
            f'Names after Rereferencing: {newnames}\n'
        )

    if len(data.shape) == 3:
### FIX timeRows
        newdata[:, 0, :] = data[:, 0, :]  # copy time rows
        for w in np.arange(data.shape[0]):  # loop over windows
            refdata = data[w, 1:, :].copy()
            for ch in np.arange(refdata.shape[1]):  # chann to reref
                # take window average of all other channels
                refchs = np.delete(refdata.copy(), ch, axis=0)
                ref_mean = np.nanmean(refchs, axis=0)
                newdata[w, ch + 1, :] = data[w, ch, :] - ref_mean
        
    elif len(data.shape) == 2:

        newdata[:len(timeRows), :] = data[:len(timeRows), :]  # copy time row
        refdata = data[len(timeRows):, :].copy()

        for ch in np.arange(refdata.shape[0]):
            # take window average of all other channels
            ref_chs = np.delete(refdata.copy(), ch, axis=0)
            ref_mean = np.nanmean(ref_chs, axis=0)
            newdata[ch + len(timeRows), :] = refdata[ch, :] - ref_mean


    return newdata, newnames, report


def reref_segm_levels(
    data: array, lead: Any, 
    timerowNames, report='',
):
    """
    Function to aggregate/sum all segmented electrode contacts
    which belong to same level. The function takes into
    account which contact are empty/bad/removed during arte-
    fact removal.
    
    Arguments:
        - data (array): 3d or 2d data array
        - lead (Class): class with info of segmented contacts
        per level
        - timerowNames: list with 1 or 2 time col-names
        - report_file  (str -> path): file incl directory where rereferencing
        notes are written to

    Returns:
        - reref_data (array): rereferenced signals
        - names (list): list with corresponding ch-names
    """
    if report:

        report += (
            f'\nRereferencing: {lead.name} ({lead.side}) '
            'per total level minus neighbouring level'
        )

    print(f'\nRereferencing SEGM {lead.name} ({lead.side})'
          f' against means of neighbouring levels')

    n_timerows = len(timerowNames)
    print(timerowNames)

    if len(data.shape) == 3:

        leveldata = nan_array(dim=[
            data.shape[0],
            lead.n_levels + n_timerows,
            data.shape[2]
        ])
        leveldata[:, :n_timerows, :] = data[:, :n_timerows, :]
        only_data=data[:, n_timerows:, :]

    if len(data.shape) == 2:

        leveldata = nan_array(dim=[
            lead.n_levels + n_timerows,
            data.shape[1]
        ])

        leveldata[:n_timerows, :] = data[:n_timerows, :]
        only_data=data[n_timerows:, :]

    ch_start, ch_stop = 0, 0
    present_levels = []
    n_empty_levels = 0
    row_tofill = n_timerows

    for l in np.arange(lead.n_levels):
        
        ch_stop += len(lead.levels_str[l])
        # starts at 0, incl numb of clean contacts per level

        if len(lead.levels_str[l]) == 0:
            print(f'{l} EMPTY LEVEL')
            n_empty_levels += 1
            if len(leveldata.shape) == 2:
                leveldata = np.delete(leveldata, -1, 0)  # delete last row

            if report:
                report += f'\nLevel {lead.side, l} is empty'
            continue

        if len(only_data.shape) == 3:    
            
            leveldata[:, row_tofill, :] = np.nanmean(
                only_data[:, ch_start:ch_stop, :], axis=1
            )

        if len(only_data.shape) == 2:

            leveldata[row_tofill, :] = np.nanmean(
                only_data[ch_start:ch_stop, :], axis=0
            )

        if report:
            report += (  # documenting incl contacts per level
                f'\nLevel {lead.side, l} '
                f'contains row indices {ch_start}:{ch_stop}'
                f', with corr contacts: {lead.levels_str[l]}'
            )

        print(  # documenting incl contacts per level
            f'\nLevel {lead.side, l} '
            f'contains row indices {ch_start}:{ch_stop}'
            f', with corr contacts: {lead.levels_str[l]}'
        )
        
        ch_start = ch_stop  # updating start row
        row_tofill += 1
        present_levels.append(l)

    # print(f'leveldata shape: {leveldata.shape}, # present-levels {len(present_levels)}'
    #     f', # empty-levels: {n_empty_levels}')
    print(f'Present levels: {present_levels}, shape leveldata: {leveldata.shape}')
    print(leveldata[:, :3])
    # if n_empty_levels > 0:
    #     leveldata = leveldata[:-n_empty_levels, :]
    #     print('after empty level removal')
    #     print(leveldata[:, :3])
    
    assert (len(present_levels) + n_timerows
        == leveldata.shape[-2]), print(
        '# of present levels is not equal to # of '
        f'rows in leveldata-array'
    )
    
    rerefdata, names = reref_neighb_levels_subtract(
        leveldata=leveldata, present_levels=present_levels,
        side=lead.side, timerowNames=timerowNames,
    )
    assert rerefdata.shape[-2] == len(names), print(
        'Rereferenced #-ROWs in ARRAY NOT EQUAL to #-NAMES'
    )

    if report:
        report += f'\n\n\tNew resulting names: {names}\n'

    return rerefdata, names, report


def reref_neighb_levels_subtract(
    leveldata: array, present_levels: list, side: str,
    timerowNames: list,
):
    """
    Function to calculate differences between neighbouring
    eletrode levels. These can come from unsegmented
    electrodes, or from segmented electrodes via the
    reref_summing_levels() function.
    
    Arguments:
        - level (array): 2d or 3d array containing
            [(windows), times- and levels-rows, samples]
        - present_levels (list): list with present level-
            numbers
        - side (str): L or R for recorded hemisphere
        - timerowNames: list with 1 or 2 time col-names

    Returns:
        - newdata (array): rereferenced signals. These
            will contain 1 row less because the
            differences between rows are used
        - sig_names (list): contains corresponnding names
            to the array: 'times', followed by the
            reref'd level-diffs, e.g. 0_1, 3_4
    """
    sig_names = timerowNames

    if len(leveldata.shape) == 3:

        newdata = nan_array(dim=[
            leveldata.shape[0],
            leveldata.shape[1] - 1,
            leveldata.shape[2],
        ])  # create nan array to fill

        newdata[
            :, :len(timerowNames), :] = leveldata[
                :, :len(timerowNames), :
        ]  # copy time rows
        
        newdata[  # subtract neighbour-level for all levels w/ neighbour 
            :, len(timerowNames):, :
        ] = np.subtract(
            leveldata[:, len(timerowNames):-1, :],
            leveldata[:, len(timerowNames) + 1:, :]
        )
    
    if len(leveldata.shape) == 2:

        newdata = nan_array(dim=[
            leveldata.shape[0] - 1,
            leveldata.shape[1]
        ])  # create nan array to fill

        newdata[
            :len(timerowNames), :] = leveldata[
            :len(timerowNames), :
        ]  # copy times

        newdata[len(timerowNames):, :] = np.subtract(
            leveldata[len(timerowNames):-1, :],
            leveldata[len(timerowNames) + 1:, :]
        )  # subtract the level above one level from every level

    for i, level in enumerate(present_levels[:-1]):

        sig_names.append(f'LFP_{side}_{level}_{present_levels[i + 1]}')

    return newdata, sig_names


def reref_segm_contacts(
    data: array, lead: Any, timerowNames,
    report: str='',
):
    """
    Function for local spatial specific rereferencing
    of segmented-lead contacts.

    Arguments:
        - data (array): 3d or 2d data array with data
            from one group (lfp R or L)
        - lead: class with info of segmented contacts
        per level
        - timerowNames: list with 1 or 2 time col-names
        - report_file (str -> path): file incl directory where rereferencing
        notes are written to

    Returns:
        - reref_data (array): rereferenced signals
        - names (list): list with corresponding ch-names
    """
    if report:
        report += (
            f'\nRereferencing: {lead.name} ({lead.side}) '
            'per segmented contact (segment 1 minus '
            'averaged other segm-contacts of same level'
        )

    print(f'\n Rereferencing SEGM {lead.name} ({lead.side})'
          f' per SEGM-CONTACT against mean of own level')

    # every contact gets it rerferenced output
    reref_data = nan_array(list(data.shape))  # np.empty_like((data))

    if len(data.shape) == 3:
        reref_data[:, 0, :] = data[:, 0, :]  # copy time row

    if len(data.shape) == 2:
        reref_data[0, :] = data[0, :]  # copy time row

    rerefnames = timerowNames

    for l in lead.level_rows:  # loop over level
        chs = lead.level_rows[l]  # row numbers of channels

        if len(chs) > 1:  # if avail neighb channel on level

            for n, c in enumerate(chs):  # loop over contacts

                refs = chs.copy()  # channels to use as ref
                refs.remove(c)  # remove ch of interest itself

                # take mean of other channel-rows as reference
                if len(data.shape) == 3:
                    ref = np.nanmean(data[:, refs, :], axis=1)
                    reref_data[:, c, :] = data[:, c, :] -  ref

                elif len(data.shape) == 2:
                    ref = np.nanmean(data[refs, :], axis=0)
                    reref_data[c, :] = data[c, :] -  ref

                rerefnames.append(lead.levels_str[l][n])
                # done, only test + put lfp_reref in filename!!
                
                if report:
                    report += (
                        f'\n(level {l}) contact-row {c} is '
                        f'rereferenced against rows {refs}'
                    )

        elif len(chs) == 1:  # if not ring or 1 ch available

            try:  # reref against a neighbouring level
                refs = lead.level_rows[l + 1]  # reref: sup. level
            except KeyError:  # highest contact has no [l + 1]
                refs = lead.level_rows[l - 1]  # take lower level

            if len(data.shape) == 3:
                ref = np.nanmean(data[:, refs, :])
                reref_data[:, chs, :] = data[:, chs, :] -  ref
            if len(data.shape) == 2:
                ref = np.nanmean(data[refs, :])
                reref_data[chs, :] = data[chs, :] -  ref
            
            rerefnames.append(lead.levels_str[l][0])
            
            if report:
                report += (
                    f'\n(level {l}) Contact {lead.levels_str[l]}'
                    f' is rereferenced against {refs}'
                )

        else:
            if report:
                report += (
                    f'\n(level {l}) has no valid channels'
                    f' {lead.levels_str[l]}'
                )

    return reref_data, rerefnames, report


def main_rereferencing(
    dataDict: dict, chNameDict, runInfo: Any,
    lfp_reref: str, reportPath: str='',
):
    """
    Function to execute rereferencing of LFP and ECoG data.
    
    Arguments:
        - data: 2d or 3d-array, containing dict w/ groups
        - runInfo: class containing runInfolead_type:
            abbreviation for lead given in
            data class RunInfo (e.g. BSX, MTSS)
        - chs_clean = list with clean channels after
            artefact removal
        - lfp_reref: indicates string-name of reref-
            method: taking together one level, or re-
            referencing single segmented contacts
    
    Returns:
        - datareref: 3d array of rereferenced data of
            inserted signal group. 'time' is time since
            start recording (in sec), 'dopa_time' is
            time relative to L-Dopa intake (in sec)
        - names (list): strings of row names, 'time',
            'dopa_time' and all reref'd signal channels
    """
    report = (
        '\n\n======= REREFERENCING OVERVIEW ======\n'
    )

    rerefData = {}
    rerefNames = {}

    for group in dataDict.keys():

        if group[:3] not in ['lfp', 'eco']:
            
            report += f'\n Skip REREFERENCING for {group}\n'

            rerefData[group] = dataDict[group]
            rerefNames[group] = chNameDict[group]

            continue
    
        else:
            report += f'\n START REREFERENCING for {group.upper()}'
        
        timerow_sel = ['time' in name for name in chNameDict[group]]
        timerowNames = list(compress(chNameDict[group], timerow_sel))

        if group[:4] == 'ecog':
            
            rerefData[group], rerefNames[group], report = reref_common_average(
                data=dataDict[group], ch_names=chNameDict[group],
                group=group, report=report, timeRows=timerowNames,
            )

        elif group[:3] == 'lfp':
            
            if group[4:] == 'left': side = 'L'
            if group[4:] == 'right': side = 'R'

            lead_type = runInfo.lead_type
            lead = Segm_Lead_Setup(lead_type, side, chNameDict[group])
            
            if lfp_reref == 'levels': reref_function = reref_segm_levels
            elif lfp_reref == 'segments': reref_function = reref_segm_contacts

            rerefData[group], rerefNames[group], report = reref_function(
                data=dataDict[group],
                lead=lead,
                timerowNames=timerowNames,
                report=report,
            )

        rerefData[group], rerefNames[group], report = removeNaNchannels(
            rerefData[group], rerefNames[group],
            group, report,
        )

    if reportPath:

        with open(reportPath, 'a') as f:

            f.write(report)
            f.close()


    return rerefData, rerefNames



def removeNaNchannels(
    data, chnames, groupname, report
):
    """
    Quality check, delete only nan-channels
    (channels which only contain NaNs)

    Checks group data, not dictionary with groups
    """
    # 
    ch_del = []
    for ch in np.arange(1, data.shape[-2]):

        if len(data.shape) == 3:
            if not (np.isnan(
                data[:, ch, :]
            ) == False).any():
                ch_del.append(ch)

        elif len(data.shape) == 2:
            if not (np.isnan(
                data[ch, :]
            ) == False).any():
                ch_del.append(ch)
    
    data = np.delete(data, ch_del, axis=-2)

    if ch_del:
        names_del = [chnames[c] for c in ch_del]
        for name in names_del: chnames.remove(name)

        if report:

            report += (
                f'\n\n Auto Cleaning:\n In {groupname}: row(s) {ch_del} ('
                f'{names_del}) only contained NaNs'
                ' and were deleted\nRemaining number'
                f' of rows (incl time) is {data.shape[-2]}'
                '. If 1: group will be removed!'
            )[0]
    
    return data, chnames, report