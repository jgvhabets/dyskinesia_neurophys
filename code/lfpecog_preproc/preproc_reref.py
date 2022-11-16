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
from lfpecog_features.feats_helper_funcs import nan_array


def LeadLevels():
    '''
    Creates object (namedtuple) with segmented-contact
    levels of all Leads implanted in patients
    participating in this project.
    From anatomical point of view: Zero starts caudal/
    inferior/lower, conform default lead-nomenclature.

    Arguments: None

    Returns:
        - Leads (namedtuple containing level-dicts
            with contact-indices per level)
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
        
    Returns
        - ...
        - levels_num and levels_str return dict with available
            channels per levels of the electrode 
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

        self.levels_str = {}
        for l in self.levels_num:  # loops over keys (=levels)
            nums = self.levels_num[l]  # takes num-list
            # creates list per level with string-names index-0 -> LFP_L_01, .., i-5 -> LFP_L_06, etc
            self.levels_str[l] = [
                f'LFP_{self.side}_{(c + 1):02}' for c in nums]  # not included trailing '_'

        # exclude channels not in clean-channel list
        clean_combi_str = '\t'.join(self.chs_clean)  # all strings to one
        self.level_rows = {}
        
        for l in self.levels_str:  # loops over levels
        
            to_del_num, to_del_str = [], []  # channels to remove per level
            for ch_num, ch_str in zip(self.levels_num[l], self.levels_str[l]):
                # if ch-name is not in clean-channel combi-str
                if ch_str.lower() not in clean_combi_str.lower():
                    to_del_num.append(ch_num)
                    to_del_str.append(ch_str)

            for c in to_del_num: self.levels_num[l].remove(c)
            for c in to_del_str: self.levels_str[l].remove(c)


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
    data: array,
    lead: Any, 
    time_rowNames: list,
    ephys_rowNames: list,
    ephys_type: str,
    report='',
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
        - time_rowNames: list with 1 or 2 time col-names
        - ephys_type: LFP or ECOG
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

    n_timerows = len(time_rowNames)
    print(time_rowNames)

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

    print(f'Present levels: {present_levels}, shape leveldata: {leveldata.shape}')
    
    assert (len(present_levels) + n_timerows
        == leveldata.shape[-2]), print(
        '# of present levels is not equal to # of '
        f'rows in leveldata-array'
    )
    
    rerefdata, names, report = reref_neighb_levels_subtract(
        leveldata=leveldata,
        present_lfp_levels=present_levels,
        present_ecog_contacts=None,
        side=lead.side,
        time_rowNames=time_rowNames,
        ephys_type=ephys_type,
        report=report,
    )
    assert rerefdata.shape[-2] == len(names), print(
        'Rereferenced #-ROWs in ARRAY NOT EQUAL to #-NAMES'
    )

    if report:
        report += f'\n\n\tNew resulting signal names: {names}\n'

    return rerefdata, names, report


def reref_neighb_levels_subtract(
    leveldata: array,
    present_lfp_levels: list,
    present_ecog_contacts: list,
    side: str,
    time_rowNames: list,
    ephys_type: str,
    report = '',
):
    """
    Function to calculate differences between neighbouring
    eletrode levels. These can come from unsegmented
    electrodes, or from segmented electrodes via the
    reref_summing_levels() function.
    
    Arguments:
        - level (array): 2d or 3d array containing
            [(windows), times- and levels-rows, samples]
        - present_lfp_levels (list): list with present
            LFP level-numbers
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
    sig_names = time_rowNames

    if len(leveldata.shape) == 3:

        newdata = nan_array(dim=[
            leveldata.shape[0],
            leveldata.shape[1] - 1,
            leveldata.shape[2],
        ])  # create nan array to fill

        newdata[:, :len(time_rowNames), :] = leveldata[
            :, :len(time_rowNames), :
        ]  # copy time rows
        
        # subtract neighbour-level for all levels w/ neighbour 
        newdata[:, len(time_rowNames):, :] = np.subtract(
            leveldata[:, len(time_rowNames):-1, :],
            leveldata[:, len(time_rowNames) + 1:, :]
        )
    
    if len(leveldata.shape) == 2:

        newdata = nan_array(dim=[
            leveldata.shape[0] - 1,
            leveldata.shape[1]
        ])  # create nan array to fill

        # copy time rows
        newdata[:len(time_rowNames), :] = leveldata[
            :len(time_rowNames), :
        ]
        # subtract the level above one level from every level
        newdata[len(time_rowNames):, :] = np.subtract(
            leveldata[len(time_rowNames):-1, :],
            leveldata[len(time_rowNames) + 1:, :]
        )

    # create corresponding names
    if ephys_type == 'LFP':
        for i, level in enumerate(present_lfp_levels[:-1]):

            sig_names.append(f'LFP_{side}_{level}_{present_lfp_levels[i + 1]}')
    
    elif ephys_type == 'ECOG':
        
        for con1, con2 in zip(
            present_ecog_contacts[:-1],
            present_ecog_contacts[1:]
        ):
            sig_names.append(f'ECOG_{side}_{con1}_{con2}')

        if report:

            report += (
                f'\nNeighbour Rereferencing: ECoG {side}:\n\t'
                f'raw present ECoG ephys contact numbers: '
                f'{present_ecog_contacts},\n\n\t'
                f'New resulting signals names: {sig_names}'
            )


    return newdata, sig_names, report


def reref_two_contacts(
    name1, name2, array_names, data_array
):
    index1 = np.where([name1 in n for n in array_names])[0][0]
    data1 = data_array[index1, :]
    index2 = np.where([name2 in n for n in array_names])[0][0]
    data2 = data_array[index2, :]

    rerefd_ch = data1 - data2
    rerefd_name = f'{name1}_{name2[-2:]}'

    return rerefd_ch, rerefd_name



def reref_segm_contacts(
    data: array,
    lead: Any,
    time_rowNames: list,
    ephys_rowNames: list,
    ephys_type: str,
    report: str = '',
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
            f'\nSegmented LFP-rereferencing: {lead.name}'
            f'({lead.side})\n\tevery single segment is '
            'referenced against its neighbouring contact,'
            'inter- AND intra-level (segment or ring)\n\n'
            f'present segment per level: {lead.levels_str}\n'
        )

    print(f'\n SEGMENTED REREFFF {lead.name} ({lead.side})')

    array_names = time_rowNames + ephys_rowNames
    # every contact gets it rereferenced output
    # reref_data = nan_array(list(data.shape))

    # START reref data with time-rows
    if len(data.shape) == 3:
        reref_data = data[
            :, :len(time_rowNames), :]  # copy time row(s)  [:, :len(time_rowNames), :]

    if len(data.shape) == 2:
        reref_data = data[
            :len(time_rowNames), :]  # copy time row(s)   [:len(time_rowNames), :]

    reref_names = time_rowNames

    for l in lead.levels_str.keys():  # loop over level

        level_chs = lead.levels_str[l]  # row numbers of channels
        # skip if not consisting any channel
        if len(level_chs) == 0: continue

        # take one level higher as level to reref with
        if l < len(lead.levels_str) - 1: ref_level = l + 1
        # for most upper level, take lowest level to reref with
        elif l == len(lead.levels_str) - 1: ref_level = 0

        # INTRA-LEVEL REREF
        if len(level_chs) > 1:  # if more than one present segment on level
            # reref and add correct array indices based on channel names
            rerefd_ch, rerefd_name = reref_two_contacts(
                level_chs[0], level_chs[1], array_names, data)
            reref_data = np.vstack([reref_data, rerefd_ch])
            reref_names.append(rerefd_name)
 
        if len(level_chs) > 2:  # if three channels avail on level
            for name1, name2 in zip(
                level_chs[1:], [level_chs[2], level_chs[0]]
            ):  # reref second vs third, and third vs first
                rerefd_ch, rerefd_name = reref_two_contacts(
                    name1, name2, array_names, data)
                reref_data = np.vstack([reref_data, rerefd_ch])
                reref_names.append(rerefd_name)

        # INTER-LEVEL REREF (always use level above as ref)
        if len(level_chs) > 1:  # if more than one present segment on level
            for c, chname in enumerate(level_chs):

                try:
                    ref_ch = lead.levels_str[ref_level][c]
                except IndexError:  # if segment directly vertical not present 
                    ref_ch = lead.levels_str[ref_level][0]
                
                rerefd_ch, rerefd_name = reref_two_contacts(
                    chname, ref_ch, array_names, data)
                reref_data = np.vstack([reref_data, rerefd_ch])
                reref_names.append(rerefd_name)
        
        elif len(level_chs) == 1:

            chname = level_chs[0]
            
            for ref_ch in lead.levels_str[ref_level]:

                rerefd_ch, rerefd_name = reref_two_contacts(
                    chname, ref_ch, array_names, data)
                reref_data = np.vstack([reref_data, rerefd_ch])
                reref_names.append(rerefd_name)

            
    if report:
        report += (
            f'\nResulting reref"d channel names: {reref_names}\n\t'
            f'{len(reref_names)} channels, data is shape {reref_data.shape}\n'
        )


    return reref_data, reref_names, report


def main_rereferencing(
    dataDict: dict,
    chNameDict,
    runInfo: Any,
    reref_setup: dict,
    reportPath: str='',
):
    """
    Function to execute rereferencing of LFP and ECoG data.
    
    Arguments:
        - dataDict: containing dict w/ groups
        - chNameDict: dict with lists per group with
            clean channels after artefact removal (incl times)
        - runInfo: class containing runInfolead_type:
            abbreviation for lead given in
            data class RunInfo (e.g. BSX, MTSS)
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

        if np.logical_and(
            'lfp' not in group.lower(),
            'ecog' not in group.lower()
        ):
            
            report += f'\n Skip REREFERENCING for {group}\n'

            rerefData[group] = dataDict[group]
            rerefNames[group] = chNameDict[group]

            continue
    
        else:
            report += f'\n START REREFERENCING for {group.upper()}'
        
        time_rowNames = [name for name in chNameDict[group] if 'time' in name]
        ephys_rowNames = [name for name in chNameDict[group] if 'time' not in name]

        if 'left' in group: side = 'L'
        elif 'right' in group: side = 'R'

        if 'ecog' in group.lower():

            ephys_type = 'ECOG'
            
            if reref_setup[ephys_type] == 'common-average':
                rerefData[group], rerefNames[group], report = reref_common_average(
                    data=dataDict[group], ch_names=chNameDict[group],
                    group=group, report=report, timeRows=time_rowNames,
                )
            elif reref_setup[ephys_type] == 'bip-neighbour':
                ecog_contact_nrs = [
                    cName.split('_')[2] for cName in ephys_rowNames
                ]  # take ecog contact numbers (eg 01) from name (eg ECOG_L_01)
                rerefData[group], rerefNames[group], report = reref_neighb_levels_subtract(
                    leveldata=dataDict[group],
                    present_lfp_levels=None,
                    present_ecog_contacts=ecog_contact_nrs,
                    side=side,
                    time_rowNames=time_rowNames,
                    ephys_type=ephys_type,
                    report=report,
                )

        elif 'lfp' in group.lower():
            
            ephys_type = 'LFP'

            lead_type = runInfo.lead_type
            lead = Segm_Lead_Setup(lead_type, side, chNameDict[group])
            # print(lead.levels_num)
            # print(lead.levels_str)
            # print(dataDict[group].shape, time_rowNames, ephys_rowNames)
            # break
            
            if reref_setup[ephys_type] == 'levels':
                reref_function = reref_segm_levels
            elif reref_setup[ephys_type] == 'segments':
                reref_function = reref_segm_contacts

            rerefData[group], rerefNames[group], report = reref_function(
                data=dataDict[group],
                lead=lead,
                time_rowNames=time_rowNames,
                ephys_rowNames=ephys_rowNames,
                report=report,
                ephys_type='LFP',
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