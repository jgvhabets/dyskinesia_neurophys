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
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any
import numpy as np

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
        'BSX MTSS'  # extent when necessary
    )
    Leads = LeadLevels(
        {  # first is Boston Sc Verc Cart X
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
        3: [9, 10, 11],
        4: [12, 13, 14],
        5: [15, ]
        },
        {  # second is MedTronic SenseSight
        0: [0, ],
        1: [1, 2, 3],
        2: [4, 5, 6],
        3: [7, ]   
        })

    return Leads


@ dataclass
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
    name: str = None  # full description of lead-type
    n_levels: int = None  # number of levels on lead
    n_contacts: int = None  # number of total contacts
    levels_str: dict = None  # dict with channel-names per level
    levels_num: dict = None  # numerical lead-setup per level
    level_rows: dict = None  # correspoding data array-rows
    # in clean channels in level_str
    
    def __post_init__(self):
        Leads = LeadLevels()  # create Leads-tuple
        fullnames = {'BSX': 'BS Vercise Cartesia X',
                     'MTSS': 'MT SenseSight',
                     }
        self.name = fullnames[self.code]
        self.levels_num = getattr(Leads, self.code)
        self.n_levels = len(self.levels_num)
        self.n_contacts = sum(
            [len(self.levels_num[l]) for l in self.levels_num]
        )  # sums the length of every level-list in the dict

        # lists of all contact-names per level (regardless clean)
        contacts = [f'LFP_{self.side}_{n}_'
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


def reref_common_average(data, ch_names):
    """
    Function to perform Common Average Rereferencing.
    Default in ECoG signal processing.

    Arguments:
        - data (array): 3d array containing time- and signal-
        rows
        - ch_names (list): strings of clean channel names,
        incl 'times', excl removed channels
    
    Returns:
        - newdata (array): identical shape array as input,
        signals re-referenced by extracting the mean over
        all signals for each channel
    """
    print(f'\n ECoG Rereferencing: Common Average')
    newdata = np.empty(data.shape)  # empty array with same shape
    newnames = [n[:8] for n in ch_names]

    if len(data.shape) == 3:
        newdata[:, 0, :] = data[:, 0, :]  # copy time rows
        for w in np.arange(data.shape[0]):  # loop over windows
            refdata = data[w, 1:, :].copy()
            for ch in np.arange(refdata.shape[1]):  # chann to reref
                # take window average of all other channels
                refchs = np.delete(refdata.copy(), ch, axis=0)
                ref_mean = np.nanmean(refchs, axis=0)
                newdata[w, ch + 1, :] = data[w, ch, :] - ref_mean
        
    elif len(data.shape) == 2:
        newdata[0, :] = data[0, :]  # copy time row
        refdata = data[1:, :].copy()
        for ch in np.arange(refdata.shape[0]):  # chann to reref
            # take window average of all other channels
            refchs = np.delete(refdata.copy(), ch, axis=0)
            ref_mean = np.nanmean(refchs, axis=0)
            newdata[ch + 1, :] = data[ch, :] - ref_mean


    return newdata, newnames


def reref_segm_levels(data: array, lead: Any, report_file: str):
    """
    Function to aggregate/sum all segmented electrode contacts
    which belong to same level. The function takes into
    account which contact are empty/bad/removed during arte-
    fact removal.
    
    Arguments:
        - data (array): 3d or 2d data array
        - lead (Class): class with info of segmented contacts
        per level
        - report_file  (str -> path): file incl directory where rereferencing
        notes are written to

    Returns:
        - reref_data (array): rereferenced signals
        - names (list): list with corresponding ch-names
    """

    ### TODO: write functionality with 2d-array input
    with open(report_file, 'a') as f:
        f.write(f'\nRereferencing: {lead.name} ({lead.side}) '
                 'against mean of neighbouring level')
    print(f'\n Rereferencing {lead.name} ({lead.side})'
          f' against means of neighbouring levels')

    if len(data.shape) == 3:
        leveldata = np.empty((
            data.shape[0],
            lead.n_levels + 1,
            data.shape[2]
        ))  # data array to store contact-means per level
        leveldata[:, 0, :] = data[:, 0, :]  # copy time row
        data=data[:, 1:, :]  # only contact-channels, drop time
    if len(data.shape) == 2:
        leveldata = np.empty((
            lead.n_levels + 1,
            data.shape[1]
        ))  # data array to store contact-means per level
        leveldata[0, :] = data[0, :]  # copy time row
        data=data[1:, :]  # only contact-channels, drop time

    ch_start = 0
    ch_stop = 0
    for l in np.arange(lead.n_levels):
        # loops over levels of lead
        ch_stop += len(lead.levels_str[l])
        # starts at 0, incl numb of clean contacts per level
        if len(data.shape) == 3:    
            leveldata[:, l + 1, :] = np.nanmean(
                data[:, ch_start:ch_stop, :], axis=1
            )  # row is l + 1 bcs of time in first row
        if len(data.shape) == 2:
            leveldata[l + 1, :] = np.nanmean(
                data[ch_start:ch_stop, :], axis=0
            )  # row is l + 1 bcs of time in first row
        # write in txt-file to check and document
        with open(report_file, 'a') as f:
            f.write(  # documenting incl contacts per level
                f'\nLevel {lead.side, l} '
                f'contains rows {ch_start}:{ch_stop}'
                f', or: {lead.levels_str[l]}'
            )
        ch_start = ch_stop  # updating start row

    rerefdata, names = reref_neighb_levels_diff(
        leveldata=leveldata, side=lead.side,
    )

    return rerefdata, names


def reref_neighb_levels_diff(
    leveldata: array, side: str):
    """
    Function to calculate differences between neighbouring
    eletrode levels. These can come from unsegmented
    electrodes, or from segmented electrodes via the
    reref_summing_levels() function.
    
    Arguments:
        - level (array): 3d array containing [windows,
        time- and level-rows, window_lenght]
        - side (str): L or R for recorded hemisphere

    Returns:
        - newdata (array): rereferenced signals. These
        will contain 1 row less because the differences
        between rows are used
        - sig_names (list): contains corresponnding names
        to the array: 'times', followed by the reref'd
        level-diffs, e.g. 0_1, 3_4
    """
    sig_names = ['time', ]
    if len(leveldata.shape) == 3:
        newdata = np.empty((
            leveldata.shape[0],
            leveldata.shape[1] - 1,
            leveldata.shape[2],
        ))
        newdata[:, 0, :] = leveldata[:, 0, :]  # copy times
        # subtract the level above one level from every level 
        newdata[:, 1:, :] = np.subtract(
            leveldata[:, 1:-1, :], leveldata[:, 2:, :]
        )
    if len(leveldata.shape) == 2:
        newdata = np.empty((
            leveldata.shape[0] - 1,
            leveldata.shape[1]
        ))
        newdata[0, :] = leveldata[0, :]  # copy times
        # subtract the level above one level from every level 
        newdata[1:, :] = np.subtract(
            leveldata[1:-1, :], leveldata[2:, :]
        )
    # adding signal names (Channel N minus N + 1)
    for level in np.arange(newdata.shape[-2] - 1):
        # shape[-2] gives row number for 2d and 3d arrays
        sig_names.append(f'LFP_{side}_{level}_{level + 1}')

    return newdata, sig_names


def reref_segm_contacts(
    data: array, lead: Any, report_file: str
):
    """
    Function for local spatial specific rereferencing
    of segmented-lead contacts.

    Arguments:
        - data (array): 3d data array
        - lead: class with info of segmented contacts
        per level
        - report_file (str -> path): file incl directory where rereferencing
        notes are written to

    Returns:
        - reref_data (array): rereferenced signals
        - names (list): list with corresponding ch-names
    """
    with open(report_file, 'a') as f:
        f.write(f'\nRereferencing: {lead.name} ({lead.side}) '
                ' against other contacts of same level')
    print(f'\n Rereferencing {lead.name} ({lead.side})'
          f' against other contacts of same level')
    # every contact gets it rerferenced output
    reref_data = np.empty_like((data))
    if len(data.shape) == 3:
        reref_data[:, 0, :] = data[:, 0, :]  # copy time row
    if len(data.shape) == 2:
        reref_data[0, :] = data[0, :]  # copy time row
    names = ['time', ]
    for l in lead.level_rows:  # loop over level
        chs = lead.level_rows[l]  # row numbers of channels
        if len(chs) > 1:  # if avail neighb channel on level
            for n, c in enumerate(chs):  # loop over contacts
                refs = chs.copy()  # channels to use as ref
                refs.remove(c)  # remove ch of interest itself
                # take mean of other channel-rows as reference
                print(f'Row REFS {refs}, SHAPE {data.shape}')
                if len(data.shape) == 3:
                    ref = np.nanmean(data[:, refs, :], axis=1)
                    reref_data[:, c, :] = data[:, c, :] -  ref
                if len(data.shape) == 2:
                    ref = np.nanmean(data[refs, :], axis=0)
                    reref_data[c, :] = data[c, :] -  ref
                names.append(lead.levels_str[l][n])
                # done, only test + put lfp_reref in filename!!
                with open(report_file, 'a') as f:
                    f.write(f'\n(level {l}) contact-row {c} is '
                            f'rereferenced against rows {refs}')

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
            names.append(lead.levels_str[l][0])
            with open(report_file, 'a') as f:
                f.write(f'\n(level {l}) Contact {lead.levels_str[l]}'
                        f' is rereferenced against {refs}')

        else:
            with open(report_file, 'a') as f:
                f.write(f'\n(level {l}) has no valid channels'
                        f' {lead.levels_str[l]}')

    return reref_data, names


def rereferencing(
    data: dict, group: str, runInfo: Any, chs_clean: list,
    lfp_reref: str=None,
):
    """
    Function to execute rereferencing of LFP and ECoG data.
    
    Arguments:
        - data: 2d or 3d-array, containing one group
        - group: signal group inserted (LFP-L/R, ECoG)
        - runInfo: class containing runInfolead_type:
        abbreviation for lead given in
        data class RunInfo (e.g. BSX, MTSS)
        - chs_clean = list with clean channels after
        artefact removal
        - lfp_reref: indicates string-name of reref-
        method: taking together one level, or rerefer-
        encing single segmented contacts
    
    Returns:
        - datareref: 3d array of rereferenced data of
        inserted signal group
        - names (list): strings of row names, 'times',
        all reref'd signal channels
    """
    report_file = os.path.join(runInfo.data_path,
                            'reref_report.txt')
    # existing report file is deleted in main-script
    with open(report_file, 'a') as f:
        f.write('\n\n======= REREFERENCING OVERVIEW ======\n')
 
    if group == 'ecog':
        with open(report_file, 'a') as f:
            f.write(
                f'For {group}: Common Average Reref\n'
            )
        # print(f'\nFor {group}: Common Average Reref')
        rerefdata, names = reref_common_average(
            data=data, ch_names=chs_clean
        )

    elif group[:3] == 'lfp':
        if 'time' in chs_clean: chs_clean.remove('time')
        side = chs_clean[0][4]  # takes 'L' or 'R'
        with open(report_file, 'a') as f:
            f.write(
                f'For {group}: Rereferencing '
                f'per separate {lfp_reref}.\n'
            )

        lead_type = runInfo.lead_type  # code for lead-type
        lead = Segm_Lead_Setup(lead_type, side, chs_clean)
        
        if lfp_reref == 'levels':
            # averaging each level, subtracting neighbour
            rerefdata, names = reref_segm_levels(
                data=data,
                lead=lead,
                report_file=report_file,
            )

        elif lfp_reref == 'segments':
            # using only own level mean to reref
            rerefdata, names = reref_segm_contacts(
                data=data,
                lead=lead,
                report_file=report_file,
            )

    # Quality check, delete only nan-channels
    ch_del = []
    for ch in np.arange(1, rerefdata.shape[-2]):
        # check whether ALL values in channel are NaN
        if len(rerefdata.shape) == 3:
            if not (np.isnan(rerefdata[:, ch, :]) == False).any():
                ch_del.append(ch)
        if len(rerefdata.shape) == 2:
            if not (np.isnan(rerefdata[ch, :]) == False).any():
                ch_del.append(ch)
    for ch in ch_del:
        rerefdata = np.delete(rerefdata, ch, axis=-2)
        with open(report_file, 'a') as f:
            f.write(
                f'\n\n Auto Cleaning:\n In {group}: row {ch} ('
                f'{names[ch]}) only contained NaNs and is deleted'
            ) 
        names.remove(names[ch])


    return rerefdata, names



