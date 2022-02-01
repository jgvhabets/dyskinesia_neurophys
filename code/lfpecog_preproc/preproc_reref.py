# Import packages and functions
import numpy as np
from dataclasses import dataclass
from typing import Any
import os


"""
Functions to re-reference ECoG and LFP signals.
Some relevant literature regarding rereferencing:
Liu ea, J Neural Eng 2015:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5485665/;
Dubey, J Neurosc 2019:
https://www.jneurosci.org/content/39/22/4299

"""

def reref_common_average(data, ch_names):
    """
    Function to perform Common Average Rereferencing.
    Default in ECoG signal processing.
    DISCUSS: Is channel of interest itself included in mean,
    for now: Yes, included in mean.

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
    newdata = np.empty(data.shape)  # empty array with same shape
    newdata[:, 0, :] = data[:, 0, :]  # copy time rows
    for w in np.arange(data.shape[0]):  # loop over windows
        # take window average of all channels
        ref_mean = np.nanmean(data[w, 1:, :], axis=0)
        for ch in np.arange(1, data.shape[1]):  # loop channels to reref
            newdata[w, ch, :] = data[w, ch, :] - ref_mean
    newnames = [n[:8] for n in ch_names]

    return newdata, newnames

@ dataclass
class Segm_Lead_Setup:
    """
    Class which stores levels and contacts of different
    DBS-electrode types and brands.
    
    Required Arguments:
        - codename: to identify lead-type
        - side: L or R
        - chs_clean: list with channel names after artefact removal
    """
    codename: str
    side: str
    chs_clean: list
    name: str = None
    num_levels: int = None
    num_contacts: int = None
    levels_str: dict = None
    levels_num: dict = None
    
    def __post_init__(self):
        # consider putting lead_setup in predefined dict/classes
        # per lead_type, instead of semi-automated defining like now
        if self.codename == 'BSX':
            self.name = 'BS Vercise Cartesia X'
            self.num_levels = 6
            self.num_contacts = 16
            self.levels_num = {}
            for l in np.arange(self.num_levels - 1):
                self.levels_num[l] = [
                    l * 3, (l * 3) + 1, (l * 3) + 2]
            # all levels 3 contacts, last level ringcontact
            self.levels_num[5] = [15]

        elif self.codename == 'MTSS':
            self.name = 'MT SenseSight'
            self.num_levels = 4
            self.num_contacts = 8
            self.levels_num[0] = [0]
            for l in [1, 2]:
                self.levels_num[l] = [
                    (l * 3) - 2, (l * 3) - 1, (l * 3)]
            # last level is ring contact
            self.levels_num[3] = [7]

        # create lists per level of possible contact-names
        contacts = [f'LFP_{self.side}_{n}_'
            for n in np.arange(1, self.num_contacts + 1)]
        self.levels_str = {}
        for l in self.levels_num:
            nums = self.levels_num[l]
            self.levels_str[l] = [contacts[c] for c in nums]
        # exclude channels not in clean-channel list
        combi_str = '\t'.join(self.chs_clean)  # paste all strings to one
        for l in self.levels_str:  # loops over levels
            for n, c in enumerate(self.levels_str[l]):
                # if ch-str is not in clean-channel combi-str
                if c not in combi_str:
                    self.levels_str[l].remove(c)
                    # lead_setup.level_num[l].remove(levels_num[l][n])


def reref_segm_levels(ch_data, times, lead_setup, report_file):
    """
    Function to aggregate/sum all segmented electrode contacts
    which belong to same level. The function takes into
    account which contact are empty/bad/removed during arte-
    fact removal.
    
    Arguments:
        - data (array): original signal
        - time (array): corresponding time array
        - lead_setup: class with info of segmented contacts
        per level, etc.
        - report_file: file incl directory where rereferencing
        notes are written to.

    Returns:
        - data (array): rereferenced signals
        - side (str): L or R for hemisphere
    """
    with open(report_file, 'a') as f:
        f.write('\nRereferencing against whole neighbouring levels')
    leveldata = np.empty((ch_data.shape[0], lead_setup.num_levels + 1,
                    ch_data.shape[2]))
    leveldata[:, 0, :] = times
    ch_start = 0
    ch_stop = 0
    for l in np.arange(lead_setup.num_levels):
        ch_stop += len(lead_setup.levels_str[l])
        leveldata[:, l + 1, :] = np.nanmean(
            ch_data[:, ch_start:ch_stop, :], axis=1
        )
        # write in txt-file to check and document
        with open(report_file, 'a') as f:
            f.write(
                f'\nLevel {lead_setup.side, l} contains rows {ch_start}:{ch_stop}'
                f', or: {lead_setup.levels_str[l]}'
            )

        ch_start = ch_stop

    rerefdata, names = reref_neighb_levels_diff(
        leveldata=leveldata, side=lead_setup.side,
    )

    return rerefdata, names


def reref_neighb_levels_diff(leveldata, side):
    """
    Function to calculate differences between neighbouring
    eletrode levels. These can come from unsegmented
    electrodes, or from segmented electrodes via the
    reref_summing_levels() function.
    
    Arguments:
        - level (array): 3d array containing [windows,
        time- and level-rows, window_lenght]
        - side(str): L or R for hemisphere

    Returns:
        - newdata (array): rereferenced signals. These
        will contain 1 row less because the differences
        between rows are used
        - sig_names (list): contains corresponnding names
        to the array: 'times', followed by the reref'd
        level-diffs, e.g. 0_1, 3_4
    """
    newdata = np.empty((leveldata.shape[0],
        leveldata.shape[1] - 1, leveldata.shape[2]))
    newdata[:, 0, :] = leveldata[:, 0, :]  # copy times
    sig_names = ['time', ]
    # subtract the level above one level from every level 
    newdata[:, 1:, :] = np.subtract(leveldata[:, 1:-1, :],
                                    leveldata[:, 2:, :])
    # adding signal names
    for level in np.arange(newdata.shape[1] - 1):
        sig_names.append(f'LFP_{side}_{level}_{level + 1}')

    return newdata, sig_names


def reref_segm_contacts(data, ch_names):
    """
    Function for local spatial specific rereferencing
    of segmented-lead contacts.

    Arguments:
        -

    Returns:
        -  
    """
#### WORK ON!!!
    # elif lfp_reref == 'segments':
    #     print('Rereferencing against neigbouring contacts (same level)')
    #     names = ['time'] + chs_clean  # maybe delete last parts of names??
    #     newdata = np.empty((chdata.shape[0],
    #         chdata.shape[1] - 1, chdata.shape[2]))
    #     newdata[:, 0, :] = times
    #     # TODO: involve lead_setup.level_str and lead_setup.level_num
    #     # renumber the level-groups from 0 to end, so it corresponds
    #     # with the ch_data rows.
    #     # Then make means per level.
    #     # Then subtract the corresponding level-mean from every row.




def rereferencing(
    data: dict, group: str, runInfo: Any, ch_names_clean,
    lfp_reref: str=None,
):
    """
    Function to rereference LFP and ECoG data.
    
    Arguments:
        - data: dict containing 3d-arrays per signal
        group (LFP-L/R, ECoG)
        - group: signal group inserted
        - runInfo: class containing runInfolead_type: abbreviation for lead given in
        data class RunInfo (e.g. BSX, MTSS)
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
    # make sure to delete a possibly existing report file
    with open(report_file, 'a') as f:
        f.write('\n\n======= REREFERENCING OVERVIEW ======\n')
    # print('\n\n======= REREFERENCING OVERVIEW ======')
    data = data[group]  # take group nd-array
    chs_clean = ch_names_clean[group]  # take group list

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
        # get (segmented) lead type and corresponding setup
        lead_type = runInfo.lead_type
        lead_setup = Segm_Lead_Setup(lead_type, side, chs_clean)
        
        if lfp_reref == 'levels':
            # averaging each level, subtracting neighbour
            rerefdata, names = reref_segm_levels(
                ch_data=data[:, 1:, :],
                times=data[:, 0, :],
                lead_setup=lead_setup,
                report_file=report_file,
            )

        elif lfp_reref == 'segments':
            # using only own level mean to reref
            rerefdata, names = reref_segm_contacts(
                data=data, ch_names=chs_clean
            )

    # Quality check, delete only nan-channels
    ch_del = []
    for ch in np.arange(rerefdata.shape[1]):
        # check whether ALL values in channel are NaN
        if not (np.isnan(rerefdata[:, ch, :]) == False).any():
            ch_del.append(ch)
    for ch in ch_del:
        rerefdata = np.delete(rerefdata, ch, axis=1)
        with open(report_file, 'a') as f:
            f.write(
                f'\n\n Auto Cleaning:\n In {group}: row {ch} ('
                f'{names[ch]}) only contained NaNs and is deleted'
            ) 
        # print(f'\n Auto Cleaning:\n In {group}: row {ch} ('
        #     f'{names[ch]}) only contained NaNs and is deleted')
        names.remove(names[ch])
    
    assert (rerefdata.shape[1] == len(names),
        'Length of data rows and channel names are not equal!')


    return rerefdata, names



