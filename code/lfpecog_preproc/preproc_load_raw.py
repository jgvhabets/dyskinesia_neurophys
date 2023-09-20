"""
Load in raw data without bids conversion
"""
import json
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Any
from itertools import compress
from datetime import timedelta

# Import own functions
from utils.utils_fileManagement import (
    get_project_path, get_onedrive_path
)
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side
)
import lfpecog_preproc.tmsi_poly5reader as poly5_reader
import lfpecog_preproc.preproc_plotting as plotting


@dataclass(init=True, repr=True)
class LoadRawData:
    """
    Function to process raw ephys data from TMSi-Poly5
     format, without being bids-converted

    Input:

    Raises:
        - ValueError if hand-side (left or right) is not
            defined in neither filename or channelnames
    """
    runInfo: Any  # Class of RunInfo
    subSettings: dict
    

    def __post_init__(self,):
        # Load raw Poly5
        raw = poly5_reader.Poly5Reader(self.runInfo.runDict['filepath'])
        self.sfreq = raw.sample_rate
        self.ch_names = raw.ch_names
        self.raw_data = raw.samples
        run_duration = (1 / self.sfreq) * self.raw_data.shape[1]
        self.run_times = np.arange(0, run_duration, 1 / self.sfreq)

        report = (f'poly5 loaded {self.runInfo.runDict["filepath"]}, '
                  f'fs: {self.sfreq}, chs: {self.ch_names}, shape: {self.raw_data.shape}\n'
                  '\nNo channel dropping because data was NOT BIDS converted.\n')

        self.ch_names = rename_raw_ch_names(self.ch_names, sub=self.runInfo.sub)

        ch_sel = [any([n in ch.lower() for n in ['acc', 'lfp', 'ecog']]) for ch in self.ch_names]
        self.ch_names = list(compress(self.ch_names, ch_sel))
        self.raw_data = self.raw_data[ch_sel]

        assert len(self.ch_names) == self.raw_data.shape[0], 'n-channels NOT MATCH data shape'
        report += (f'\nChannels renamed to {self.ch_names}\n')

        for group in self.runInfo.data_groups:

            if 'ecog' in group or 'lfp' in group or 'acc' in group:
                # select data and channels per data group
                sel = [group.upper()[:5] in ch for ch in self.ch_names]  # :5 refers to ECOG_, LFP_L, LFP_R, ACC_L, ACC_R
                group_chs = list(compress(self.ch_names, sel))
                group_data = self.raw_data[sel]
                setattr(self, group, (group_chs, group_data, self.run_times))

        if self.runInfo.reportTxt_path:
            with open(self.runInfo.reportTxt_path, 'a') as f:

                f.write(report)
                f.close()



def get_raw_runs(sub: str):
    """
    Find available dysk-protocol runs for subject,
    not bids-converted
    """


    data_path = os.path.join(get_project_path('data'),
                             'raw_poly5', f'sub-{sub}')
    if not os.path.exists(data_path):
        print(f'sub-{sub}, TRY RAW POLY5 FOLDER ON HARD DISK')
        data_path = os.path.join(get_project_path('data', extern_HD=True),
                                 'raw_poly5', f'sub-{sub}')
    
    assert os.path.exists(data_path), (
        f'No folder ({data_path}) with raw data for subject-{sub}'
    )
    folders = os.listdir(data_path)
    folders = [f for f in folders if 'dopa' in f.lower()]
    
    sub_json_path = os.path.join(get_onedrive_path('data'),
                                 'preprocess_jsons',
                                 f'runInfo_{sub}.json')
    with open(sub_json_path) as f: sub_json = json.load(f, )  # list of runinfo-dicts
    
    sub_runs = {}

    for i, f in enumerate(folders):

        f = os.path.join(data_path, f)
        filename = [file for file in os.listdir(f) if file.endswith('.Poly5')][0]  # is always only one file in folder
        file_path = os.path.join(f, filename)
        
        # find run info
        t = filename.split('.')[-3][-15:]  # find time of run start
        rec_time = f'{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:]}'

        for t in ['rest', 'tap', 'free']:
            if t in filename.lower(): task = t
        if task in sub_json["tasks_exclude"]:
            print(f'\t...{filename} EXCLUDED FROM PREPROC (TASK: {task})')
            continue  # find task and skip if defined

        if 'dopapre' in filename.lower(): acq = 'DopaPre'
        else: acq = 'Dopa' + filename.lower().split('dopa')[-1][:2]
        if acq in sub_json["acq_exclude"]:
            print(f'\t...{filename} EXCLUDED FROM PREPROC (ACQ: {acq})')
            continue  # find acq (dopaXX) and skip if defined

        sub_runs[i] = {
            'sub': sub,
            'bids_sub': False,
            'ses': False,
            'task': task,
            'acq': acq,
            'run': 1,
            'acq_time': rec_time,
            'dopaIn_time': sub_json["dopa_intake_time"],
            'tasks_excl': sub_json["tasks_exclude"],
            'data_include': sub_json["data_include"],
            'lead_type': sub_json["lead_type"],
            'raw_path': False,
            'filepath': file_path
        }
    
    return sub_runs


def rename_raw_ch_names(ch_names, sub=None):

    acc_side = 'R'
    acc_count = 0

    for i, ch in enumerate(ch_names):
        # add laterality to ACC axes
        if ch in ['X', 'Y', 'Z']:
            if acc_count == 3: acc_side = 'L'  # first 3 acc are right
            ch_names[i] = f'ACC_{acc_side}_{ch}'
            acc_count += 1

        if 'LFP' in ch.upper() or 'STN' in ch.upper():
            ch_names[i] = f'LFP_{ch[3]}_{ch[4:6]}'
            
        if 'ECOG' in ch.upper() or 'ECX' in ch.upper():
            if sub[0] == '1': continue  # no ecog for stn-only subs
            ecog_side = get_ecog_side(sub=sub).upper()[0]
            ch_names[i] = f'ECOG_{ecog_side}_{ch[4:6]}'
        
        if ch_names[i].endswith('SMC_AT'):
            print(f'remove SMC_AT from {ch}')
            ch_names[i] = ch_names[i][:-7]
    
    return ch_names



def get_raw_data_and_channels(
    rawRun, runInfo, to_plot: bool=False,
    settingsVersion: str = 'vX',
):
    """
    Extract data and channels from non-bids data

    Create two dict's, resp data and names per
    data-group. Include time (sample time, passed
    since recording started), and dopa_time (time
    delta relative to intake of Madopar LT)

    Input:
        - rawRun (class from LoadRawData()):
            containing all info of
            recording-session
        - runInfo (class from runInfo()): containing
            info from inserted JSON-files about
            desired data to preprocess
        - Fs (int): sample frequency of inserted array
    
    Returns:
        - data_arrays (dict): 2d-array (n-channels, n-samples)
            per data group (lfp, ecog, acc, etc) with
            all times and timeseries of defined session.
            'run_time' as seconds since start-recording,
            'dopa_time' as seconds since/before intake
            of L-Dopa (Madopar LT)
        - ch_names (dict): list per data group with
            column names corresponding to data_arrays
    
    """
    Fs = rawRun.sfreq

    data_arrays, ch_names = {}, {}

    for g in runInfo.data_groups:
        # get data and time arrays via mne-functions (load/get_data())
        (ch_names[g], dat, times) = getattr(rawRun, g)
        
        # create relative timestamps vs dopa-intake moment (in secs)
        dopa_t = [
            (runInfo.dopa_time_delta + timedelta(seconds=(i / Fs))
             ).total_seconds() for i in range(len(times))
        ]
        times = np.array([times, dopa_t])
        # prevent one date too short or long
        if times.shape[-1] == dat.shape[-1] + 1:
            times = times[:, :-1]
        data_arrays[g] = np.vstack((times, dat))
    
        ch_names[g] = ['run_time', 'dopa_time'] + ch_names[g]
        
        assert data_arrays[g].shape[0] == len(ch_names[g]), print(
            '\n\nASSERTION ERROR in get_data_and_channels() --> '
            f'Nr of DATA-ARRAY VARIABLES and CHANNEL NAMES for {g}'
            f' are NOT EQUAL (# in array: {data_arrays[g].shape[0]}'
            f' and # in names: {len(ch_names[g])}\n'
        )

        if to_plot:

            plotting.plot_groupChannels(
                ch_names=ch_names[g], groupData=data_arrays[g],
                Fs=Fs, groupName=g, runInfo=runInfo,
                moment='raw', settingsVersion=settingsVersion,
            )

        
    return data_arrays, ch_names