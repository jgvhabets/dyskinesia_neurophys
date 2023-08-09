'''
Functions to pre-process neurophysiology data (LFP and ECOG)
in ReTune's Dyskinesia Project

Specifically contains functions and DataClasses to structure,
save, and read data Objects as input and output in the preprocessing. 
'''

# Import general packages and functions
from array import array
from os.path import join, exists
from os import makedirs
import datetime
import numpy as np
from pandas import read_csv
from dataclasses import dataclass
from typing import Any
import mne_bids
import csv
import json

from utils.utils_fileManagement import get_project_path, get_onedrive_path


def get_sub_runs(sub,):
    """
    Extract which runs to preprocess
    """
    onedrive_data_path = get_onedrive_path('data')
    sub_json_path = join(onedrive_data_path, 'preprocess_jsons',
                         f'runInfo_{sub}.json')
    sub_runs = {}
        
    with open(sub_json_path) as f:

        try:
            sub_json = json.load(f, )  # list of runinfo-dicts

        except json.decoder.JSONDecodeError:
            print(
                '\n\t json.decoder.JSONDecodeError ERROR'
                f' while reading {sub_json_path}'
            )
            print(
                'If JSON-file looks correct, try writing '
                'a Python-Dict with json.dump, and retry'
                )
    
    assert sub_json, print('\n JSON FILE COULDNOT BE LOADED')

    # adjust preamble of subject code to ECoG or STN-only
    if 'stn_only' in sub_json.keys():
        if sub_json['stn_only']:
            bids_sub = sub_json['bids_sub']
        else:
            bids_sub = f'EL{sub}'
    else:
        bids_sub = f'EL{sub}'

    scans_df = read_csv(join(get_onedrive_path('bids_rawdata'),
                             f'sub-{bids_sub}',
                             f'ses-{sub_json["ses"]}',
                             f'sub-{bids_sub}_ses-{sub_json["ses"]}_scans.tsv'),
                        sep='\t',)

    for i in range(scans_df.shape[0]):
        skip_file = False
        run_string = scans_df['filename'][i]
        splits = run_string.split(sep='_')

        for task_ex in sub_json["tasks_exclude"]:
            if task_ex.lower() in splits[2].lower():
                skip_file = True

        for acq_ex in sub_json["acq_exclude"]:
            if acq_ex.lower() in splits[3].lower():
                skip_file = True

        if skip_file: continue
        # if skip_file is not set True, then include file in sub_runs
        sub_runs[i] = {
            'sub': sub,
            'bids_sub': bids_sub,
            'ses': sub_json["ses"],
            'task': splits[2][5:],
            'acq': splits[3][4:],
            'run': splits[4][4:],
            'acq_time': scans_df["acq_time"][i],
            'dopaIn_time': sub_json["dopa_intake_time"],
            'tasks_excl': sub_json["tasks_exclude"],
            'data_include': sub_json["data_include"],
            'lead_type': sub_json["lead_type"],
            'raw_path': get_onedrive_path('bids_rawdata')
        }
    
    return sub_runs


@dataclass(init=True, repr=True, )
class RunInfo:
    '''Stores the details of one specific run to import'''
    mainSettings: dict
    runDict: dict  #  dict from sub_runs (dict)
    project_path: str
    BIDS_DATA: bool = True

    def __post_init__(self,):  # is called after initialization
        if self.BIDS_DATA:
            self.rawdata_path = get_onedrive_path('bids_rawdata')
            
            self.bidspath = mne_bids.BIDSPath(
                subject=f'{self.runDict["bids_sub"]}',
                session=self.runDict["ses"],
                task=self.runDict["task"],
                acquisition=self.runDict["acq"],
                run=self.runDict["run"],
                suffix='ieeg',
                extension='.vhdr',
                datatype='ieeg',
                root=self.rawdata_path,
            )

        self.data_groups = self.runDict["data_include"]
        
        acq_time = datetime.datetime.fromisoformat(
            self.runDict["acq_time"]
        )
        ldopa_intake_time = datetime.datetime.fromisoformat(
            self.runDict["dopaIn_time"]
        )
        self.dopa_time_delta = acq_time - ldopa_intake_time

        self.store_str = (f'{self.runDict["sub"]}_{self.runDict["task"]}'
                          f'_{self.runDict["acq"]}')
        
        self.fig_path = join(self.project_path, 'figures',
                             'preprocessing', f'sub-{self.runDict["sub"]}',
                             self.mainSettings["settingsVersion"],)
        
        self.data_path = join(self.project_path, 'data',
                              'preprocessed_data', f'sub-{self.runDict["sub"]}',
                              self.mainSettings["settingsVersion"],)
        
        for folder in [self.data_path, self.fig_path]:
            if not exists(folder): makedirs(folder)

        if self.mainSettings['report_file']:

            report_folder = join(self.project_path, 'data',
                                 f'preproc_reports', f'sub-{self.runDict["sub"]}')
            if not exists(report_folder): makedirs(report_folder)

            now = datetime.datetime.now()
            now = now.strftime("%Y%m%d_%H%M")
            report_fname =  (f'preprocReport_{self.runDict["sub"]}_'
                             f'{self.runDict["task"]}{self.runDict["acq"][-6:]}'
                             f'_{self.mainSettings["settingsVersion"]}'
                             f'_{now}.txt')
            self.reportTxt_path = join(
                report_folder, report_fname, 
            )
            with open(self.reportTxt_path, 'w') as f:

                print(f'\n\tWRITING TXT FOR {self.store_str}'
                      f'\n\t\ton path: {self.reportTxt_path}')
                f.write(
                    '##### PREPROCESSING REPORT #####\n\n'
                    f'\tSub-{self.runDict["sub"]} (bids-subject: {self.runDict["bids_sub"]})\n\t'
                    f'Task: {self.runDict["task"]}\n\t'
                    f'Acquisition: {self.runDict["acq"]}\n\t'
                    f'Settings-Version: {self.mainSettings["settingsVersion"]}'
                    f'\n\n\tTasks excl: {self.runDict["tasks_excl"]}'
                    f'\n\tData incl: {self.runDict["data_include"]}'
                    f'\n\n\tRecording time relative to L-DOPA (Madopar'
                    f' LT intake): {self.dopa_time_delta} (hh:mm:ss)'  
                )
                f.close()

        self.lead_type = self.runDict["lead_type"]


@dataclass(init=True, repr=True,)
class defineMneRunData:
      '''Collect data from BIDS and stores them per data type.
      Actial loading in of the data happens later.'''
      runInfo: Any  # Class of RunInfo
      subSettings: dict

      def __post_init__(self, ):
      # read raw bids files into mne RawBrainVision object
      # doc mne.io.raw: https://mne.tools/stable/generated/mne.io.Raw.html

            self.bids = mne_bids.read_raw_bids(
                self.runInfo.bidspath, verbose="Warning",
            )
            self.sfreq = self.bids.info["sfreq"]
            report = (
                '\n\n------------ BIDS DATA INFO ------------\n'
                f'The raw-bids-object contains {len(self.bids.ch_names)} channels with '
                f'{self.bids.n_times} datapoints and sample freq '
                f'{self.bids.info["sfreq"]} Hz\n'
                f'Bad channels are: {self.bids.info["bads"]}\n'
            )

            # select ECOG vs DBS channels (bad channels are dropped!)
            if any([
                'ecog' in g for g in self.runInfo.data_groups
            ]):

                for group in self.runInfo.data_groups:
                    
                    if group[:4] == 'ecog':
                        ecog_group = group
                        setattr(self,
                                ecog_group,
                                self.bids.copy().pick_types(ecog=True, exclude='bads')
                        )
                        ecogBool = True
            
            else:
                ecogBool = False
            
            self.lfp = self.bids.copy().pick_types(dbs=True, exclude='bads')
            lfp_chs_L = [c for c in self.lfp.ch_names if c[4]=='L' ]
            lfp_chs_R = [c for c in self.lfp.ch_names if c[4]=='R' ]
            
            if 'lfp_right' in self.runInfo.data_groups:
                self.lfp_right = self.lfp.copy().drop_channels(lfp_chs_L)

            if 'lfp_left' in self.runInfo.data_groups:
                self.lfp_left = self.lfp.copy().drop_channels(lfp_chs_R)

            try:
                self.acc = self.bids.copy().pick_types(misc=True, exclude='bads')

                try:
                    acc_chs_L = [c for c in self.acc.ch_names if c[4]=='L' ]
                    acc_chs_R = [c for c in self.acc.ch_names if c[4]=='R' ]
                    
                    if 'acc_left' in self.runInfo.data_groups:
                        self.acc_left = self.acc.copy().drop_channels(acc_chs_R)

                    if 'acc_right' in self.runInfo.data_groups:
                        self.acc_right = self.acc.copy().drop_channels(acc_chs_L)

                except IndexError:

                    if set(['X', 'Y', 'Z']).issubset(self.acc.ch_names):

                        acc_unkwn = [c for c in self.acc.ch_names if len(c) == 1]

                        if any([c for c in self.acc.ch_names if 'ACC_L' in c]):

                            acc_l = [c for c in self.acc.ch_names if len(c) > 1]
                            self.acc_left = self.acc.copy().drop_channels(acc_unkwn)
                            self.acc_right = self.acc.copy().drop_channels(acc_l)
                        
                        elif any([c for c in self.acc.ch_names if 'ACC_R' in c]):

                            acc_r = [c for c in self.acc.ch_names if len(c) > 1]
                            self.acc_left = self.acc.copy().drop_channels(acc_r)
                            self.acc_right = self.acc.copy().drop_channels(acc_unkwn)

            except ValueError:
                print('\n### WARNING: No ACC-channels available ###\n')
            
            # select EMG and ECG signals
            if 'emg' in self.runInfo.data_groups:
                try:
                    self.emg = self.bids.copy().pick_types(emg=True, exclude='bads')
                except ValueError:
                    print('\n### WARNING: No EMG-channels available ###\n')

            if 'ecg' in self.runInfo.data_groups:
                try:
                    self.ecg = self.bids.copy().pick_types(ecg=True, exclude='bads')
                except ValueError:
                    print('\n### WARNING: No ECG-channels available ###\n')

            report = report + '\nBIDS channels contains:\n\t'
            
            if ecogBool:
                report = report + f'{len(getattr(self, ecog_group).ch_names)} ECOG channels,'
            
            if self.lfp:
                report = report + (
                    f'\n\t{len(self.lfp.ch_names)} DBS channels:'
                )
                try:
                    report = report + (
                        f' {len(self.lfp_left.ch_names)} left, '
                    )
                except AttributeError:
                    pass
                try:
                    report = report + (
                        f'{len(self.lfp_right.ch_names)} right'
                    )
                except AttributeError:
                    pass
            
            if 'emg' in self.runInfo.data_groups:
                report = report + f'\n\t{len(self.emg.ch_names)} EMG channels,\n'
            
            if 'ecg' in self.runInfo.data_groups:
                if self.ecg:
                    report = report + f'\n\t{len(self.ecg.ch_names)} ECG channel(s),\n'
            
            if self.acc:
                report = report + f'\n\t{len(self.acc.ch_names)} Accelerometry channels.\n\n'
            
            print(report)
            if self.runInfo.reportTxt_path:
                with open(self.runInfo.reportTxt_path, 'a') as f:

                    f.write(report)
                    f.close()



def save_dict(
    dataDict, namesDict, FsDict, runInfo,
):
    """
    Save preprocssed dictionaries containing
    data arrays, per array
    """
    for group in dataDict.keys():

        save_array(
            data=dataDict[group],
            names=namesDict[group],
            Fs=FsDict[group],
            group=group,
            runInfo=runInfo,
        )

def save_array(
    data: array, names: list, group: str,
    Fs, runInfo: Any,
):
    '''
    Function to save preprocessed np-arrays as npy-files.

    Arguments:
        - data (array): 3d-arrays with preprocessed data
        - names (list): containing channel-name lists
            corresponding to data arrays
        - group (str): group name
        - runInfo (class): class containing info of spec-run

    Returns:
        - None
    '''
    # save data array as .npy
    f_name = (f'data_{runInfo.store_str}_'
              f'{runInfo.mainSettings["settingsVersion"]}'
              f'_{group}_{Fs}Hz.npy')
    np.save(join(runInfo.data_path, f_name), data)

    print(f'Data (npy) saved\n\t{f_name} @ {runInfo.data_path})')

    # save list of channel-names as .csv
    f_name = (f'names_{runInfo.store_str}_'
              f'{runInfo.mainSettings["settingsVersion"]}'
              f'_{group}.csv')
    
    with open(join(runInfo.data_path, f_name), 'w') as f:
        write = csv.writer(f)
        write.writerow(names)
        f.close()

    return

