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
import numpy as np
from pandas import read_csv
from dataclasses import dataclass
from collections import namedtuple
from typing import Any
import mne_bids
import csv
import json


def get_sub_runs(
    sub, proj_path
):
    """
    Extract which runs to preprocess
    """
    sub_file = join(
        proj_path,
        'data/preprocess_jsons/'
        f'runInfo_{sub}.json'
    )
    sub_runs = {}
        
    with open(join(proj_path, sub_file)) as f:
        try:
            sub_json = json.load(f, )  # list of runinfo-dicts
        except json.decoder.JSONDecodeError:
            print('\n json.decoder.JSONDecodeError ERROR')
            print(
                'If JSON-file looks correct, try writing '
                'a Python-Dict with json.dump, and retry'
                )

    scans_df = read_csv(
        join(
            sub_json['raw_path'],
            f'sub-{sub}',
            f'ses-{sub_json["ses"]}',
            f'sub-{sub}_ses-{sub_json["ses"]}_scans.tsv'
        ),
        sep='\t',
    )
    
    for i in range(scans_df.shape[0]):
        run_string = scans_df['filename'][i]
        splits = run_string.split(sep='_')

        if splits[2][5:] in sub_json["tasks_exclude"]: continue
        if splits[3][4:] in sub_json["acq_exclude"]: continue

        sub_runs[i] = {
            'sub': sub,
            'ses': sub_json["ses"],
            'task': splits[2][5:],
            'acq': splits[3][4:],
            'run': splits[4][4:],
            'acq_time': scans_df['acq_time'][i],
            'dopaIn_time': sub_json["dopa_intake_time"],
            'data_include': sub_json["data_include"],
            'raw_path': sub_json["raw_path"]
        }
    
    return sub_runs


@dataclass(init=True, repr=True, )
class RunInfo:
    '''Stores the details of one specific run to import'''
    mainSettings: dict
    runDict: dict  #  dict from sub_runs (dict)
    project_path: str

    def __post_init__(self,):  # is called after initialization
        self.bidspath = mne_bids.BIDSPath(
            subject=self.runDict["sub"],
            session=self.runDict["ses"],
            task=self.runDict["task"],
            acquisition=self.runDict["acq"],
            run=self.runDict["run"],
            suffix='ieeg',
            extension='.vhdr',
            datatype='ieeg',
            root=self.runDict["raw_path"],
        )
        self.data_groups = self.runDict["data_include"]

        self.store_str = (
            f'{self.runDict["sub"]}_{self.runDict["task"]}_'
            f'{self.runDict["acq"]}_{self.runDict["run"]}'
        )
        
        self.fig_path = join(
            self.project_path,
            'figures',
            f'preprocessed/sub-{self.runDict["sub"]}',
            self.mainSettings["settingsVersion"],
        )
        
        self.data_path = join(
            self.project_path,
            'data',
            f'preprocessed/sub-{self.runDict["sub"]}',
            self.mainSettings["settingsVersion"],
        )
        
        for folder in [self.data_path, self.fig_path]:

            if not exists(folder):
                
                makedirs(folder)
         
        lead_type_dict = {  # PM extract from xlsl
            '002': 'BS_VC',  # Boston Cartesia Vercise
            '004': 'BS_VC',
            '005': 'MT_SS',  # Medtronic SenSight
            '008': 'BS_VC_X',  # Boston Cartesia Vercise X
            '009': 'MT_SS',
            '011': 'MT_SS',
            '012': 'MT_SS',
            '013': 'MT_SS',
            '014': 'MT_SS',
        }
        assert self.runDict["sub"] in lead_type_dict.keys(), print(
            f'No Lead-type is defined for subject {self.runDict["sub"]}'
            ' Please add to preproc_data_manegement.py'
        )
        self.lead_type = lead_type_dict[self.runDict["sub"]]


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
                self.runInfo.bidspath, verbose='WARNING'
            )
            print('\n\n------------ BIDS DATA INFO ------------\n'
                  f'The raw-bids-object contains {len(self.bids.ch_names)} channels with '
                  f'{self.bids.n_times} datapoints and sample freq ',
                  str(self.bids.info['sfreq']),'Hz')
            print('Bad channels are:',self.bids.info['bads'],'\n')

            # select ECOG vs DBS channels (bad channels are dropped!)
            for ecog_g in self.runInfo.data_groups:
                if ecog_g[:4] == 'ecog':
                    setattr(
                        self,
                        ecog_g,
                        self.bids.copy().pick_types(
                            ecog=True, exclude='bads'
                        )
                    )
                    ecogBool = True
            
            self.lfp = self.bids.copy().pick_types(dbs=True, exclude='bads')
            lfp_chs_L = [c for c in self.lfp.ch_names if c[4]=='L' ]
            lfp_chs_R = [c for c in self.lfp.ch_names if c[4]=='R' ]
            
            if 'lfp_right' in self.runInfo.data_groups:
                self.lfp_right = self.lfp.copy().drop_channels(lfp_chs_L)

            if 'lfp_left' in self.runInfo.data_groups:
                self.lfp_left = self.lfp.copy().drop_channels(lfp_chs_R)

            try:
                self.acc = self.bids.copy().pick_types(misc=True, exclude='bads')
                acc_chs_L = [c for c in self.acc.ch_names if c[4]=='L' ]
                acc_chs_R = [c for c in self.acc.ch_names if c[4]=='R' ]

                if 'acc_left' in self.runInfo.data_groups:
                    self.acc_left = self.acc.copy().drop_channels(acc_chs_R)

                if 'acc_right' in self.runInfo.data_groups:
                    self.acc_right = self.acc.copy().drop_channels(acc_chs_L)

            except ValueError:
                print('\n### WARNING: No ACC-channels available ###\n')
            
            # select EMG and ECG signals
            if 'emg' in self.runInfo.data_groups:
                try:
                    self.emg = self.bids.copy().pick_types(emg=True, exclude='bads')
                except ValueError:
                    print('\n### WARNING: No EMG-channels available ###\n')

            if 'eeg' in self.runInfo.data_groups:
                try:
                    self.ecg = self.bids.copy().pick_types(ecg=True, exclude='bads')
                except ValueError:
                    print('\n### WARNING: No ECG-channels available ###\n')

            print('BIDS contains:\n:')
            if ecogBool: print(f'{len(getattr(self, ecog_g).ch_names)} ECOG channels,\n')
            if self.lfp: print(f'{len(self.lfp.ch_names)} DBS channels:'
                               f' ({len(self.lfp_left.ch_names)} left, '
                               f'{len(self.lfp_right.ch_names)} right),\n')
            if 'emg' in self.runInfo.data_groups:
                print(f'\n{len(self.emg.ch_names)} EMG channels,\n')
            if 'eeg' in self.runInfo.data_groups:
                if self.ecg: print(f'\n{len(self.ecg.ch_names)} ECG channel(s),\n')
            if self.acc: print(f'\n{len(self.acc.ch_names)} Accelerometry channels.\n\n')



def save_arrays(
    data: array, names: list, group: str,
    runInfo: Any, lfp_reref: str,
):
    '''
    Function to save preprocessed 3d-arrays as npy-files.

    Arguments:
        - data (array): 3d-arrays with preprocessed data
        - names (dict): containing channel-name lists
        corresponding to data arrays
        - group(str): group to save
        - runInfo (class): class containing info of spec-run

    Returns:
        - None
    '''
    # define filename, save data-array per group (lfp L/R vs ecog)
    f_name = (
        f'{runInfo.store_str}_{runInfo.preproc_sett}_'
        f'{group.upper()}_PREPROC_data.npy'
    )
    np.save(os.path.join(runInfo.data_path, f_name), data)
    # save list of channel-names as txt-file
    f_name = (
        f'{runInfo.store_str}_{runInfo.preproc_sett}_'
        f'{group.upper()}_PREPROC_rownames.csv'
    )
    with open(join(runInfo.data_path, f_name), 'w') as f:
            write = csv.writer(f)
            write.writerow(names)

    return

