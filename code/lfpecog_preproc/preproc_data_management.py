'''
Functions to pre-process neurophysiology data (LFP and ECOG)
in ReTune's Dyskinesia Project

Specifically contains functions and DataClasses to structure,
save, and read data Objects as input and output in the preprocessing. 
'''

# Import general packages and functions
from array import array
import os
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from typing import Any
import mne_bids
import csv



'''
Define namedtuple to store preprocessing settings per group
'''
PreprocSettings = namedtuple('PreprocSettings', (
    'win_len '
    'artfct_sd_tresh '
    'bandpass_f '
    'transBW '
    'notchW '
    'Fs_orig '
    'Fs_resample '
    'settings_version '
))
'''
Define namedtuple to store the preprocessing settings of all
three groups
'''
Settings = namedtuple('Settings', 'lfp_left lfp_right ecog')



# creates Class-init, repr gives printing-info, frozen makes 'immutable'
# dot not use frozen here, bcs of inheritance to non-frozen RunRawData Class
@dataclass(init=True, repr=True, )
class RunInfo:
    '''Stores the details of one specific run to import'''
    sub: str  # patient id, e.g. '008'
    ses: str  # sessions id, e.g. 'EphysMedOn02'
    task: str  # task id, e.g. 'Rest'
    acq: str  # acquisition: Stim and Dysk-Meds (StimOffLevo30)
    run: str  # run sequence, e.g. '01'
    raw_path: str  # directory where raw data (.Poly5) is stored
    project_path: str  # directory with data and figures-folders
    preproc_sett: str  # name settings-version (also for folder)
    # parent-folder of code/data/figures to store created files
    bidspath: Any = None  # made after initiazing
    store_str: Any = None  # made after initiazing
    fig_path: str = None  # folder to store figures
    data_path: str = None  # folder to store preprocessed data
    lead_type: str = None  # pre-defined lead-type of subject

    def __post_init__(self,):  # is called after initialization
        bidspath = mne_bids.BIDSPath(
            subject=self.sub,
            session=self.ses,
            task=self.task,
            acquisition=self.acq,
            run=self.run,
            suffix='ieeg',
            extension='.vhdr',
            datatype='ieeg',
            root=self.raw_path,
        )
        store_str = (f'{self.sub}_{self.ses}_{self.task}_'
                     f'{self.acq}_{self.run}')
        # bidspath for MNE functions
        self.bidspath = bidspath
        # folder name to store figures and derivative
        self.store_str = store_str
        
        # define folders to store figures and processed data
        self.fig_path = os.path.join(self.project_path, 'figures')
        self.data_path = os.path.join(self.project_path, 'data')
        # check (+ create) sub figure- and data-folders
        for folder in [self.data_path, self.fig_path]:
            if not os.path.exists(os.path.join(
                folder, f'preprocess/sub-{self.sub}',
            )):
                os.mkdir(os.path.join(
                    folder, f'preprocess/sub-{self.sub}',
                ))
        ''' Make sure there are sub-XXX folders in
        - project-folder/data/preprocess,
        - project-folder/figures/preprocess,
        - project-folder/figures/exploration,
        '''
        # check ses and version specific preproc folders
        for folder in [self.data_path, self.fig_path]:
            version_folder = os.path.join(
                folder, f'preprocess/sub-{self.sub}',
                self.preproc_sett,
            )
            if not os.path.exists(version_folder):
                os.mkdir(version_folder)
            # # check (+ create) ft-version folders
            # if not os.path.exists(os.path.join(
            #     ses_folder, self.preproc_sett,
            # )):
            #     os.mkdir(os.path.join(
            #         ses_folder, self.preproc_sett,
            #     ))
        # check ses + version exploration-fig folders
        ses_fig_expl = os.path.join(
            self.fig_path, f'exploration/sub-{self.sub}',
            self.store_str,
        )
        # print(ses_fig_expl)
        if not os.path.exists(ses_fig_expl):
            os.mkdir(ses_fig_expl)
        if not os.path.exists(os.path.join(
            ses_fig_expl, self.preproc_sett, 
        )):
            os.mkdir(os.path.join(
                ses_fig_expl, self.preproc_sett, 
            ))
        # Finally overwrites data_path and fig_path with
        # specific path's incl run and ft-version
        self.data_path = os.path.join(
                self.data_path,
                f'preprocess/sub-{self.sub}',
                self.preproc_sett,
            )
        self.fig_path = os.path.join(
            self.fig_path,
            f'preprocess/sub-{self.sub}',
            self.preproc_sett,
        )  
        
        lead__type_dict = {
            '008': 'BSX',  # Boston Cartesia Vercice X
            '009': 'MTSS',  # Medtronic SenSight
        }
        self.lead_type = lead__type_dict[self.sub]



@dataclass(init=True, repr=True,)
class RunRawData:
      '''Collect data from BIDS and stores them per data type.
      Actial loading in of the data happens later.'''
      bidspath: Any  # takes RunIno.bidspath as input kw
      bids: Any = None
      lfp: Any = None
      lfp_left: Any = None
      lfp_right: Any = None
      ecog: Any = None
      acc: Any = None
      emg: Any = None
      ecg: Any = None


      def __post_init__(self, ):
      # read raw bids files into mne RawBrainVision object
      # doc mne.io.raw: https://mne.tools/stable/generated/mne.io.Raw.html
            self.bids = mne_bids.read_raw_bids(self.bidspath, verbose='WARNING')
            # print bids-info to check
            print('\n\n------------ BIDS DATA INFO ------------\n'
                  f'The raw-bids-object contains {len(self.bids.ch_names)} channels with '
                  f'{self.bids.n_times} datapoints and sample freq ',
                  str(self.bids.info['sfreq']),'Hz')
            print('Bad channels are:',self.bids.info['bads'],'\n')

            # select ECOG vs DBS channels (bad channels are dropped!)
            self.ecog = self.bids.copy().pick_types(ecog=True, exclude='bads')
            self.lfp = self.bids.copy().pick_types(dbs=True, exclude='bads')
            # accelerometer channels are coded as misc(ellaneous)
            self.acc = self.bids.copy().pick_types(misc=True, exclude='bads')
            # splitting LFP in left right
            lfp_chs_L = [c for c in self.lfp.ch_names if c[4]=='L' ]
            lfp_chs_R = [c for c in self.lfp.ch_names if c[4]=='R' ]
            self.lfp_left = self.lfp.copy().drop_channels(lfp_chs_R)
            self.lfp_right = self.lfp.copy().drop_channels(lfp_chs_L)
            # select EMG and ECG signals
            self.emg = self.bids.copy().pick_types(emg=True, exclude='bads')
            self.ecg = self.bids.copy().pick_types(ecg=True, exclude='bads')

            print(f'BIDS contains:\n{len(self.ecog.ch_names)} ECOG '
                  f'channels,\n{len(self.lfp.ch_names)} DBS channels:'
                  f' ({len(self.lfp_left.ch_names)} left, '
                  f'{len(self.lfp_right.ch_names)} right), '
                  f'\n{len(self.emg.ch_names)} EMG channels, '
                  f'\n{len(self.ecg.ch_names)} ECG channel(s), '
                  f'\n{len(self.acc.ch_names)} Accelerometry (misc) channels.\n\n')



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
    rrf = lfp_reref.capitalize()
    # define filename, save data-array per group (lfp L/R vs ecog)
    f_name = (
        f'preproc_data_{group}_{runInfo.store_str}_'
        f'{runInfo.preproc_sett}_reref{rrf}.npy'
    )
    np.save(os.path.join(runInfo.data_path, f_name), data)
    # save list of channel-names as txt-file
    f_name = (
        f'preproc_chnames_{group}_{runInfo.store_str}_'
        f'{runInfo.preproc_sett}_reref{rrf}.csv')
    with open(os.path.join(runInfo.data_path, f_name), 'w') as f:
            write = csv.writer(f)
            write.writerow(names)

    return



def read_preprocessed_data(runInfo):
    '''
    Function which reads the saved preprocessed 3d-ararys and lists
    with channel-names again into Python Objects.

    Arguments:
        - runInfo: class contains the location where the files of the defined
        ft-version are saved

    Returns:
        - data (dict): dict with a 3d array of processed data, per group
        (e.g. lfp-l, lfp-r, ecog)
        - names (dict): dict containing the corresponding row-names
        belonging to the data groups (incl 'time', 'LFP_L_01', etc)
    '''
    files = os.listdir(runInfo.data_path)  # get filenames in folder
    sel_npy = [f[-3:] == 'npy' for f in files]  # make Boolean for .npy
    sel_csv = [f[-3:] == 'csv' for f in files]  # make Boolean for .csv
    # only includes files for which npy/csv-Boolean is True
    f_npy = [f for (f, s) in zip(files, sel_npy) if s]
    f_csv = [f for (f, s) in zip(files, sel_csv) if s]

    data = {}  # dict to store processed data arrays
    names = {}  # dict to store corresponding names
    groups = ['ecog', 'lfp_left', 'lfp_right']
    for g in groups:
        for fdat in f_npy:
            for fname in f_csv:
                if np.logical_and(g in fdat, g in fname):
                    # for matching files with defined group
                    data[g] = np.load(os.path.join(
                        runInfo.data_path, fdat))
                    names[g] = []
                    with open(os.path.join(runInfo.data_path, fname),
                            newline='') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',')
                        for row in reader:
                            names[g] = row
        s1 = data[g].shape[-2]
        s2 = len(names[g])
        assert s1 == s2, ('# rows not equal for data (npy)'
                         f'and names (csv) in {g}')

    return data, names


