'''
Functions to read in and organize pre-processed neurophysiology
data (LFP and ECOG) in ReTune's Dyskinesia Project
'''

# Import general packages and functions
from array import array
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
import csv

# Import own functions
from tapping_feat_calc import nan_array


@dataclass(init=True, repr=True, )
class subjectData:
    """
    Creates Class with all data of one subject,
    stored per datatype. Data is preprocessed and
    ordered by relative time to L-Dopa-intake.

    Input:
        - sub (str): subject code
        - data_version (str): e.g. v2.2
        - project_path (str): main project-directory
            where data is stored 
    """
    sub: str
    data_version: str
    project_path: str
    dType_excl: list = field(default_factory=lambda: [])

    def __post_init__(self,):
        self.dtypes, self.nameFiles, self.dataFiles, sub_path = find_proc_data(
            sub=self.sub,
            version=self.data_version,
            project_path=self.project_path
        )

        if len(self.dType_excl) > 0:
            dType_remove = []
            for dType in self.dtypes:
                if dType in self.dType_excl:
                    dType_remove.append(dType)
            [self.dtypes.remove(d) for d in dType_remove]
        
        for dType in self.dtypes:

            setattr(
                self,
                dType,
                dopaTimedDf(
                    dType,
                    self.nameFiles,
                    self.dataFiles,
                    sub_path
                )
            )


@dataclass(init=True, repr=True, )
class dopaTimedDf:
    """
    Create Class including preproceseed data,
    Fs, behavorial tasks, and clinical scores
    of one data type, of one subject.
    pd.DataFrame including all data is in
    resampled frequency and chronologically
    ordered by time relative to L-Dopa intake

    In case a dataype should be skipped this
    should be defined in dType_excl.

    Input:
    dType (str): specific datatype to handle (acc, lfp, ecog)
    nameFiles (list): all name-files corresponding to subj
    dataFiles (list): all data-files corresponding to subj
    sub_path (str): path where data/name files are stored
    """
    dType: str
    nameFiles: list
    dataFiles: list
    sub_path: str

    def __post_init__(self,):
        df, fs = create_dopa_timed_array(
            self.dType,
            self.nameFiles,
            self.dataFiles,
            self.sub_path
        )
        self.data = df
        self.fs = int(fs)


def find_proc_data(
    sub: str,
    version: str,
    project_path: str,
):
    '''
    Find available runs, datatypes, and all filenames
    for one subject, considers specific data-version
    used for preprocessing settings
    '''
    sub_proc_path = os.path.join(
        project_path, 
        'data/preprocessed_data',
        f'sub-{sub}',
        version,
    )
    # get available files for sub + version
    files = os.listdir(sub_proc_path)
    # select npy files
    datafiles = [f for f in files if f[-3:] == 'npy']
    namefiles = [f for f in files if f[-3:] == 'csv']

    dtypes = []
    for f in namefiles:
        
        dType = f'{f.split("_")[-2]}_{f.split("_")[-1][:-4]}'
        if dType not in dtypes: dtypes.append(dType)
    
    return dtypes, namefiles, datafiles, sub_proc_path


def create_dopa_timed_array(
    dType: str,
    allNames: list,
    allData: list,
    sub_proc_path: str
):
    """
    Takes output of find_proc_data() and creates pd.DF
    with all data from one datatype from one subject,
    ordered by time relative to L-Dopa intake.
    Extract corresponding Fs, includes behavioral task
    label per data row.

    dType (str): specific datatype to handle (acc, lfp, ecog)
    allNames (list): all name-files corresponding to subj
    allData (list): all data-files corresponding to subj
    sub_proc_path (str): path where data/name files are stored
    """
    nameFiles = [f for f in allNames if dType in f]
    dataFiles = [f for f in allData if dType in f]

    for nFile, datFile in enumerate(dataFiles):

        rec = datFile.split('_')[3]
        nameFile = [f for f in nameFiles if rec in f][0]
        names = list(pd.read_csv(os.path.join(
            sub_proc_path, nameFile
        )))[1:]  # exclude run_time

        task = datFile.split('_')[2]
        if 'rest' in task.lower(): task = 'rest'
        if 'tap' in task.lower(): task = 'tap'
        if 'free' in task.lower(): task = 'free'

        fs = datFile.split('Hz')[0]
        fs = fs.split('_')[-1]

        if nFile == 0:

            data_out = pd.DataFrame(
                data=np.load(os.path.join(
                    sub_proc_path, datFile
                ))[1:, :].T,
                columns=names
            )
            data_out['task'] = [task] * data_out.shape[0]

        
        else:

            rec_data = pd.DataFrame(
                data=np.load(os.path.join(
                    sub_proc_path, datFile
                ))[1:, :].T,
                columns=names,
                index=None,
            )
            rec_data['task'] = [task] * rec_data.shape[0]

            data_out = pd.concat(
                [
                    data_out,
                    rec_data
                ])  # vertical adding of empty rows


    data_out = data_out.sort_values(
        axis=0, by='dopa_time'
    ).reset_index(drop=True)

    return data_out, fs




# def select_runs(
#     sub: str,
#     version: str,
#     project_path: str,
#     sess_incl: list=[],
#     tasks_incl: list=[],
#     acqs_incl: list=[],
#     groups_incl: list=[],
# ):
#     '''
#     Select run-files to use for eploration, or
#     feature extraction.
#     Write to DataClass
#     '''
#     # get available files for sub + version
#     fdir = os.path.join(
#         project_path, 
#         'data/preprocess',
#         f'sub-{sub}',
#         version,
#     )
#     files = os.listdir(fdir)
#     # filter files on npy, session, task, acq
#     files = [f for f in files if f[-3:] == 'npy']
#     if sess_incl:  # if session(s) given
#         files = [f for f in files
#             if f.split('_')[1] in sess_incl]
#     if tasks_incl:  # if task(s) given
#         files = [f for f in files
#             if f.split('_')[2] in tasks_incl]
#     if acqs_incl:  # if acq(s) given
#         files = [f for f in files
#             if f.split('_')[3] in acqs_incl]
#     f_sel = []  # filter on group
#     for f in files:
#         g = f.split('_')[7:-2]
#         if len(g) == 2: g = f'{g[0]}_{g[1]}'
#         if len(g) == 1: g = g[0]
#         if g in groups_incl: f_sel.append(f)

#     if len(f_sel) == 0:
#         print('\nWARNING: No files are included for '
#               'data exploration or feature extraction!')
#         return

#     return f_sel, groups_incl, fdir


def read_ieeg_file(npy_f, fdir):
    '''
    
    '''
    csv_f = f'{npy_f[:-8]}rownames.csv'

    data = np.load(os.path.join(fdir, npy_f))
    names = []
    try:
        with open(os.path.join(fdir, csv_f),
                newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                names = row
    except:
        print('fill except...')

    # Data vs names check
    s1 = data.shape[-2]
    s2 = len(names)
    assert s1 == s2, ('# rows of data does not match the # of'
                        f' names for {npy_f}')

    return data, names


def read_files(fsel, groups, fdir):
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
    data = {}  # dict to store processed data arrays
    names = {}  # dict to store corresponding names
    for g in groups:
        # for matching files with defined group
        g_files = [f for f in fsel if g in f]
        print(g_files)
        for f in g_files:
            csv_f = f'{f[:-8]}rownames.csv'
            # TODO: CURRENTLY FILES ARE OVERWRITING EACH OTHER
            # MAKE DATA CLASS HERE, WITH STRUCTURE
            # CLASS PER PT (SESSION?)
            #   ACQ/TASK
            #       GROUP
            #           - DATA
            #           - NAMES
            data[g] = np.load(os.path.join(fdir, f))
            print('\n', f, 'loaded,\nshape: ', data[g].shape)
            names[g] = []
            try:
                with open(os.path.join(fdir, csv_f),
                        newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    for row in reader:
                        names[g] = row
            except:
                print('fill except...')
        
        # Data vs names check
        s1 = data[g].shape[-2]
        s2 = len(names[g])
        assert s1 == s2, ('# rows not equal for data (npy)'
                         f'and names (csv) in {g}')

    return data, names



@dataclass(init=True, repr=True, )
class RunData:
    '''
    Creates dataclass containing all data from one specific
    run.
    Function is called within SessionData, and will be set
    to a dict in SessionData.
    Every RunData consists the requested and available data-
    types. Every datatype has its own RunData-Class containing
    the data-array (2d or 3d), rownames, current sample freq.
    '''
    npy_files: list
    fdir: str
    sub: str
    ses: str
    task: str
    acq: str
    run: int
    proc_version: str
    run_string: str
    # lfp  -> consider to make extra DataClass type per
    # data source containing array/names/Fs/... ?
    lfp_left_arr: array = np.array(0)  # std empty fields
    lfp_left_names: list = field(default_factory=list)
    lfp_left_Fs: int = field(default_factory=int)
    lfp_right_arr: array = np.array(0)
    lfp_right_names: list = field(default_factory=list)
    lfp_right_Fs: int = field(default_factory=int)
    # ecog
    ecog_arr: array = np.array(0)
    ecog_names: list = field(default_factory=list)
    ecog_Fs: int = field(default_factory=int)
    # acc
    acc_left_arr: array = np.array(0)
    acc_left_names: list = field(default_factory=list)
    acc_left_Fs: int = field(default_factory=int)
    acc_right_arr: array = np.array(0)
    acc_right_names: list = field(default_factory=list)
    acc_right_Fs: int = field(default_factory=int)
    present_datatypes: list = field(default_factory=list)
 
    def __post_init__(self, ):
        # fill present and available ieeg datatypes of given run
        ### TODO: check whether dataclass variables can be initialized
        ## right away within loop with setattr()
        for dtype in [
            'LFP_LEFT', 'LFP_RIGHT', 'ECOG', 'ACC_LEFT', 'ACC_RIGHT'
        ]:
            if (f'{self.run_string}_{dtype}_PREPROC_data.npy' not in
                    self.npy_files):  # check if dtype exist in files
                continue

            arr, names = read_ieeg_file(
                f'{self.run_string}_{dtype}_PREPROC_data.npy', self.fdir
            )  # set datatype array, names, current sample freq
            setattr(self, f'{dtype.lower()}_arr', arr)  # set to fields
            setattr(self, f'{dtype.lower()}_names', names)
            if len(arr.shape) == 3: timediff = arr[0, 0, 1] - arr[0, 0, 0]
            if len(arr.shape) == 2: timediff = arr[0, 1] - arr[0, 0]
            Fs = int(1 / timediff)
            setattr(self, f'{dtype.lower()}_Fs', Fs)
            # add dtype to list
            self.present_datatypes.append(dtype)



@dataclass(init=True, repr=True, )
class SessionData:
    '''Stores different RunData classes from the same Session'''
    npy_files: list
    fdir: str
    runs: dict = field(default_factory=dict)
    runs_incl: list = field(default_factory=list)
    ses_name: str = field(default_factory=str)

    def __post_init__(self,):  # is called after initialization
        for n, f in enumerate(self.npy_files):
            split_f = f.split('_')
            sub = split_f[0]
            ses = split_f[1]
            task = split_f[2]
            acq = split_f[3]
            run = split_f[4]
            proc_version = f'{split_f[5]}_{split_f[6]}'
            run_shortname = f'{task}_{acq}'
            run_str = f'{sub}_{ses}_{task}_{acq}_{run}_{proc_version}'

            if n == 0:
                self.ses_name = f'{sub}_{ses}'

            if run_shortname in self.runs_incl:
                continue  # skip run if already incl

            # store all runs in dictionary, since the runs
            # are not known prior to creating this class.
            # Therefore runs as (data)class is not possible (?)
            ### TODO: rewrite classes to create one array per
            # session, order based on dopa-time (unique values)
            # include different array-attributes for ephys, acc,
            # and score data.
            # each has it's own dopa-time array (with corresponding
            # sample freq (eg 1600/800/1)
            # Implement each data type (source) as its own class containing
            # time and data attributes (list/array)

            self.runs[run_shortname] = RunData(
                npy_files=self.npy_files,
                fdir=self.fdir,
                sub=sub,
                ses=ses,
                task=task,
                acq=acq,
                run=run,
                proc_version=proc_version,
                run_string=run_str,
            )
            self.runs_incl.append(run_shortname)


# def import_tap_annotations(
#     data_path, sub/run-info
# ):
#     """
#     Import tap-annotations from manually created xlsx-
#     files
#     """


#     return



# def import_dysk_scores(
#     data_path, sub/run-info
# ):
#     """
#     Import clinical rated dyskinesia scores
#     from xlsx-files
#     """
