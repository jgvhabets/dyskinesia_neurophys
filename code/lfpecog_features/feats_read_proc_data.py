'''
Functions to read in and organize pre-processed neurophysiology
data (LFP and ECOG) in ReTune's Dyskinesia Project
'''

# Import general packages and functions
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import csv
from array import array

# import own functions
import utils.utils_fileManagement as fileMng



@dataclass(init=True, repr=True, )
class main_loadMergedData:
    """
    Main class to run to load merged dataframes per subject,
    takes merged dataframes from already preprocessed
    and stored data

    TODO: include filtering on ACC-state

    Input:
        - list_of_subs: list of subject to include,
            every sub coded as '001', '002', etc.
        - tasks: list of tasks for which a merged-
            df is created, can be ['all', 'rest']
        - data_version: of preprocesing e.g. 'v0.1'
        - float_convert: convert py-floats to np.float64
        - data_as_df: subXXX attribute is pandas DF,
            if false: subXXX attr is class with
            separate data_arr (array), fs (int), col_names (list),
            and time_index (array)
    
    Returns:
        - class containing merged dataframes, sorted
            by task, then by sub (e.g.: className.rest.subXXX)
    
    Raises:
        - ValueError if an incorrect list of tasks
            is given
    """
    list_of_subs: list
    data_version: str
    float_convert: bool = True
    tasks: list = field(default_factory=lambda: ['all', 'rest'])
    data_as_df: bool = True
    # filter_on_ACC: str/list

    def __post_init__(self,):

        # create a seperate class per task-selection 
        for task in self.tasks:

            if task not in ['all', 'rest']:
                raise ValueError(
                    f'task ("{task}") not "all", or "rest"'
                    f', TODO: change input tasks ({self.tasks})')

            print(f'start loading {task} Data')
            setattr(
                self,
                task,
                get_mergedData_perTask(
                    list_of_subs=self.list_of_subs,
                    task=task,
                    data_version=self.data_version,
                    float_convert=self.float_convert,
                    return_as_df=self.data_as_df,
                )
            )        


@dataclass(repr=True, init=True,)
class get_mergedData_perTask:
    """
    Class to store merged dataframes of
    different subject, one class of this
    is created per 'task', can be 'all'-tasks
    (e.g. no filtering), or only data from
    'rest'-recordings

    Input:
        - list_of_subs: list of sub-code to
            include
        - task: task code to include, must be
            in ['all', 'rest',]
        - data_version: of preprocesing e.g. 'v0.1'
        - float_convert: convert py-floats to np.float64
        - return_as_df: return as pandas DF, or return
            separate dat_arr (np), col_names (list),
            and time_arr (floats)
        - acc_filter: defaults to None, if
            defined: data is also filtered
            on a specifc detected-ACC-state
        TODO: consider to move ACC-filtering
            to segmenting-function, in order
            to only exclude segments instead
            of windows based on ACC-states

    """
    list_of_subs: list
    task: str
    data_version: str
    float_convert: bool = True
    return_as_df: bool = True
    acc_filter: str = None
    

    def __post_init__(self,):

        for sub in self.list_of_subs:

            print(
                f'\tloading Sub {sub} ({self.task})'
            )
            # LOAD DATA
            df = load_stored_merged_data(
                sub=sub,
                data_version=self.data_version,
                float_convert=self.float_convert,
                return_as_df=self.return_as_df,
                )
            if not self.return_as_df:  # unpack tuple if necessary
                dat_arr, fs, col_names, time_arr = df
            
            # SELECT ON TASK
            if self.task == 'rest':  # select only rest-data
                
                if self.return_as_df:
                    sel = df['task'] == 'rest'
                    df = df[sel]
                # for non-df return this happens in class-function

            # Add ACC-selection here
            # if self.acc_filter in []:
                # df = filter_df_on_ACC(df, self.acc_filter)
            
            # SET AS ATTRIBUTE (dataframe or class)
            if self.return_as_df:
                setattr(
                    self,
                    f'sub{sub}',  # name is subXXX
                    df
                )
            
            else:
                setattr(
                    self,
                    f'sub{sub}',
                    subData_asArrays(
                        data_arr=dat_arr,
                        fs=fs,
                        col_names=col_names,
                        time_index=time_arr,
                        task_sel=self.task,
                    )
                )
                
            print(f'{sub} FINISHED')


@dataclass(init=True, repr=True, )
class subData_asArrays:
    data_arr: array
    fs: int
    col_names: list
    time_index: array
    task_sel: str = field(default_factory='')

    def __post_init__(self,):

        if len(self.task_sel) > 1:

            i_task = np.where(self.col_names == 'task')[0][0]
            sel = self.data_arr[:, i_task] != 'rest'
            self.data_arr = self.data_arr[sel]
            self.time_index = self.time_index[sel]
        


def load_stored_merged_data(
    sub: str,
    data_version,
    float_convert: bool = True,
    return_as_df: bool = False,
):
    """
    Loads stored array with all merged data
    from a patient, currently all ephys and
    aggregated ACC-states

    Inputs:
        - sub: string code of subject, e.g. '001'
        - data_version: e.g. 'v0.1'
        - float_convert: set True to convert from
            python-floats to np.float64's, important
            for scipy/fft compatibility
        - return_as_df: set True to return pandas DF,
            else data-array (np), fs (int), col_names (list),
            and indices (floats) are returned
        
    """
    data_path = fileMng.get_project_path('data')
    # set subject-spec pathbase
    path = os.path.join(
        data_path,
        'merged_sub_dfs',
        data_version,
    )
    files = os.listdir(path)
    files = [f for f in files if sub in f]

    print(f'-> loading sub {sub} data from {path}')
    # load data from npy array
    data_file = [f for f in files if 'data' in f][0]
    dat_arr = np.load(os.path.join(path, data_file),
                      allow_pickle=True,)
    print('\t...data loaded')

    # load indices from npy array
    index_file = [f for f in files if 'timeIndex' in f][0]
    time_arr = np.load(os.path.join(path, index_file),
                       allow_pickle=True,)

    # load column-names from txt
    names_file = [f for f in files if 'columnNames' in f][0]
    col_names = np.loadtxt(os.path.join(path, names_file),
                           dtype=str, delimiter=',')
    
    fs_string = data_file.split('Hz')[0]
    fs = int(fs_string.split('_')[-1])

    if float_convert:
        print('\t...correct npy floats')
        dat_arr = convert_to_npfloats(dat_arr, col_names)

    if return_as_df:
        sub_df = pd.DataFrame(
            data=dat_arr,
            index=time_arr,
            columns=col_names,
        )
        print(
            f'--> ...resulting DF has shape {sub_df.shape}, Fs: {fs}'
            f' and columns: {sub_df.keys()}')

        return sub_df
    
    else:
        print('--> ...return data as np.array, fs, col_names, and indices')
        return dat_arr, fs, col_names, time_arr


def convert_to_npfloats(dat_arr, col_names):
    """
    Convert LFP and ECOG signals from py
    floats to np.float64's. This for
    spectral decomposition functions
    """
    for col, key in enumerate(col_names):
        # skip non ephys channels
        if np.logical_or(
            'lfp' in key.lower(), 
            'ecog' in key.lower()
        ):

            dat_arr[:, col] = dat_arr[:, col].astype(np.float64)
    
    return dat_arr


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

    Applied in SubjectData() (run_mergeDataClass())

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
        'data', 'preprocessed_data',
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

            data_out = pd.concat([data_out, rec_data])  # vertical adding of empty rows


    data_out = data_out.sort_values(
        axis=0, by='dopa_time'
    ).reset_index(drop=True)

    return data_out, fs


def merge_ephys_sources(
    subdat
):
    sources = [
        s for s in subdat.dtypes if np.logical_or(
            'ecog' in s, 'lfp' in s
        )
    ]
    print('... merging dataframes from different data-sources')
    merge_df = getattr(
        subdat, sources[0]
    ).data.set_index('dopa_time')
    
    for i in np.arange(1, len(sources)):

        merge_df = merge_df.join(
            getattr(subdat, sources[i]).data.set_index(
                'dopa_time'), rsuffix='_DUPL'
        )

    # remove duplicate columns (task/move-states)
    cols_del = [c for c in merge_df.columns if 'DUPL' in c ]
    merge_df = merge_df.drop(columns=cols_del)

    # include sample frequencies
    fs = getattr(subdat, sources[0]).fs
    # check if EPHYS has same FS
    for i in np.arange(1, len(sources)):
        fs2 = getattr(subdat, sources[i]).fs
        if fs2 != fs:
            raise ValueError(
                'Preprocessed Ephys-Loading Error: '
                'Unequal Sampling Freqs for: '
                f'{fs} -> {sources[0]} vs '
                f'{fs2} -> {sources[i]}'
            )

    return merge_df, fs



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


# def read_files(fsel, groups, fdir):
#     '''
#     OBSOLETE????
    
#     Function which reads the saved preprocessed 3d-ararys and lists
#     with channel-names again into Python Objects.

#     Arguments:
#         - runInfo: class contains the location where the files of the defined
#         ft-version are saved

#     Returns:
#         - data (dict): dict with a 3d array of processed data, per group
#         (e.g. lfp-l, lfp-r, ecog)
#         - names (dict): dict containing the corresponding row-names
#         belonging to the data groups (incl 'time', 'LFP_L_01', etc)
#     '''
#     data = {}  # dict to store processed data arrays
#     names = {}  # dict to store corresponding names
#     for g in groups:
#         # for matching files with defined group
#         g_files = [f for f in fsel if g in f]
#         print(g_files)
#         for f in g_files:
#             csv_f = f'{f[:-8]}rownames.csv'
#             # TODO: CURRENTLY FILES ARE OVERWRITING EACH OTHER
#             # MAKE DATA CLASS HERE, WITH STRUCTURE
#             # CLASS PER PT (SESSION?)
#             #   ACQ/TASK
#             #       GROUP
#             #           - DATA
#             #           - NAMES
#             data[g] = np.load(os.path.join(fdir, f))
#             print('\n', f, 'loaded,\nshape: ', data[g].shape)
#             names[g] = []
#             try:
#                 with open(os.path.join(fdir, csv_f),
#                         newline='') as csvfile:
#                     reader = csv.reader(csvfile, delimiter=',')
#                     for row in reader:
#                         names[g] = row
#             except:
#                 print('fill except...')
        
#         # Data vs names check
#         s1 = data[g].shape[-2]
#         s2 = len(names[g])
#         assert s1 == s2, ('# rows not equal for data (npy)'
#                          f'and names (csv) in {g}')

#     return data, names



# @dataclass(init=True, repr=True, )
# class RunData:
#     '''
#     Creates dataclass containing all data from one specific
#     run.
#     Function is called within SessionData, and will be set
#     to a dict in SessionData.
#     Every RunData consists the requested and available data-
#     types. Every datatype has its own RunData-Class containing
#     the data-array (2d or 3d), rownames, current sample freq.
#     '''
#     npy_files: list
#     fdir: str
#     sub: str
#     ses: str
#     task: str
#     acq: str
#     run: int
#     proc_version: str
#     run_string: str
#     # lfp  -> consider to make extra DataClass type per
#     # data source containing array/names/Fs/... ?
#     lfp_left_arr: array = np.array(0)  # std empty fields
#     lfp_left_names: list = field(default_factory=list)
#     lfp_left_Fs: int = field(default_factory=int)
#     lfp_right_arr: array = np.array(0)
#     lfp_right_names: list = field(default_factory=list)
#     lfp_right_Fs: int = field(default_factory=int)
#     # ecog
#     ecog_arr: array = np.array(0)
#     ecog_names: list = field(default_factory=list)
#     ecog_Fs: int = field(default_factory=int)
#     # acc
#     acc_left_arr: array = np.array(0)
#     acc_left_names: list = field(default_factory=list)
#     acc_left_Fs: int = field(default_factory=int)
#     acc_right_arr: array = np.array(0)
#     acc_right_names: list = field(default_factory=list)
#     acc_right_Fs: int = field(default_factory=int)
#     present_datatypes: list = field(default_factory=list)
 
#     def __post_init__(self, ):
#         # fill present and available ieeg datatypes of given run
#         ### TODO: check whether dataclass variables can be initialized
#         ## right away within loop with setattr()
#         for dtype in [
#             'LFP_LEFT', 'LFP_RIGHT', 'ECOG', 'ACC_LEFT', 'ACC_RIGHT'
#         ]:
#             if (f'{self.run_string}_{dtype}_PREPROC_data.npy' not in
#                     self.npy_files):  # check if dtype exist in files
#                 continue

#             arr, names = read_ieeg_file(
#                 f'{self.run_string}_{dtype}_PREPROC_data.npy', self.fdir
#             )  # set datatype array, names, current sample freq
#             setattr(self, f'{dtype.lower()}_arr', arr)  # set to fields
#             setattr(self, f'{dtype.lower()}_names', names)
#             if len(arr.shape) == 3: timediff = arr[0, 0, 1] - arr[0, 0, 0]
#             if len(arr.shape) == 2: timediff = arr[0, 1] - arr[0, 0]
#             Fs = int(1 / timediff)
#             setattr(self, f'{dtype.lower()}_Fs', Fs)
#             # add dtype to list
#             self.present_datatypes.append(dtype)



# @dataclass(init=True, repr=True, )
# class SessionData:
#     '''Stores different RunData classes from the same Session'''
#     npy_files: list
#     fdir: str
#     runs: dict = field(default_factory=dict)
#     runs_incl: list = field(default_factory=list)
#     ses_name: str = field(default_factory=str)

#     def __post_init__(self,):  # is called after initialization
#         for n, f in enumerate(self.npy_files):
#             split_f = f.split('_')
#             sub = split_f[0]
#             ses = split_f[1]
#             task = split_f[2]
#             acq = split_f[3]
#             run = split_f[4]
#             proc_version = f'{split_f[5]}_{split_f[6]}'
#             run_shortname = f'{task}_{acq}'
#             run_str = f'{sub}_{ses}_{task}_{acq}_{run}_{proc_version}'

#             if n == 0:
#                 self.ses_name = f'{sub}_{ses}'

#             if run_shortname in self.runs_incl:
#                 continue  # skip run if already incl

#             self.runs[run_shortname] = RunData(
#                 npy_files=self.npy_files,
#                 fdir=self.fdir,
#                 sub=sub,
#                 ses=ses,
#                 task=task,
#                 acq=acq,
#                 run=run,
#                 proc_version=proc_version,
#                 run_string=run_str,
#             )
#             self.runs_incl.append(run_shortname)

