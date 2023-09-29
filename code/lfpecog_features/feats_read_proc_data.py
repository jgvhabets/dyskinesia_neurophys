'''
Functions to create/sotre and read/load merged-data
consisting of preprocessed ephys data and accelero-
meter based movement states

Both functionalitites (create and load) are called
from main functions in:
- main_run_ftExctraction (load)
- run_mergeDataClass (create and store)
'''

# Import general packages and functions
import os
from os.path import join
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import csv
from array import array

# import own functions
from utils.utils_fileManagement import (
    mergedData, save_class_pickle,
    get_project_path, load_class_pickle
)



@dataclass(init=True, repr=True, )
class main_loadMergedData:
    """
    Main class to run to load merged dataframes per subject,
    takes merged dataframes from already preprocessed
    and stored data

    TODO: include filtering on ACC-state
    TODO: get rid of task, sub subclasses

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
    tasks: list = field(default_factory=lambda: ['all', 'rest', 'tap'])
    data_as_df: bool = False
    data_as_class: bool = True
    # filter_on_ACC: str/list

    def __post_init__(self,):

        if type(self.tasks) == str:
            self.tasks = [self.tasks]

        # create a seperate class per task-selection 
        for task in self.tasks:

            if task not in ['all', 'rest', 'tap']:
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
                    return_as_class=self.data_as_class,
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
    return_as_df: bool = False
    return_as_class: bool = True
    acc_filter: str = None
    

    def __post_init__(self,):

        for sub in self.list_of_subs:

            print(
                f'\tloading Sub {sub} ({self.task})'
            )
            # LOAD DATA
            mergedData_class = load_stored_merged_data(
                sub=sub,
                data_version=self.data_version,
                float_convert=self.float_convert,
                return_as_df=self.return_as_df,
                return_as_class=self.return_as_class,
            )
            # choose for now scenario return as class
            # TODO: get rid of extra subject layer, since always done per subject
            # perform task selection within this class
            setattr(
                    self,
                    f'sub{sub}',
                    subData_asArrays(
                        data_arr=mergedData_class.data,
                        fs=mergedData_class.fs,
                        col_names=mergedData_class.colnames,
                        time_index=mergedData_class.times,
                        task_sel=self.task,
                    )
                )
                
            print(f'...data merged for sub-{sub}')


            # # unpack tuple if not returning as df
            # if self.return_as_class:
                
            # elif not self.return_as_df:
            #     dat_arr, fs, col_names, time_arr = merged_data
            
            # # SELECT ON TASK (if returned as df)
            # if self.return_as_df:
            #     sel = df['task'] == self.task
            #     df = df[sel]
            #     # for non-df return this happens in class-function
            
            # # SET AS ATTRIBUTE (dataframe or class)
            # if self.return_as_df:
            #     setattr(
            #         self,
            #         f'sub{sub}',  # name is subXXX
            #         df
            #     )
            
            # else:
            #     setattr(
            #         self,
            #         f'sub{sub}',
            #         subData_asArrays(
            #             data_arr=dat_arr,
            #             fs=fs,
            #             col_names=col_names,
            #             time_index=time_arr,
            #             task_sel=self.task,
            #         )
            #     )
                
            # print(f'...data merged for sub-{sub}')


@dataclass(init=True, repr=True, )
class subData_asArrays:
    data_arr: array
    fs: int
    col_names: list
    time_index: array
    task_sel: str = field(default_factory='')

    def __post_init__(self,):

        if len(self.task_sel) > 1:
            # find task column
            i_task = np.where([c == 'task' for c in self.col_names])[0][0]
            # select on performed task
            sel = self.data_arr[:, i_task] == self.task_sel
            self.data_arr = self.data_arr[sel]
            self.time_index = np.array(self.time_index)[sel]
        


def load_stored_merged_data(
    sub: str,
    data_version,
    return_as_df: bool = False,
    return_as_class: bool = False,
    float_convert: bool = True,
    save_as_pickle: bool = True,
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
        - return_as_df: set True to return pandas DF
        - return_as_class: set True to return
            mergedData() dataclass
            PM: if both are False: default return is as
            separate data-array (np), fs (int),
            col_names (list), and indices (floats)        
    """
    data_path = get_project_path('data')
    path = join(data_path, 'merged_sub_data',
                        data_version,)
    files = os.listdir(path)
    files = [f for f in files if sub in f]

    print(f'-> loading stored merged data for sub {sub} data\n\tfrom {path}')

    # load pickle if present
    pickle_name = f'{sub}_mergedDataClass_{data_version}'
    if f'{pickle_name}.P' in files:
        print('...load pickled mergedData Class')
        mergedData_class = load_class_pickle(join(
            path, f'{sub}_mergedDataClass_{data_version}.P'
        ))
        dat_arr = mergedData_class.data
        fs = mergedData_class.fs
        col_names = mergedData_class.colnames
        time_arr = np.array(mergedData_class.times)


    else:
        # load DATA from npy array
        if len(files) == 0:
            raise ValueError('No merged data files found for '
                f'sub {sub} in data version {data_version}')
        
        data_file = [f for f in files if 'data' in f][0]
        dat_arr = np.load(join(path, data_file),
                        allow_pickle=True,)
        print('\t...data loaded')
        
        # load COLUMN-NAMES from txt
        names_file = [f for f in files if 'columnNames' in f][0]
        col_names = np.loadtxt(join(path, names_file),
                            dtype=str, delimiter=',')
        
        # extract TIMES for data array
        i_time = np.where(col_names == 'dopa_time')[0][0]
        time_arr = dat_arr[:, i_time]
        
        fs_string = data_file.split('Hz')[0]
        fs = int(fs_string.split('_')[-1])

        if float_convert:
            print('\t...correct npy floats')
            dat_arr = convert_to_npfloats(dat_arr, col_names)
        
        # convert and save file as pickle for later usage
        if save_as_pickle:
            print('...saving pickle')
            mergedData_class = mergedData(
                sub=sub,
                data_version=data_version,
                data=dat_arr,
                colnames=col_names,
                times=time_arr,
                fs=fs,
            )
            save_class_pickle(
                class_to_save=mergedData_class,
                path=path, filename=pickle_name,
            )
            # seperate mergedDataClass for none-ephys data
            sel = ['LFP' not in k and 'ECOG' not in k for k in col_names]
            mergedData_class = mergedData(
                sub=sub,
                data_version=data_version,
                data=dat_arr[:, sel],
                colnames=col_names[sel],
                times=time_arr,
                fs=fs,
            )
            save_class_pickle(
                class_to_save=mergedData_class,
                path=path, filename=f'{pickle_name}_noEphys',
            )


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
    
    elif return_as_class:

        return mergedData_class
    
    else:

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


def find_proc_data(sub: str, version: str,
                   load_EXT_HD: bool = False):
    '''
    Find available runs, datatypes, and all filenames
    for one subject, considers specific data-version
    used for preprocessing settings
    '''
    # first try local directory
    try:
        # get available files for sub + version
        sub_proc_path = join(
            get_project_path('data', extern_HD=load_EXT_HD),
            'preprocessed_data', f'sub-{sub}', version,
        )
        files = os.listdir(sub_proc_path)
        # select relevant npy/json files
        datafiles = [f for f in files if f[-3:] == 'npy']
        namefiles = [f for f in files if f[-3:] == 'csv']
    
    # try ext HD if local not working
    except:
        # get available files for sub + version
        sub_proc_path = join(
            get_project_path('data', extern_HD=True),
            'preprocessed_data', f'sub-{sub}', version,
        )
        files = os.listdir(sub_proc_path)
        # select relevant npy/json files
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

    Input:
        - dType (str): specific datatype to handle (acc, lfp, ecog)
        - allNames (list): all name-files corresponding to subj
        - allData (list): all data-files corresponding to subj
        - sub_proc_path (str): path where data/name files are stored
    """
    nameFiles = [f for f in allNames if dType in f]
    dataFiles = [f for f in allData if dType in f]

    for nFile, datFile in enumerate(dataFiles):
                    
        rec = datFile.split('_')[3]
        nameFile = [f for f in nameFiles if rec in f][0]
        names = list(pd.read_csv(join(sub_proc_path,
                                      nameFile)))[1:]  # exclude run_time  # TODO. hardcoded -> change

        task = datFile.split('_')[2]
        if 'rest' in task.lower(): task = 'rest'
        elif 'tap' in task.lower(): task = 'tap'
        elif 'free' in task.lower(): task = 'free'
        elif 'flip' in task.lower(): task = 'tap'
        elif 'rotation' in task.lower():
            print(f'NOTE: Rotation task detected and skipped in {datFile}')
            continue

        fs = datFile.split('Hz')[0]
        fs = fs.split('_')[-1]

        if nFile == 0:

            data_out = pd.DataFrame(
                data=np.load(join(sub_proc_path, datFile))[1:, :].T,
                columns=names)
            data_out['task'] = [task] * data_out.shape[0]
        
        else:

            rec_data = pd.DataFrame(
                data=np.load(join(sub_proc_path, datFile))[1:, :].T,
                columns=names, index=None,
            )

            rec_data['task'] = [task] * rec_data.shape[0]

            data_out = pd.concat([data_out, rec_data])  # vertical adding of empty rows

    data_out = data_out.sort_values(
        axis=0, by='dopa_time'
    ).reset_index(drop=True)

    # correct NON-BIDS preproc for micro-units
    if 'stim' not in datFile.lower():
        print(f'CORRECT MICRO-UNITS for none-BIDS-load ({datFile})')
        for i, c in enumerate(data_out.keys()):
            if 'acc' in c.lower() or 'lfp' in c.lower() or 'ecog' in c.lower():
                print(f'...correct {c}')
                data_out[c] = data_out[c].values * 1e-6
    
    return data_out, fs


def merge_ephys_sources(
    subdat
):
    """
    Merges ephys and acc-data from one subject

    Input:
        - subdat: class from subjectData(), containing
            all data per datatype (see lfpecog_features/
            run_mergeDataClass.py)
    """
    sources = [
        s for s in subdat.dtypes if np.logical_or(
            'ecog' in s.lower(), 'lfp' in s.lower()
        )
    ]
    print('...merging dataframes from different data-sources (merge_ephys_sources())')
    print(f'...# {len(sources)} source-datatypes found')
    merged_df = getattr(
        subdat, sources[0]
    ).data.set_index('dopa_time')
    
    for i in np.arange(1, len(sources)):

        merged_df = merged_df.join(getattr(subdat, sources[i]).data.set_index(
                'dopa_time'), rsuffix='_DUPL'
        )
        print(f'...datatype added ({i + 1}/{len(sources)})...')

    # remove duplicate columns (task/move-states)
    cols_del = [c for c in merged_df.columns if 'DUPL' in c ]
    merged_df = merged_df.drop(columns=cols_del)

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

    print('...dataframe merged (merge_ephys_sources())')


    return merged_df, fs



def read_ieeg_file(npy_f, fdir):
    '''
    
    '''
    csv_f = f'{npy_f[:-8]}rownames.csv'

    data = np.load(join(fdir, npy_f))
    names = []
    try:
        with open(join(fdir, csv_f), newline='') as csvfile:
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

