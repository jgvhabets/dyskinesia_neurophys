"""
General utilisation functions
"""

# import public packages and functions
from os import getcwd, listdir, makedirs
from os.path import join, exists, dirname
from numpy import (
    logical_and, save, ndarray, where,
    ravel, arange, array, float64, int64
)
from csv import writer
import pickle
from dataclasses import dataclass
import json


def get_project_path(
    subfolder: str = '', USER=False,
):
    """
    Finds path of projectfolder, and
    subfolder if defined, on current machine
    For projectfolder path, no input is required.

    Input:
        - subfolder: data/code/figure to retrieve
            subfolder path
    """
    
    path = getcwd()

    while path[-20:] != 'dyskinesia_neurophys':

        path = dirname(path)
    
    if subfolder in ['data', 'code', 'figures', 'results']:

        if subfolder == 'data' and USER.lower() == 'timon':
            return r'D:\dyskinesia_project\data'

        return join(path, subfolder)
    
    elif len(subfolder) > 0:

        raise ValueError('WARNING: incorrect subfolder')

    elif len(subfolder) == 0:
        return path


def get_onedrive_path(
    folder: str
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder where data is stored

    Folder has to be in ['onedrive', 'figures', 'bids_rawdata']
    """
    folder_options = [
        'onedrive', 'figures', 'bids_rawdata', 'data'
    ]
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')

    path = getcwd()

    while dirname(path)[-5:].lower() != 'users':
        path = dirname(path)
    # path is now Users/username
    onedrive_f = [
        f for f in listdir(path) if logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        ) 
    ]
    path = join(path, onedrive_f[0])
    bidspath = join(path, 'BIDS_Berlin_ECOG_LFP')

    if folder == 'onedrive': return bidspath

    elif folder == 'bids_rawdata':
        return join(bidspath, 'rawdata')

    else:  # must be data or figures
        return join(path, 'dysk_ecoglfp', folder.lower())


def get_beta_project_path(
    folder: str = 'beta'
):
    """
    Device and OS independent function to find
    the synced-OneDrive folder for BETA-project

    Folder has to be in ['onedrive', 'figures', 'bids_rawdata']
    """
    folder = folder.lower()
    folder_options = [
        'beta', 'figures', 'results', 'data'
    ]
    if folder.lower() not in folder_options:
        raise ValueError(
            f'given folder: {folder} is incorrect, '
            f'should be {folder_options}')

    path = getcwd()

    while dirname(path)[-5:].lower() != 'users':
        path = dirname(path)
    # path is now Users/username
    onedrive_f = [
        f for f in listdir(path) if logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        ) 
    ]
    path = join(path, onedrive_f[0])
    betapath = join(path, 'aDBS beta explorations')
    assert exists(betapath), (f'created path {betapath}'
                              ' does not exist')

    if folder == 'beta': return betapath

    else:  # must be data, results or figures
        return join(betapath, folder)
    

def load_ft_ext_cfg(cfg_fname: str, cfg_folder=None):
    # define folder to use, either default or
    # given if cfg_folder is defined
    if isinstance(cfg_folder, str):
        json_path = join(cfg_folder, cfg_fname)
    else:
        json_path = join(get_onedrive_path('data'),
                         'featureExtraction_jsons',
                         cfg_fname)
    
    # open json
    with open(json_path, 'r') as json_data:
        ft_settings =  json.load(json_data)

    return ft_settings

def save_dfs(
    df,
    folder_path: str,
    filename_base: str
):
    """
    Save a dataframe in separate files for
    data, column names, and indices as resp.
    numpy-array's (.npy-), .csv-, and .npy-files.

    Inputs:
        - df: dataframe to store
        - folder_path: folder to store data in
        - filename_base: base of filenames added
            with data / timeIndex / columnNames,
            e.g.: 'sub000_mergedDf'
    """
    # check if (parent)folder exists, if not make folder
    if not exists(folder_path): makedirs(folder_path)

    # save data as npy array with numpy's save function
    save(
        join(folder_path, f'{filename_base}_data'),
        df.values,
    )
    # save index as npy array
    save(
        join(folder_path, f'{filename_base}_timeIndex'),
        df.index.values
    )
    # save column-names as csv
    with open(
        join(folder_path, f'{filename_base}_columnNames.csv'), 'w'
    ) as csvfile:

        write = writer(csvfile)
        write.writerow(
            list(df.keys())
        )
        csvfile.close()
    
    print(
        f'\n\tDataFrame ({filename_base}) '
        f' is stored to {folder_path}\n'
    )



@dataclass(init=True, repr=True,)
class mergedData:
    """
    Class to store merged-data in a class
    with pickle, makes loading from data
    factor 10 faster

    Input:
        - sub: string code of sub, e.g. '001'
        - data_version: e.g. 'v0.0'
        - data: np array of merged data
        - colnames: corresponding column names
        - times: corresponding time vs dopa-intake
    """
    sub: str
    data_version: str
    data: ndarray  # numpy array
    colnames: list
    times: list
    fs: int

    def __post_init__(self,):
        assert len(self.colnames) == self.data.shape[1], (
            f'within mergedData (sub {self.sub}), # col_names '
            f'{len(self.colnames)} does not'
            f'match shape of array {self.data}'
        )
        assert len(self.times) == self.data.shape[0], (
            f'within mergedData (sub {self.sub}), # times '
            f'{len(self.times)} does not'
            f'match shape of array {self.data}'
        )


def save_class_pickle(
    class_to_save,
    path,
    filename,
    extension='.P',
):

    if not exists(path): makedirs(path)

    if not filename[-2:] == '.P': filename + extension
    
    pickle_path = join(path, filename)

    with open(pickle_path, 'wb') as f:
        pickle.dump(class_to_save, f)
        f.close()

    return print(f'inserted class saved as {pickle_path}')


def load_class_pickle(
    file_to_load, convert_float_np64: bool = False,
):
    """
    Loads saved Classes. When running this code
    the class-definitions have to be called before
    executign this code.
    
    So, for example:
    from utils.utils_windowing import windowedData

    loaded_class = utils_fileMng.load_class_pickle(os.path.join(deriv_path, 'classFileName.P'))
    
    Input:
        - file_to_load: string including path,
            filename, and extension
    
    Returns:
        - output: variable containing the class
    """

    try:
        with open(file_to_load, 'rb') as f:
            output = pickle.load(f)
            f.close()
    except:
        print(f'ERROR while pickle loading of {file_to_load}')
        with open(file_to_load, 'rb') as f:
            output = pickle.load(f)
            f.close()

    # if times is just indices 0,1,2,...
    if 'times' in vars(output).keys():
        if (output.times[:5] == arange(5)).all():
            # find dopa-time values
            i_time = where(array(output.colnames) == 'dopa_time')[0][0]
            # set dopa-time values as times attr
            if 'data_arr' in vars(output).keys():
                setattr(output, 'times' , output.data_arr[:, i_time])
            elif 'data' in vars(output).keys():
                setattr(output, 'times' , output.data[:, i_time])
    
    if convert_float_np64:
        if 'data' in vars(output).keys():
            if isinstance(output.data, ndarray):
                setattr(output, 'data', output.data.astype(float64))


    return output


def make_object_jsonable(obj):
    """
    give object to convert content 
    to json-compatible datatypes (list instead
    of array, no np floats or integers)
    """
    if isinstance(obj, ndarray):

        obj = list(obj)

        new_list = []

        for item in obj:
            
            new_item = make_object_jsonable(item)
            new_list.append(new_item)
        
        return new_list
            
    elif isinstance(obj, float64):

        obj = float(obj)

        return obj
    
    elif isinstance(obj, int64):

        obj = int(obj)

        return obj

    elif isinstance(obj, dict):
        
        for k in obj.keys():
            
            item = obj[k]
            new_item = make_object_jsonable(item)
            obj[k] = new_item

        return obj

    elif isinstance(obj, list):
        
        new_list = []

        for item in obj:
            
            new_item = make_object_jsonable(item)
            new_list.append(new_item)
        
        return new_list


    return obj


def correct_acc_class(acc):
    """
    Correct for dopa times and
    flipped sides in Acc-DataClass, needed
    in dataversion v3.1
    """
    flipped_sides = ['009',]

    if (list(acc.times[:3]) == ['0', '1', '2'] or
        list(acc.times[:3]) == [0, 1, 2]):

        sel = [c == 'dopa_time' for c in acc.colnames]
        time_arr = ravel(acc.data.T[sel])
        setattr(acc, 'times', time_arr)
    
    if acc.sub in flipped_sides and acc.data_version == 'v3.1':
        names = acc.colnames.copy()

        for m in ['tap', 'move']:
            i_left = where([c == f'left_{m}' for c in acc.colnames])[0][0]
            i_right = where([c == f'right_{m}' for c in acc.colnames])[0][0]
            names[i_left] = f'right_{m}'
            names[i_right] = f'left_{m}'
        setattr(acc, 'colnames', names)
    
    return acc
