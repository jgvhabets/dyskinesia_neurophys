"""
General utilisation functions
"""

# import public packages and functions
from os import getcwd, listdir, mkdir
from os.path import join, exists, dirname
from numpy import logical_and, save
from csv import writer


def get_project_path(
    subfolder: str = '',
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
    
    if subfolder in ['data', 'code', 'figures']:

        return join(path, subfolder)
    
    elif len(subfolder) > 0:

        print('WARNING: incorrect subfolder')

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
    if not exists(dirname(folder_path)): mkdir(dirname(folder_path))
    if not exists(folder_path): mkdir(folder_path)

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
