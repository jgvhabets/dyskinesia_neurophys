"""
General utilisation functions
"""

# import public packages and functions
import os
from numpy import logical_and


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
    
    path = os.getcwd()

    while path[-20:] != 'dyskinesia_neurophys':

        path = os.path.dirname(path)
    
    if subfolder in ['data', 'code', 'figures']:

        return os.path.join(path, subfolder)
    
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
        
    path = os.getcwd()
    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)
    # path is now Users/username
    onedrive_f = [
        f for f in os.listdir(path) if logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        ) 
    ]
    path = os.path.join(path, onedrive_f[0])
    bidspath = os.path.join(path, 'BIDS_Berlin_ECOG_LFP')

    if folder == 'onedrive': return bidspath

    elif folder == 'bids_rawdata':
        return os.path.join(bidspath, 'rawdata')

    else:  # must be data or figures
        return os.path.join(path, 'dysk_ecoglfp', folder.lower())


