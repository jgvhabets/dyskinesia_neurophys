"""
Run Merging data frames
"""

# import public functions
import sys
from os.path import join
from os import listdir
from numpy import array

# import own functions
from utils.utils_fileManagement import (
    get_project_path,
    mergedData,
    load_class_pickle,
    save_class_pickle
)


if __name__ == '__main__':
    """
    Check and create pickled mergedData classes with
    no ephys, only ACC and task-labels

    Arguments:
        - dataversion as 'v1.1'

    Running on WIN (from repo_folder/code):
        python -m lfpecog_features.run_extract_noEphys_mergedData "v3.1"

    Output:
        - dataframe data/ index/ columnnames stored
            in path: data/merged_sub_dfs/DATAVERSION/
    """
    ### get and check variables out of run-command

    data_version = sys.argv[1]

    assert data_version[0]=='v', print('WRONG INPUT: '
        f'data_version (2nd variable: {data_version}) incorrect, e.g.: "v0.1"'
    )
    
    pickle_path = join(get_project_path('data'),
                       'merged_sub_data', data_version)

    # Find subs with pickled full mergedData
    files = [f for f in listdir(pickle_path)
             if f[-2:] == '.P' and data_version in f
             and 'noEphys' not in f]
    subs = list(set([f[:3] for f in files]))
    # exclude subs with already noEphys pickled data
    noEphys_files = [f for f in listdir(pickle_path)
             if f[-2:] == '.P' and data_version in f
             and 'noEphys' in f]
    doneSubs = list(set([f[:3] for f in noEphys_files]))
    subs = [s for s in subs if s not in doneSubs]

    # create noEphys mergedData per sub
    for sub in subs:
        
        print(f'...start sub {sub}')
        f = f'{sub}_mergedDataClass_{data_version}.P'
        fullClass = load_class_pickle(join(pickle_path, f))
        # select and insert none Ephys columns
        sel = ['LFP' not in c and 'ECOG' not in c
               for c in fullClass.colnames]
        new_data = fullClass.data[:, sel]
        new_cols = array(fullClass.colnames)[sel]
        setattr(fullClass, 'data', new_data)
        setattr(fullClass, 'colnames', new_cols)
        # save new class
        save_class_pickle(class_to_save=fullClass,
                          path=pickle_path,
                          filename=f'{sub}_mergedDataClass_{data_version}_noEphys.P')
        
        print(f'saved noEphys mergedData for sub {sub}')
        del(fullClass)

    
    