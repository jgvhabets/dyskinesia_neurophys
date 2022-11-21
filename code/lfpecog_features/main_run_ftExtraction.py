"""
Run Feature Extraction
"""

# import public functions
import sys
import time
from os.path import join
from dataclasses import dataclass, field

# import own functions
from lfpecog_features import feats_read_proc_data as read_data
from utils.utils_fileManagement import get_project_path, save_dfs

# import other functions
import hackathon_mne_mvc as multiVarConn

if __name__ == '__main__':

    """
    Running on WIN (from repo_folder/code):
        python -m lfpecog_features.main_run_ftExtraction ["012",]
    """
    starttime = time.time()

    sub = sys.argv[1]
    data_as_df = False
    data = read_data.main_loadMergedData(
        list_of_subs = [sub,],
        tasks = ['rest'],
        data_version='v3.0',
        float_convert=True,
        data_as_df=data_as_df,
    )
    endtime = time.time()

    print(f'TIME PASSED: {endtime - starttime} seconds')    
    
    if data_as_df:
        print(data.rest.sub012.shape, data.rest.sub012.keys)
    
    else:
        classDat = data.rest.sub012

        print(classDat.fs, classDat.col_names,
              classDat.data_arr.shape, classDat.time_index.shape)
    
    del(data)

    