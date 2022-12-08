"""
Run Merging data frames
"""

# import public functions
import sys
from os.path import join
from dataclasses import dataclass, field
from numpy import ndarray

# import own functions
from lfpecog_features import feats_read_proc_data as read_data_funcs
from utils.utils_fileManagement import (
    get_project_path, save_dfs,
    mergedData, save_class_pickle)
import lfpecog_features.moveDetection_run as run_tap_detect
from lfpecog_features.feats_add_move_states import (
    add_detected_acc_states as add_moveStates)



@dataclass(init=True, repr=True, )
class subjectData:
    """
    Creates Class with all ephys-data of one subject,
    incl task and sorted by dopa-relative-time,
    optionally include accelerometer labels.

    Class has subclasses per ephys-source (LFP-left,
    LFP-right, ECoG).
    PM: consider adding raw accelerometer, however
    currenbtly resamplede to different sampling frequency.
    
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

        self.dtypes, self.nameFiles, self.dataFiles, sub_path = read_data_funcs.find_proc_data(
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
                read_data_funcs.dopaTimedDf(
                    dType,
                    self.nameFiles,
                    self.dataFiles,
                    sub_path,
                )
            )



if __name__ == '__main__':
    """
    Calling subjectData from command line, creates
    and potentially saves merged dataframes with all
    data per subject (repo/data/...).

    To exclude accelerometer-data, or to NOT save the created
    dataframes, add "no_acc", or "no_save" resp. as a variable
    to the command line call.
    
    Running on WIN (from repo_folder/code):
        python -m lfpecog_features.run_mergeDataClass "012" "v3.0" (opt: "no_acc" "no_save")

    Output:
        - dataframe data/ index/ columnnames stored
            in path: data/merged_sub_dfs/DATAVERSION/
    """
    ### get and check variables out of run-command
    sub = sys.argv[1]
    assert len(sub)==3, print('WRONG INPUT: '
        f'subject code (1st variable: {sub}) incorrect e.g.:"001"'
    )

    data_version = sys.argv[2]
    assert data_version[0]=='v', print('WRONG INPUT: '
        f'data_version (2nd variable: {data_version}) incorrect, e.g.: "v0.1"'
    )
    
    if "no_acc" in sys.argv: incl_acc = False 
    else: incl_acc = True

    if "no_save" in sys.argv:
        to_save = False 
    else:
        to_save = True
        save_as_pickle = True
        if "no_pickle" in sys.argv:
            save_as_pickle = False


    
    ### create dataclass with data sorted per source (lfp-L/R / ecog / acc-L/R)
    data = subjectData(
        sub=sub,
        data_version=data_version,
        project_path=get_project_path(),
    )

    ### Merge dataframes of different data-groups
    merged_df, fs = read_data_funcs.merge_ephys_sources(data)

    from pandas import isna
    print('\n\tCHECK MISSINGS', merged_df.shape)
    for col in list(merged_df.keys()):
        sumnan = sum(isna(merged_df[col]))
        print(f'{col}: {sumnan} nans')
    
    
    ### Optionally add Accelerometer
    if incl_acc:
        accStates = run_tap_detect.runTapDetection(data)
        merged_df = add_moveStates(merged_df, accStates)

    
    ### Store dataframe
    if to_save:
        if save_as_pickle:
            merged_class = mergedData(
                sub=sub,
                data_version=data_version,
                data_array=merged_df.values,
                data_colnames=list(merged_df.keys()),
                data_times=list(merged_df.index.values),
                fs=fs,
            )
            save_class_pickle(
                class_to_save=merged_class,
                path=join(get_project_path(), 'data',
                         'merged_sub_data', f'{data_version}'),
                filename=f'{sub}_mergedDataClass_{data_version}',
            )

        else:  # save as data.npy, index.npy, columnNames.csv
            save_dfs(
                df=merged_df,
                folder_path=join(
                    get_project_path(), 'data',
                    'merged_sub_data', f'{data_version}'
                ),
                filename_base=f'{sub}_mergedDf_{fs}Hz',
            )




