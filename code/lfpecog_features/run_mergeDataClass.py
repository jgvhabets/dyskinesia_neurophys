"""
Run Merging data frames
"""

# import public functions
import sys
from os.path import join, exists
from os import listdir, makedirs
from dataclasses import dataclass, field
from numpy import ndarray, logical_and

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

        (self.dtypes,
         self.nameFiles,
         self.dataFiles,
         sub_path) = read_data_funcs.find_proc_data(
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
        OR: python -m lfpecog_features.run_mergeDataClass "all" "v4.0"
        NOTE: 'all' refers to the list all_subs defined below
    Output:
        - dataframe data/ index/ columnnames stored
            in path: data/merged_sub_data/DATAVERSION/
    """
    all_subs = ['105', '107', '108', '109',
                '019', '020', '021',]  # '106' exteme noisy
    # all_subs = 'define all subs'

    OVERWRITE = False  # if False, existing files are not new created and overwritten

    ### get and check variables out of run-command
    SUB = sys.argv[1]

    DATA_VERSION = sys.argv[2]
    assert DATA_VERSION[0]=='v', print('WRONG INPUT: '
        f'data_version (2nd variable: {DATA_VERSION}) incorrect, e.g.: "v0.1"'
    )
    
    if "no_acc" in sys.argv: INCL_ACC = False 
    else: INCL_ACC = True

    if "no_save" in sys.argv:
        TO_SAVE = False 
    else:
        if "no_pickle" in sys.argv:
            TO_SAVE = 'df'
        else:
            TO_SAVE = 'P'  # save as pickle

    # create separate pickles (mergedData) per LFP-L/R and ECOG
    if DATA_VERSION in ['v3.0', 'v4.0']:
        PICKLE_PER_SOURCE = True  # true for SSD pipeline
    elif DATA_VERSION == 'v3.1':
        PICKLE_PER_SOURCE = False  # False for MNE-MVC pipeline
    else:
        raise ValueError(f'define PICKLE_PER_SOURCE info for dataversion: {DATA_VERSION}')
    
    if SUB.lower() == 'all': sub_list = all_subs
    elif isinstance(SUB, str) and len(SUB) == 3: sub_list = [SUB,]
    
    for SUB in sub_list:
        assert len(SUB)==3, ('WRONG INPUT: subject code '
            f'(1st variable: {SUB}) incorrect e.g.:"001"')
        ### create dataclass with data per source as attribute
        # (lfp-L/R / ecog / acc-L/R) as dopaTimedDf Class
        data = subjectData(
            sub=SUB,
            data_version=DATA_VERSION,
            project_path=get_project_path(),
        )

        ### Merge dataframes of different data-groups
        if not PICKLE_PER_SOURCE:
            merged_df, fs = read_data_funcs.merge_ephys_sources(data)

            from pandas import isna
            print('\n\tCHECK MISSINGS', merged_df.shape)
            for col in list(merged_df.keys()):
                sumnan = sum(isna(merged_df[col]))
                print(f'{col}: {sumnan} nans')
        
        
        ### Optionally add Accelerometer
        if INCL_ACC:
            print(f'START MOVEMENT DETECTION for sub-{SUB}')
            accStates = run_tap_detect.runTapDetection(data)
        
            if not PICKLE_PER_SOURCE:
                merged_df = add_moveStates(merged_df, accStates)
                print('...movement states added (in run_mergeDataClass)')

        ### Save per dType or all together

        if PICKLE_PER_SOURCE:

            for dType in data.dtypes:
                # if not logical_and(SUB == '019', dType == 'lfp_right'):
                #     print(f'skip sub-{SUB}, {dType}')
                #     continue
                print(f'PROCESS sub-{SUB}, {dType}')
                # skip other dtypes
                if True not in [dType.lower().startswith(s)
                                for s in ['acc', 'lfp', 'ecog']]:
                    continue

                if not OVERWRITE:
                    fname = f'{SUB}_mergedData_{DATA_VERSION}_{dType}.P'
                    path = join(get_project_path('data'), 'merged_sub_data',
                                f'{DATA_VERSION}', f'sub-{SUB}')
                    if not exists(path): makedirs(path)
                    if fname in listdir(path):
                        print(f'{fname} ALREADY EXISTING IN {path}')
                        continue


                # add ACC states to dType df
                if INCL_ACC:
                    data_df = add_moveStates(
                        getattr(data, dType).data.set_index('dopa_time'),
                        accStates
                    )
                else:
                    data_df = getattr(data, dType).data.set_index('dopa_time')
                
                # get sampling freq, convert to class, and save
                fs = getattr(data, dType).fs
                data_class = mergedData(
                    sub=SUB,
                    data_version=DATA_VERSION,
                    data=data_df.values,
                    colnames=list(data_df.keys()),
                    times=list(data_df.index.values),
                    fs=fs,
                )
                if TO_SAVE == False:
                    print(f'dataclass created WITHOUT saving for {dType}')
                    continue
                elif TO_SAVE == 'P':
                    save_class_pickle(
                        class_to_save=data_class,
                        path=join(get_project_path('data'),
                                'merged_sub_data',
                                f'{DATA_VERSION}',
                                f'sub-{SUB}'),
                        filename=f'{SUB}_mergedData_{DATA_VERSION}_{dType}.P',
                    )
                    print(f'...pickled mergedData sub-{SUB}: {dType}')

        elif not PICKLE_PER_SOURCE:
            merged_class = mergedData(
                sub=SUB,
                data_version=DATA_VERSION,
                data=merged_df.values,
                colnames=list(merged_df.keys()),
                times=list(merged_df.index.values),
                fs=fs,
            )
            if TO_SAVE == 'P':
                save_class_pickle(
                    class_to_save=merged_class,
                    path=join(get_project_path(), 'data',
                            'merged_sub_data', f'{DATA_VERSION}'),
                    filename=f'{SUB}_mergedDataClass_{DATA_VERSION}.P',
                )
                print(f'...pickled new mergedData sub-{SUB}')
            
            elif TO_SAVE == 'df':  # save as data.npy, index.npy, columnNames.csv
                save_dfs(
                    df=merged_df,
                    folder_path=join(
                        get_project_path(), 'data',
                        'merged_sub_data', f'{DATA_VERSION}'
                    ),
                    filename_base=f'{SUB}_mergedDf_{fs}Hz',
                )
                print(f'...new merged data ({SUB}) saved as data.npy, index.npy, and colnames.csv')

            else:
                print(f'Full dataclass created WITHOUT saving')

        



