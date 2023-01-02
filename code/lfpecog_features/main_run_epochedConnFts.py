"""
Run Feature Extraction of (multi-
variate) Connectivity based on
epoched windows of data

TODO: dont save all in between steps,
only save final class-pickle of epoched-
arrays due to computational speed
"""

# import public functions
import sys
import time
from os.path import join, exists
from os import listdir, makedirs
import csv
from numpy import save, load, array
from pandas import DataFrame, read_csv


# import own functions
from utils.utils_fileManagement import (
    get_project_path,
    save_class_pickle,
    load_class_pickle,
)
from lfpecog_features import feats_read_proc_data as read_data
from lfpecog_features.feats_multivarConn import run_mne_MVC


if __name__ == '__main__':

    """
    Runs multivariate connectivity computation over
    time (windows, mne-epoching within windowa) per
    subject.

    Give subject-code as first argument,
    give mvc-method (mim or mic) as second arg
    
    Running on WIN (from repo_folder/code):
        (activate conda environment with custom mne_connectivity)
        python -m lfpecog_features.main_run_epochedConnFts "012" "mic"
    """
    sub = sys.argv[1]
    mvc_method = sys.argv[2].upper()
    assert mvc_method in ["MIC", "MIM"], (
        'second argument should be MVC method MIC or MIM'
    )
    
    MAIN_starttime = time.time()

    # variables (set into cfg.json later)
    tasks = ['rest']
    data_version = 'v3.1'
    winLen_sec = 60
    part_winOverlap = 0.5
    mne_format = True
    epochLen_sec = .5
    take_abs = True

    # create dict with results per task for plotting
    mvc_values_per_task = {}

    # method to print in filenames
    if take_abs: print_method = f'abs{mvc_method}'
    else: print_method = mvc_method

    # create directories
    results_sub_dir = join(get_project_path('results'), 'features', 'mvc', f'sub{sub}')
    data_sub_dir = join(get_project_path('data'), f'windowed_data_classes_{winLen_sec}s',
                        data_version, f'sub-{sub}')
    mvc_figures_dir = join(get_project_path('figures'), 'ft_exploration', 'mvc')

    for f in [results_sub_dir, data_sub_dir, mvc_figures_dir]:
        if not exists(f): makedirs(f)

    for task in tasks:

        # check whether results are already present
        mvc_fts_task_file = (f'mvc_fts_{sub}_{print_method}_{task}_{data_version}'
            f'win{winLen_sec}s_overlap{part_winOverlap}.csv'
        )
        if exists(join(results_sub_dir, mvc_fts_task_file)):

            results_df = read_csv(join(results_sub_dir, mvc_fts_task_file), index_col=0)
            mvc_values_per_task[task] = results_df.values
            mvc_freqs = results_df.keys()
            mvc_times = results_df.index
            print(f'\t...existing mvc-features are loaded for task: {task}')
            # if results for task are loaded, skip remainder of loop for this task
            continue

        # IF RESULTS ARE NOT PRESENT YET, THEY ARE CALCULATED HERE

        # define path names
        windowed_class_path = join(data_sub_dir, task)
        mneEpochs_pickle_fname = (f'{sub}_mneEpochs_{task}_{data_version}_'
                        f'win{winLen_sec}s_overlap{part_winOverlap}')
        windows_pickle_fname = (f'{sub}_windows_{task}_{data_version}_'
                        f'win{winLen_sec}s_overlap{part_winOverlap}')

        pickled_epochs_path = join(windowed_class_path, f'{mneEpochs_pickle_fname}.P')
        pickled_windows_path = join(windowed_class_path, f'{windows_pickle_fname}.P')
        
        # load mne-Epochs saved as pickle Class
        if exists(pickled_epochs_path):
            print('...loading pickled mneEpochs (from: '
                  f'{pickled_epochs_path})')

            from utils.utils_pickle_mne import pickle_EpochedArrays
            class_mne_epochs = load_class_pickle(pickled_epochs_path)
            list_mneEpochArrays = class_mne_epochs.list_mne_objects
            print(f'...loaded list with {len(list_mneEpochArrays)}'
                  ' mneEpochedArrays is loaded')

        # if MNE-epochs are not existing yet as pickled file
        else:  
            # load data if windowed-dataclass-pickle exists
            if exists(pickled_windows_path):
                
                print(f'...load existing pickled windowed class for sub-{sub}')
                starttime = time.time()
                
                from utils.utils_windowing import windowedData
                windows_class = load_class_pickle(pickled_windows_path)

                endtime = time.time()
                print(f'\tloading took {endtime - starttime} seconds')  

            # create windowed data class if it doesnt exist yet
            else:
                
                print(f'...create and store windowed class for sub-{sub}')
                from utils.utils_windowing import get_windows
                starttime = time.time()

                data_as_df = False  # get data object as class
                data = read_data.main_loadMergedData(
                    list_of_subs = [sub,],
                    tasks = tasks,
                    data_version=data_version,
                    float_convert=True,
                    data_as_df=data_as_df,
                )  # data consists full unwindowed data
                
                print('...data loaded')

                # TODO: get rid of sub task subclasses
                mergedData_class = getattr(data, task)
                mergedData_class = getattr(mergedData_class, f'sub{sub}')
                
                print('...start windowing')
                windows_class = get_windows(
                    data=mergedData_class.data_arr,
                    fs=mergedData_class.fs,
                    winLen_sec=winLen_sec,
                    part_winOverlap=part_winOverlap,
                    col_names=mergedData_class.col_names,
                    return_as_class=True,
                )
                print('...windowed the data into windowedData() Class')
                endtime = time.time()
                print(f'\tcreating took {round(endtime - starttime)} seconds')    

                save_class_pickle(
                    class_to_save=windows_class,
                    path=windowed_class_path,
                    filename=windows_pickle_fname,
                    extension='.P',
                )
            
            # create epochs based on windowed data class (loaded or created)            
            from utils.utils_windowing import window_to_epochs
            
            epoched_windows_list = []
            # define fs and ch_names in variables for mne transform
            fs = windows_class.fs
            ch_names = windows_class.keys
            window_times = windows_class.win_starttimes

            for n, win in enumerate(windows_class.data):

                epochs = window_to_epochs(
                    win_array=win,
                    fs=windows_class.fs,
                    epochLen_sec=epochLen_sec,
                    remove_nan=True,
                    mne_format=mne_format
                )
                epoched_windows_list.append(epochs)
                
        
            # transform epoched array into mne-EpochedArray
            from utils.utils_windowing import create_mne_epochs
            mne_starttime = time.time()
            print(f'start MNE transform with # {len(epoched_windows_list)} windows')
            list_mneEpochArrays = create_mne_epochs(
                epoched_windows_list,
                fs=fs, ch_names=ch_names,
                pick_only_ephys=True,
            )
            mne_stoptime = time.time()
            print(f'MNE transform took: {mne_stoptime - mne_starttime} seconds')
            
            from utils.utils_pickle_mne import pickle_EpochedArrays

            class_mne_epochs = pickle_EpochedArrays(
                list_mne_objects=list_mneEpochArrays,
                window_times=window_times,
            )
            save_class_pickle(
                class_to_save=class_mne_epochs,
                path=windowed_class_path,
                filename=mneEpochs_pickle_fname,
                extension='.P',
            )

        # (within task loop) get MVC feature values from MNE-Epochs
        # use custom multivariate connectivity git-fork
        MAIN_stoptime = time.time()
        print(f'FULL prep-SCRIPT TOOK for task: {task}: '
             f'{round((MAIN_stoptime - MAIN_starttime) / 60, 1)} minutes')
        # keep track of calculations in report txt-file
        mvc_report_path = join(
            get_project_path('results'),
            'features', 'mvc',
            f'{sub}_mvc_{mvc_method}_report_{task}_{data_version}'
            f'win{winLen_sec}s_overlap{part_winOverlap}.txt'
        )

        mne_starttime = time.time()
        mvc_results = run_mne_MVC(
            mvc_method=mvc_method,
            list_mneEpochArrays=list_mneEpochArrays,
            report=True,
            report_path=mvc_report_path,
        )
        # returns list of mne_connectivity.base.SpectralConnectivity per epoched-window in list
        mne_stoptime = time.time()
        
        print(f'mne-mvc script time: {round((mne_stoptime - mne_starttime) / 60, 1)} minutes'
              f'\n\t# MVC-result-windows: {len(mvc_results)}')

        # make 2d array of mvc results of several windows
        mvc_values_per_task[task] = array([
            mvc_results[i].get_data()[0]
            for i in range(len(mvc_results))
        ])
        # take absolute imag coh
        if take_abs: mvc_values_per_task[task] = abs(mvc_values_per_task[task])
        # get mvc-freqs and -times
        mvc_freqs = mvc_results[0].freqs
        mvc_times = class_mne_epochs.window_times

        # save mvc results in csv
        mvc_task_df = DataFrame(
            data=mvc_values_per_task[task], index=mvc_times, columns=mvc_freqs)
        mvc_task_df.to_csv(
            join(results_sub_dir, mvc_fts_task_file),
            index=True, header=True, sep=',',
        )


    ### PLOT RESULTS ###
    from lfpecog_plotting.plot_timeFreq_Connectivity import plot_mvc
    
    # TODO: merge result dataframes from different tasks
    add_CDRS = True
    add_ACC = True

    if add_CDRS: print_method += '_cdrs'
    if add_ACC: print_method += '_acc'

    # Run Plotting
    plot_mvc(
        sub=sub,
        plot_data=mvc_values_per_task[task],
        plot_freqs=mvc_freqs,
        plot_times=mvc_times,
        add_CDRS=add_CDRS,
        add_ACC=add_ACC,
        data_version=data_version,
        fs=16,
        mvc_method=mvc_method,
        cmap='viridis',
        to_save=True,
        save_path=mvc_figures_dir,
        fname=(f'{sub}_mvc_{print_method}_{task}_{data_version}_'
              f'win{winLen_sec}s_overlap{part_winOverlap}'),
    )
 

