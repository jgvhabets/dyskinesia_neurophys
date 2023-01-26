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
from os import listdir, makedirs, getcwd
import csv
import numpy as np
from pandas import DataFrame, read_csv, concat

wd = getcwd()
sys.path.append(getcwd())  # for debugger
# import own functions
from utils.utils_fileManagement import (
    get_project_path,
    save_class_pickle,
    load_class_pickle,
)
from lfpecog_features import feats_read_proc_data as read_data


def run_mvc_per_sub(sub):

    """
    Runs multivariate connectivity computation over
    time (windows, mne-epoching within windowa) per
    subject.

    Give subject-code as first argument,
    give mvc-method (mim or mic) as second arg
    
    Running on WIN (from repo_folder/code):
        (activate conda environment with custom mne_connectivity)
        python -m lfpecog_features.cmdRun_mvc_subs "000" "001" etc
    """
    # single run deprecated, no if main anymore, only via group running (cmdRun_mvc_subs)
    # sub = sys.argv[1]
    # mvc_method = sys.argv[2].upper()
    # # DEBUGGING WITHOUT ARGUMENT FILE
    # sub = '013'
    ft_method = 'mic'

    assert ft_method.lower() in ["mic", "mim", "gamma", 'rel_gamma'], (
        'ft_method should be MIC or MIM or gamma'
    )
    
    MAIN_starttime = time.time()

    # variables (set into cfg.json later)
    tasks = ['rest', 'tap']
    data_version = 'v3.1'
    winLen_sec = 60
    part_winOverlap = 0.5
    mne_format = True
    epochLen_sec = .5
    take_abs = True
    gammaFreq_low=60
    gammaFreq_high=90
    plot_CDRS = True
    plot_ACC = True
    acc_plottype = 'bars'
    plot_task = True

    # create dict with results per task for plotting
    ft_values_per_task, ft_times_per_task, ft_keys = {}, {}, {}

    # ft-method-string to print in filenames
    if take_abs and ft_method.lower() == 'mic':
        print_method = f'abs{ft_method}'
    elif 'gamma' in ft_method:
        print_method = f'{ft_method}{gammaFreq_low}{gammaFreq_high}'
    else:
        print_method = ft_method
    
    if ft_method.lower() in ['mic', 'mim']: ft_code = 'mvc'
    elif 'gamma' in ft_method.lower(): ft_code = ft_method.lower()

    # create directories
    results_sub_dir = join(get_project_path('results'), 'features', ft_code, f'sub{sub}')
    data_sub_dir = join(get_project_path('data'), f'windowed_data_classes_{winLen_sec}s',
                        data_version, f'sub-{sub}')
    ft_figures_dir = join(get_project_path('figures'), 'ft_exploration', data_version, ft_code)

    for f in [results_sub_dir, data_sub_dir, ft_figures_dir]:
        if not exists(f): makedirs(f)

    for task in tasks:

        # check whether results are already present
        mvc_fts_task_file = (f'{ft_code}_fts_{sub}_{print_method}_{task}_{data_version}'
            f'win{winLen_sec}s_overlap{part_winOverlap}.csv'
        )
        if exists(join(results_sub_dir, mvc_fts_task_file)):

            results_df = read_csv(join(results_sub_dir, mvc_fts_task_file), index_col=0)
            ft_values_per_task[task] = results_df.values
            ft_keys[task] = results_df.keys()  # is freqs for mvc, ch_names for gamma
            ft_times_per_task[task] = results_df.index
            print(f'\t...existing mvc-features are loaded for task: {task}')
            # if results for task are loaded, skip remainder of loop for this task
            continue

        # IF RESULTS ARE NOT PRESENT YET, THEY ARE CALCULATED HERE

        # define path names
        windowed_class_path = join(data_sub_dir, task)
        mneEpochs_pickle_fname = (f'{sub}_mneEpochs_{task}_{data_version}_'
                        f'win{winLen_sec}s_overlap{part_winOverlap}_onlyEcogSide')
        windows_pickle_fname = (f'{sub}_windows_{task}_{data_version}_'
                        f'win{winLen_sec}s_overlap{part_winOverlap}_onlyEcogSide')

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
                print(f'DEBUG: {task} before main_loadMergedData')
                # get data object as class
                data = read_data.main_loadMergedData(
                    list_of_subs=[sub,],
                    tasks=task,
                    data_version=data_version,
                    float_convert=True,
                    data_as_df=False,
                    data_as_class=True,
                )  # data consists full unwindowed data
                
                print('...merged data (class) loaded')
                print(f'data is type: {type(data)}')
                # TODO: get rid of sub task subclasses
                mergedData_class = getattr(data, task)
                mergedData_class = getattr(mergedData_class, f'sub{sub}')
                # mergedData_class is type subData_asArrays()
                print(f'DEBUG: shape data in mergedData_class: {mergedData_class.data_arr.shape}')
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
            'features', ft_code,
            f'{sub}_{ft_code}_{ft_method}_report_{task}_{data_version}'
            f'win{winLen_sec}s_overlap{part_winOverlap}.txt'
        )

        # TODO. CREATE IF ELSE FOR MVC METHODS AND GAMMA POWER
        if ft_method in ['mic', 'mim']:
            # import only now because of specific required conda env (mne_mvc)
            from lfpecog_features.feats_multivarConn import run_mne_MVC

            # returns list of mne_connectivity.base.SpectralConnectivity per epoched-window in list
            ft_results = run_mne_MVC(
                mvc_method=ft_method,
                list_mneEpochArrays=list_mneEpochArrays,
                report=True,
                report_path=mvc_report_path,
            )
            # make 2d array of mvc results of several windows (n-windwos x n-coh-freqs)
            ft_values_per_task[task] = np.array([
                ft_results[i].get_data()[0]
                for i in range(len(ft_results))
            ])
            # take absolute imag coh
            if take_abs: ft_values_per_task[task] = abs(ft_values_per_task[task])
            # get mvc-freqs and -times
            ft_keys[task] = ft_results[0].freqs
            ft_times_per_task[task] = class_mne_epochs.window_times
        
        elif 'gamma' in ft_method:
            from lfpecog_features.feats_epoched_gamma import run_epoched_gamma
            (
                ft_values_per_task[task],
                ft_keys[task]  # is ch_names for gamma
            ) = run_epoched_gamma(
                freq_low=gammaFreq_low,
                freq_high=gammaFreq_high,
                list_mneEpochArrays=list_mneEpochArrays,
                ft_method=ft_method,
                report=True,
                report_path=mvc_report_path,
            )
            ft_times_per_task[task] = class_mne_epochs.window_times
        
        
        # save mvc results in csv
        mvc_task_df = DataFrame(
            data=ft_values_per_task[task],
            index=ft_times_per_task[task],
            columns=ft_keys[task])
        mvc_task_df.to_csv(
            join(results_sub_dir, mvc_fts_task_file),
            index=True, header=True, sep=',',
        )


    ### PLOT RESULTS ###
    from lfpecog_plotting.plot_timeFreq_Connectivity import ftPlot_over_dopaTime
    
    # add CDRS or ACC to fname
    if plot_CDRS: print_method += '_cdrs'
    if plot_ACC:
        if acc_plottype == 'fill': print_method += '_accFill'
        elif acc_plottype == 'bars': print_method += '_accBar'
    # process correct tasks in filename and plot-input
    task_fname = '_'
    for t in tasks: task_fname += f'{t}_'

    # merge values and times from different tasks for plot
    # TODO. FIX GAMMA ARRAYS WITH DIFFERENT CHANNELS NAMES
    if len(tasks) == 1:
        mvc_values = ft_values_per_task[task]
        mvc_times = ft_times_per_task[task]
    else:
        mvc_times = np.concatenate(list(ft_times_per_task.values()))
        try:
            mvc_values = np.concatenate(list(ft_values_per_task.values()))
        except ValueError:
            if 'gamma' in ft_method:
                value_dfs = {}
                for n, t in enumerate(tasks):
                    value_dfs[n] = DataFrame(data=ft_values_per_task[t],
                                             columns=ft_keys[t])
                concat_values = value_dfs[0]
                for n in np.arange(1, len(tasks)):
                    concat_values = concat([concat_values, value_dfs[n]])
                ft_keys = concat_values.keys()
                mvc_values = concat_values.values

            else:
                raise ValueError('concatenate fails in main_run_epochFts()')
        # order combined times and values chronologically
        t_order = np.argsort(mvc_times)
        mvc_values = mvc_values[t_order]
        mvc_times = mvc_times[t_order]

    if isinstance(ft_keys, dict): ft_keys = ft_keys[task]  # both are same (in gamma with same ch-names)

    # Run Plotting
    ftPlot_over_dopaTime(
        sub=sub,
        plot_data=mvc_values,
        plot_ft_keys=ft_keys,
        plot_times=mvc_times,
        add_CDRS=plot_CDRS,
        add_ACC=plot_ACC,
        acc_plottype=acc_plottype,
        add_task=plot_task,
        data_version=data_version,
        winLen_sec=winLen_sec,
        fontsize=18,
        ft_method=ft_method,
        cmap='viridis',
        to_save=True,
        save_path=ft_figures_dir,
        fname=(f'{sub}_{ft_code}_{print_method}{task_fname}'
               f'{data_version}_win{winLen_sec}s'
              f'_overlap{part_winOverlap}'),
    )
 

