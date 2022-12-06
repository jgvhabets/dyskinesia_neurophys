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


# import own functions
from utils.utils_fileManagement import (
    get_project_path,
    save_class_pickle,
    load_class_pickle,
)
from lfpecog_features import feats_read_proc_data as read_data
from lfpecog_features.feats_multivarConn import run_mne_MVC

# import mne-mvc functions
# import hackathon_mne_mvc as multiVarConn

if __name__ == '__main__':

    """
    Loads full dataframe per subject, with selected tasks, if
    
    Running on WIN (from repo_folder/code):
        (activate conda environment with custom mne_connectivity)
        python -m lfpecog_features.main_run_epochedConnFts "012"
    """
    sub = sys.argv[1]
    
    MAIN_starttime = time.time()

    # variables (set into cfg.json later)
    tasks = ['rest']
    data_version = 'v3.0'
    winLen_sec = 60
    part_winOverlap = 0.5
    mne_format = True
    epochLen_sec = .5

    for task in tasks:

        # define path names
        windowed_class_path = join(
            get_project_path('data'),
            f'windowed_data_classes_{winLen_sec}s',
            data_version,
            f'sub-{sub}',
            task,
        )
        epoch_dirname = f'list_of_epochs_{int(epochLen_sec * 1000)}ms'
        epoch_path = join(windowed_class_path, epoch_dirname)

        pickled_epochs_path = join(
            windowed_class_path,
            f'{sub}_mneEpochs_{task}_{data_version}.P'
        )
        
        # # define list (final mne-input) here, for if-loop later
        # list_mneEpochArrays = []  # if not defined in beginning, defined at end

        # load mne-Epochs saved as pickle Class
        if exists(pickled_epochs_path):
            print('...loading pickled mneEpochs (from: '
                  f'{pickled_epochs_path})')

            from utils.utils_pickle_mne import pickle_EpochedArrays

            class_mne_epochs = load_class_pickle(pickled_epochs_path)
            list_mneEpochArrays = class_mne_epochs.list_mne_objects

            print(f'...loaded list with {len(list_mneEpochArrays)}'
                  ' mneEpochedArrays is loaded')

        # # load aimed epochs if they exist
        # elif exists(epoch_path):
        #     # TODO: CONSIDER THIS WHOLE epochClass creation in seprate script

        #     print('...load pickled list of epochs')
        #     epoched_windows_list = []

        #     arr_names = [f for f in listdir(epoch_path) if f[-3:] == 'npy']
        #     arr_names = sorted(arr_names)

        #     for f_arr in arr_names:
        #         # print('...load', f_arr)
        #         epochs_win = load(join(epoch_path, f_arr), allow_pickle=True)
        #         epoched_windows_list.append(epochs_win)
            
        #     print(f'...loaded list with {len(epoched_windows_list)} epoched windows')
            
        #     # load fs from stored arrays
        #     hz_string = f_arr.split('Hz')[0]
        #     fs = int(hz_string.split('_')[-1])
        #     # load ch_names of stored arrays
        #     csv_content = []
        #     with open(join(epoch_path, 'ch_names.csv'),) as f:
        #         file = csv.reader(f, delimiter=',')
        #         for r in file:
        #             csv_content.append(r)
        #     ch_names = csv_content[0]


        # if aimed epochs are not existing yet
        else:

            # pickle name of windowed data class
            pickle_fname = f'{sub}_windows_{task}_{data_version}'
            # load data if windowed data class exists
            if exists(windowed_class_path):
                
                if f'{pickle_fname}.P' in listdir(windowed_class_path):

                    print(f'...load existing pickled windowed class for sub-{sub}')
                    starttime = time.time()
                    
                    from utils.utils_windowing import windowedData
                    windows_class = load_class_pickle(
                        join(windowed_class_path, f'{pickle_fname}.P')
                    )
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
                endtime = time.time()

                print(f'\tcreating took {endtime - starttime} seconds')    
                
                print('...included', sub, task)

                # TODO: get rid of sub task subclasses
                
                mergedData_class = getattr(data, task)
                mergedData_class = getattr(mergedData_class, f'sub{sub}')
                
                windows_class = get_windows(
                    data=mergedData_class.data_arr,
                    fs=mergedData_class.fs,
                    winLen_sec=winLen_sec,
                    part_winOverlap=part_winOverlap,
                    col_names=mergedData_class.col_names,
                    return_as_class=True,
                )
                print('...windowed the data into windowedData() Class')

                save_class_pickle(
                    class_to_save=windows_class,
                    path=windowed_class_path,
                    filename=pickle_fname,
                    extension='.P',
                )
            
            # create epochs based on windowed data class (loaded or created)
            makedirs(epoch_path)
            print('...create list of epochs')
            from utils.utils_windowing import window_to_epochs
            
            epoched_windows_list = []
            # define fs and ch_names in variables for mne transform
            fs = windows_class.fs
            ch_names = windows_class.keys

            for n, win in enumerate(windows_class.data):

                epochs = window_to_epochs(
                    win_array=win,
                    fs=windows_class.fs,
                    epochLen_sec=epochLen_sec,
                    remove_nan=True,
                    mne_format=mne_format
                )
                epoched_windows_list.append(epochs)
                
                # # save array as npy file
                # save(
                #     join(
                #         epoch_path,
                #         f'epoched_win{str(n).zfill(3)}_{windows_class.fs}Hz.npy'
                #     ),  # save with three digit padding (for later sorting on name)
                #     epochs
                # )
                
                # with open(join(epoch_path, 'ch_names.csv'), 'w') as f:
                #     write = csv.writer(f)
                #     write.writerow(windows_class.keys)
                #     f.close()
                
                # with open(join(epoch_path, 'window_times.csv'), 'w') as f:
                #     write = csv.writer(f)
                #     write.writerow(windows_class.win_starttimes)
                #     f.close()
        
            # transform epoched array into mne-EpochedArray
            from utils.utils_windowing import create_mne_epochs
            mne_starttime = time.time()
            print(f'start MNE transform with # {len(epoched_windows_list)}')
            list_mneEpochArrays = create_mne_epochs(
                epoched_windows_list,
                fs=fs, ch_names=ch_names,
                pick_only_ephys=True,
            )
            mne_stoptime = time.time()
            print(f'MNE transform took: {mne_stoptime - mne_starttime} seconds')
            
            from utils.utils_pickle_mne import pickle_EpochedArrays
            # add window_starttimes as attr

            class_mne_epochs = pickle_EpochedArrays(
                list_mne_objects = list_mneEpochArrays
            )
            save_class_pickle(
                class_to_save=class_mne_epochs,
                path=windowed_class_path,
                filename=f'{sub}_mneEpochs_{task}_{data_version}',
                extension='.P',
            )

    MAIN_stoptime = time.time()

    print(f'FULL prep-SCRIPT TOOK: {round((MAIN_stoptime - MAIN_starttime) / 60, 1)} minutes')

    mne_starttime = time.time()
    mvc_results = run_mne_MVC(
        list_mneEpochArrays=list_mneEpochArrays
    )
    # return list of mne_connectivity.base.SpectralConnectivity per epoched-window in list
    mne_stoptime = time.time()
    print(f'MNE SCRIPT TOOK: {round((mne_stoptime - mne_starttime) / 60, 1)} minutes')

    print(f'NUMBER OF RESULTS WINDOWS {len(mvc_results)}')



    ### PLOT RESULTS ###
    from lfpecog_plotting.plot_timeFreq_Connectivity import plot_mvc
    # make 2d array of mvc results of several windows
    mvc_results_arr = array([
        mvc_results[i].get_data()[0]
        for i in range(len(mvc_results))
    ])
    # take absolute imag coh
    mvc_results_arr = abs(mvc_results_arr)

    # Run Plotting
    plot_mvc(
        sub=sub,
        plot_data=mvc_results_arr,
        plot_freqs=mvc_results[0].freqs,
        plot_times=windows_class.win_starttimes,
        fs=16,
        cmap='viridis',
        to_save=True,
        save_path=join(get_project_path('figures'),
                    'ft_exploration', 'rest', 'mvc'),
        fname=f'{sub}_mvc_absICOH_{task}_'
    )
 


