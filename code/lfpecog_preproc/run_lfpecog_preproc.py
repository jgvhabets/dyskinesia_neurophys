'''
Python script to run full pre-processing pipeline for
neurphysiology and kinematic data (ECoG, STN LFP,
acceleration).

- Call script dfrom command-line including the json-file
    with the preprocessing settings
    (python code/../run_lfpecog_preproc.py /../Settings_vX.json)

- Script runs preprocessing for the recordings (runs)
    of subjects defined in the JSON Setting-File
- Script uses the preprocessing-variables defined in
the 'settings_xxx' JSON-file.
- In subject-specific runInfo, recordings to exclude
    can be defined.
- Report-file (.txt) is created when report_file is
    true in mainSettings-json. If true, .txt is
    generated in dataMng.RunInfo Class and 
    edited during the whole preprocessing process.

Instructions to run and keep Git-Sync to repo:
- set work dir: cd /.../dyskinesia_neurophys/

- run file MacOS: python3 code/lfpecog_preproc/run_lfpecog_preproc.py
    - add json as argument for sys: preprocSettings_v1.1.json

- run file Windows (from cwd ...\code): python -m lfpecog_preproc.run_lfpecog_preproc
    - add json filename as argument for sys, eg: preprocSettings_v1.1.json

'''

if __name__ == '__main__':
    '''makes sure code is only run when script is called
    prevents code from running when script is
    only imported by another script'''

    # Import packages and functions
    import sys
    import json
    from os.path import join

    from utils.utils_fileManagement import get_project_path, get_onedrive_path

    proj_path = get_project_path()
    code_path = get_project_path('code')
    sys.path.append(code_path)

    # Import own functions
    import lfpecog_preproc.preproc_data_management as dataMng
    import lfpecog_preproc.preproc_artefacts as artefacts
    import lfpecog_preproc.preproc_filters as fltrs
    import lfpecog_preproc.preproc_resample as resample
    import lfpecog_preproc.preproc_reref as reref
    import lfpecog_preproc.preproc_get_mne_data as loadData
    import lfpecog_preproc.preproc_plotting as plotting


    # open argument (json file) defined in command (line)
    json_folder = join(get_onedrive_path('data'), 'preprocess_jsons')
    json_path = join(json_folder, sys.argv[1])

    with open(json_path, 'r') as json_data:
    
        mainSettings = json.load(json_data)  # gets dir
    
    for sub in mainSettings['subs_include']:

        sub_runs = dataMng.get_sub_runs(sub)  # get bids info from subject-specific-json and scans.tsv

        for run in list(sub_runs.values()):
        
            if 'dopa' not in run['acq'].lower():
                print(f'\n\tRun {run} SKIPPED, NO "DOPA" IN NAME')
                continue

            print(f'\nSTART PREPROCESSING SUB {sub} Run: {run}\n')

            runInfo = dataMng.RunInfo(
                mainSettings=mainSettings,
                runDict=run,
                project_path=proj_path,
            )
            
            rawRun = dataMng.defineMneRunData(
                runInfo=runInfo,
                subSettings=run,
            )

            dataDict, chNameDict = loadData.get_data_and_channels(
                rawRun=rawRun,
                runInfo=runInfo,
                Fs=rawRun.bids.info['sfreq'],
                to_plot=mainSettings['report_plots'],
                settingsVersion=mainSettings['settingsVersion'],
            )

            dataDict, chNameDict = loadData.remove_flatlines_empties(
                data=dataDict,
                chNames=chNameDict,
                fs=rawRun.bids.info['sfreq'],
                reportPath=runInfo.reportTxt_path,
            )

            dataDict, chNameDict = loadData.delete_empty_groups(
                dataDict, chNameDict
            )

            # BandPass-Filtering
            dataDict = fltrs.filters_for_dict(
                dataDict=dataDict,
                chNamesDict=chNameDict,
                settings=mainSettings,
                Fs=rawRun.bids.info['sfreq'],
                filtertype='bandpass',
            )

            # Notch-Filtering
            dataDict = fltrs.filters_for_dict(
                dataDict=dataDict,
                chNamesDict=chNameDict,
                settings=mainSettings,
                Fs=rawRun.bids.info['sfreq'],
                filtertype='notch',
            )

            # Resampling
            dataDict, Fs_dict = resample.resample_for_dict(
                dataDict=dataDict,
                chNamesDict=chNameDict,
                settings=mainSettings,
                orig_Fs=rawRun.bids.info['sfreq']
            )  # new resampled sample-freqs can vary between datatypes
        
            if mainSettings['report_plots']:
                print('\n\n MAIN SETTINGS TO PLOT POSITIVE')

                plotting.dict_plotting(
                    dataDict=dataDict,
                    Fs_dict=Fs_dict,
                    chNameDict=chNameDict,
                    runInfo=runInfo,
                    moment='pre-artefact-removal',
                    settingsVersion=mainSettings['settingsVersion']
                )

            # Artefact Removal
            dataDict, chNameDict = artefacts.artf_removal_dict(
                dataDict=dataDict,
                Fs_dict=Fs_dict,
                namesDict=chNameDict,
                runInfo=runInfo,
                edge_removal_sec=5,
                settingsVersion=mainSettings['settingsVersion'],
            )

            # Rereferencing with clean, corrected signals
            dataDict, chNameDict = reref.main_rereferencing(
                dataDict=dataDict,
                chNameDict=chNameDict,
                runInfo=runInfo,
                reref_setup=mainSettings['reref_setup'],
                reportPath=runInfo.reportTxt_path,
            )

            if mainSettings['report_plots']:
                plotting.dict_plotting(
                    dataDict=dataDict,
                    Fs_dict=Fs_dict,
                    chNameDict=chNameDict,
                    runInfo=runInfo,
                    moment='post-reref',
                    settingsVersion=mainSettings['settingsVersion']
            )

            # Saving Preprocessed Data
            dataMng.save_dict(
                dataDict=dataDict,
                namesDict=chNameDict,
                FsDict=Fs_dict,
                runInfo=runInfo,
            )
            
            print(f'\nFINISHED PREPROCESSING SUB {sub} Run: {run}\n')

            del(dataDict)
