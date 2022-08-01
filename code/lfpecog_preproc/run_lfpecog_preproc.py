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
- run file: python3 code/lfpecog_preproc/run_lfpecog_preproc.py
    - add json as argument for sys: data/preprocess.../xxx.json
'''

if __name__ == '__main__':
    '''makes sure code is only run when script is called
    prevents code from running when script is
    only imported by another script'''

    # Import packages and functions
    from os.path import join
    from os import getcwd
    import sys
    import json

    proj_path = getcwd()  # should be /../dyskinesia_neurophys

    ft_path = join(proj_path, 'code/lfpecog_features')
    sys.path.append(ft_path)

    # Import own functions
    import preproc_data_management as dataMng
    import preproc_artefacts as artefacts
    import preproc_filters as fltrs
    import preproc_resample as resample
    import preproc_reref as reref
    import preproc_get_mne_data as loadData
    import preproc_plotting as plotting


    # open argument (json file) defined in command (line)
    with open(sys.argv[1], 'r') as json_data:
    
        mainSettings = json.load(json_data)  # gets dir

    
    
    for sub in mainSettings['subs_include']:

        sub_runs = dataMng.get_sub_runs(
            sub, proj_path,
        )

        for run in list(sub_runs.values())[3:5]:
        
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
                rawRun=rawRun, runInfo=runInfo,
                Fs=mainSettings['ephys']['orig_Fs'],
                to_plot=mainSettings['report_plots'],
            )

            dataDict, chNameDict = loadData.remove_flatlines_empties(
                data=dataDict,
                chNames=chNameDict,
                fs=rawRun.bids.info['sfreq'],
                reportPath=runInfo.reportTxt_path,
            )

            dataDict, chNameDict = reref.main_rereferencing(
                dataDict=dataDict,
                chNameDict=chNameDict,
                runInfo=runInfo,
                lfp_reref=mainSettings['lfp_reref'],
                reportPath=runInfo.reportTxt_path,
            )

            dataDict, chNameDict = loadData.delete_empty_groups(
                dataDict, chNameDict
            )

            # BandPass-Filtering
            dataDict = fltrs.filters_for_dict(
                dataDict, chNameDict, mainSettings,
                'bandpass',
            )

            # Notch-Filtering
            dataDict = fltrs.filters_for_dict(
                dataDict, chNameDict, mainSettings,
                'notch'
            )

            # Resampling
            dataDict, Fs_dict = resample.resample_for_dict(
                dataDict, chNameDict, mainSettings
            )
        
            if mainSettings['report_plots']:

                plotting.dict_plotting(
                    dataDict=dataDict,
                    Fs_dict=Fs_dict,
                    chNameDict=chNameDict,
                    runInfo=runInfo,
                    moment='post-preprocess',
                )

            # Artefact Removal
            dataDict, chNameDict = artefacts.artf_removal_dict(
                dataDict=dataDict,
                namesDict=chNameDict,
                runInfo=runInfo,
            )


            # Saving Preprocessed Data
            dataMng.save_dict(
                dataDict=dataDict,
                namesDict=chNameDict,
                runInfo=runInfo,
            )
            
            print(f'\nFINISHED PREPROCESSING SUB {sub} Run: {run}\n')
