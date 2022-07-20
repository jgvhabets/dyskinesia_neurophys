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
    import numpy as np
    from pandas import read_csv

    # Import own functions
    import preproc_data_management as dataMng
    import preproc_artefacts as artefacts
    import preproc_filters as fltrs
    import preproc_resample as resample
    import preproc_reref as reref
    import preproc_get_mne_data as loadData

    
    with open(sys.argv[1], 'r') as json_data:
    
        mainSettings = json.load(json_data)  # gets dir

    proj_path = getcwd()  # should be /../dyskinesia_neurophys
    
    for sub in mainSettings['subs_include'][:3]:

        sub_runs = dataMng.get_sub_runs(
            sub, proj_path,
        )

        for run in sub_runs.values():
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
                rawRun, runInfo
            )  # data array and channel dicts are stored per data-group

            dataDict, chNameDict = loadData.remove_flatlines_empties(
                data=dataDict,
                chNames=chNameDict,
                fs=rawRun.bids.info['sfreq'],
            )

            ### TODO: INCLUDE PREPROCESSING REPORT .TXT
            # per Run (title SubRunVersion# date?)

            # # start with deleting existing report-file (if appl)
            # reportfile = os.path.join(
            #     runInfo.data_path, f'reref_report_{groups}.txt')
            # if reportfile in os.listdir(
            #         runInfo.data_path):
            #     with open(reportfile, 'w') as f:
            #         # pass  # only overwrites, doesn't fill
            #         f.write('Empty groups removed after flatline'
            #                 f'check: {del_group}')

            # Start rereferencing per group
            for group in dataDict:
                
                if np.logical_or(
                    group[:4] == 'lfp_',
                    group[:4] == 'ecog'
                ):

                    dataDict[group],
                    chNameDict[group] = reref.rereferencing(
                        data=dataDict[group],
                        group=group,
                        runInfo=runInfo,
                        lfp_reref=mainSettings['lfp_reref'],
                        chs_clean=chNameDict[group],
                        # reportfile=reportfile,
                    )

            dataDict, chNameDict = loadData.delete_empty_groups(
                dataDict, chNameDict
            )

            ### CONSIDER AUTOMATIC DICT VS ARRAY DETECTION IN
            ## PREPROCESSING FUNCTIONS TO GENERALIZE FUNCTIONS
            # BandPass-Filtering
            for group in groups:
                data[group] = fltrs.bp_filter(
                    data=data[group],
                    sfreq=getattr(settings, group).Fs_orig,
                    l_freq=getattr(settings, group).bandpass_f[0],
                    h_freq=getattr(settings, group).bandpass_f[1],
                    method='iir',
                )


            # Notch-Filters
            for group in groups:
                print(f'Start Notch-Filter GROUP: {group}')
                data[group] = fltrs.notch_filter(
                    data=data[group],
                    ch_names=ch_names[group],
                    group=group,
                    transBW=getattr(settings, group).transBW,
                    notchW=getattr(settings, group).notchW,
                    method='fir',
                    save=runInfo.fig_path,
                    verbose=False,
                    RunInfo=runInfo,
                )


            # Resampling
            for group in groups:
                data[group] = resample.resample(
                    data=data[group],
                    Fs_orig=getattr(settings, group).Fs_orig,
                    Fs_new = getattr(settings, group).Fs_resample,
                )


        # # Artefact Removal
            # for group in groups:
            #     data[group], ch_names[group] = artefacts.artefact_selection(
            #         data=data[group],
            #         group=group,
            #         win_len=getattr(settings, group).win_len,
            #         n_stds_cut=getattr(settings, group).artfct_sd_tresh,
            #         save=runInfo.fig_path,
            #         RunInfo=runInfo,
            #     )


            # Saving Preprocessed Data
            for group in groups:
                dataMng.save_arrays(
                    data=data[group],
                    names=ch_names[group],
                    group=group,
                    runInfo=runInfo,
                    lfp_reref=json_settings['lfp_reref'],
                )
                print(f'Preprocessed data saved for {group}!')
