'''
Python script to run full pre-processing pipeline for
neurphysiology data (ECoG + STN LFP) in once
for one or more specific recording(s) of a patient
(defined by the runsfile (json)).

Instructions to run and keep Git-Sync to repo:
- set work dir: cd /.../dyskinesia_neurophys/
- run file: python3 code/lfpecog_preproc/run_lfpecog_preproc.py

'''

if __name__ == '__main__':
    '''makes sure code is only run when script is called
    prevents code from running when script is
    only imported by another script'''
    
    # Import packages and functions
    import os
    import json

    # Import own functions
    import preproc_data_management as dataMng
    import preproc_artefacts as artefacts
    import preproc_filters as fltrs
    import preproc_resample as resample
    import preproc_reref as reref

    # Import py_neuromodulation functions (for later use)
    # import py_neuromodulation  # toggled inactive to save time

    OSpath = os.getcwd()  # is dyskinesia_neurophys/ (project_folder)
    print(f'\nCheck if project-path is correct: {OSpath}\n')
    data_path = os.path.join(OSpath, 'data')
    json_path = os.path.join(data_path, 'preprocess/preprocess_jsons')
    
    # Load JSON-files with settings and runinfo
    runsfile = os.path.join(json_path, f'runinfos_01FEB22.json')
    settfile = os.path.join(json_path, f'settings_v0.2_Jan22.json')

    with open(os.path.join(json_path, settfile)) as f:
        setting_lists = json.load(f, )  # dict of group-settings
    with open(os.path.join(json_path, runsfile)) as f:
        runs = json.load(f, )  # list of runinfo-dicts

    # Create Settings-DataClass, equal for all runs
    settings = dataMng.Settings(  # order: lfp_left, lfp_right, ecog
        dataMng.PreprocSettings(*setting_lists['lfp']),
        dataMng.PreprocSettings(*setting_lists['lfp']),
        dataMng.PreprocSettings(*setting_lists['ecog']),
    )
    groups = list(settings._fields)

    for Run in runs:
        print(f'\nStart Preprocessing Run: {Run}\n')

        # Create Run-DataClass
        runInfo = dataMng.RunInfo(
            sub=Run['sub'],
            ses=Run['ses'],
            task=Run['task'],
            acq=Run['acq'],
            run=Run['run'],
            raw_path=Run['raw_path'],  # used to import the source-bids-data
            preproc_sett=settings.lfp_left.settings_version,
            project_path=Run['project_path'],  # used to write the created figures and processed data
        )
        rawRun = dataMng.RunRawData(bidspath=runInfo.bidspath)

        # To Plot or Not To Plot Figures (True or False)
        if setting_lists['plot_figs'] == False:
            runInfo.fig_path = None
        # check if fig path is correctly changed
        figcode = setting_lists['plot_figs']
        print(f'\nJSON fig code: {figcode}'
              f'\nFIG-PATH: {runInfo.fig_path}')

        # Load Data
        data = {}
        for field in rawRun.__dataclass_fields__:
            # loops over variables within the data class
            if str(field)[:4] == 'lfp_':
                data[str(field)] = getattr(rawRun, field).load_data()
            elif str(field)[:4] == 'ecog':
                data[str(field)] = getattr(rawRun, field).load_data()
        ch_names = {}
        for group in groups:
            ch_names[group] = data[group].info['ch_names']

        # Artefact Removal
        for group in groups:
            data[group], ch_names[group] = artefacts.artefact_selection(
                bids_dict=data,
                group=group,
                win_len=getattr(settings, group).win_len,
                n_stds_cut=getattr(settings, group).artfct_sd_tresh,
                save=runInfo.fig_path,
            )
        
        # BandPass-Filtering
        for group in groups:
            data[group] = fltrs.bp_filter(
                clean_dict=data,
                group=group,
                sfreq=getattr(settings, group).Fs_orig,
                l_freq=getattr(settings, group).bandpass_f[0],
                h_freq=getattr(settings, group).bandpass_f[1],
                method='iir',
            )

        # Notch-Filters
        for group in groups:
            print(f'Start Notch-Filter GROUP: {group}')
            data[group] = fltrs.notch_filter(
                bp_dict=data,
                group=group,
                transBW=getattr(settings, group).transBW,
                notchW=getattr(settings, group).notchW,
                method='fir',
                ch_names=ch_names,
                save=runInfo.fig_path,
                verbose=False,
            )

        # Resampling
        for group in groups:
            data[group] = resample.resample(
                data=data,
                group=group,
                Fs_orig=getattr(settings, 'ecog').Fs_orig,
                Fs_new = getattr(settings, 'ecog').Fs_resample,
            )

        # Rereferencing
        # deleting possible existing report-file
        if 'reref_report.txt' in os.listdir(
                runInfo.data_path):
            with open(os.path.join(runInfo.data_path,
                    'reref_report.txt'), 'w'):
                pass  # only overwrites, doesn't fill
        for group in groups:
            data[group], ch_names[group] = reref.rereferencing(
                data=data,
                group=group,
                runInfo=runInfo,
                lfp_reref=setting_lists['lfp_reref'],
                ch_names_clean=ch_names,
            )
        
        # Saving Preprocessed Data
        for group in groups:
            dataMng.save_arrays(
                data=data,
                names=ch_names,
                group=group,
                runInfo=runInfo,
            )
            print(f'Preprocessed data saved for {group}!')


