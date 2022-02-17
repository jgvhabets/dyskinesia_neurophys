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

    # TODO: CHANGE ORDER, FIRST REREFERENCING, ON WHOLE SESSIONS FILTERING
    # THEN WINDOWED ARTIFACT REMOVAL (INCL LINEAR TREND)
    # THENR RESAMPLING

    OSpath = os.getcwd()  # is dyskinesia_neurophys/ (project_folder)
    print(f'\nCheck if project-path is correct: {OSpath}\n')
    data_path = os.path.join(OSpath, 'data')
    json_path = os.path.join(data_path, 'preprocess/preprocess_jsons')
    
    # Load JSON-files with settings and runinfo
    # MANUALLY DEFINE TO 2 REQUIRED JSON FILES HERE !!!
    runsfile = os.path.join(json_path, f'runinfos_11FEB22b.json')
    settfile = os.path.join(json_path, f'settings_v0.6_Feb22.json')

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

        # To Plot or Not To Plot (reset fig_path if no plotting)
        if setting_lists['plot_figs'] == False:
            runInfo.fig_path = None

        # Load Data
        data = {}
        for field in rawRun.__dataclass_fields__:
            # loops over variables within the data class
            data[str(field)] = getattr(rawRun, field).load_data()
        ch_names = {}
        for g in groups: ch_names[g] = data[g].info['ch_names']

        # Artefact Removal
        for group in groups:
            data[group], ch_names[group] = artefacts.artefact_selection(
                data_bids=data[group],
                group=group,
                win_len=getattr(settings, group).win_len,
                n_stds_cut=getattr(settings, group).artfct_sd_tresh,
                save=runInfo.fig_path,
                RunInfo=runInfo,
            )

        # Quality check: delete groups without valid channels
        to_del = []
        for group in data.keys():
            if data[group].shape[1] <= 1:
                to_del.append(group)
        for group in to_del:
            del(data[group])
            del(ch_names[group])
            groups.remove(group)
        print(f'\nREMOVED GROUP(s): {to_del}')

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
                Fs_orig=getattr(settings, 'ecog').Fs_orig,
                Fs_new = getattr(settings, 'ecog').Fs_resample,
            )

        # Rereferencing
        # deleting possible existing report-file
        if 'reref_report.txt' in os.listdir(
                runInfo.data_path):
            with open(os.path.join(runInfo.data_path,
                    'reref_report.txt'), 'w') as f:
                # pass  # only overwrites, doesn't fill
                f.write('Groups removed after Artefact Removal '
                        f'due to NO clean channels: {to_del}')

        for group in groups:
            data[group], ch_names[group] = reref.rereferencing(
                data=data[group],
                group=group,
                runInfo=runInfo,
                lfp_reref=setting_lists['lfp_reref'],
                chs_clean=ch_names[group],
            )
        
        # Saving Preprocessed Data
        for group in groups:
            dataMng.save_arrays(
                data=data[group],
                names=ch_names[group],
                group=group,
                runInfo=runInfo,
                lfp_reref=setting_lists['lfp_reref'],
            )
            print(f'Preprocessed data saved for {group}!')


