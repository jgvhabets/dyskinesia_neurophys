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
    import numpy as np

    # Import own functions
    import preproc_data_management as dataMng
    import preproc_artefacts as artefacts
    import preproc_filters as fltrs
    import preproc_resample as resample
    import preproc_reref as reref

    # set cd due to vscode debugger behavior (set to project-folder)
    os.chdir('/Users/jeroenhabets/Research/CHARITE/projects/dyskinesia_neurophys')

    OSpath = os.getcwd()  # is dyskinesia_neurophys/ (project_folder)
    print(f'\nCheck if project-path is correct: {OSpath}\n')
    json_path = os.path.join(OSpath, 'data/preprocess/preprocess_jsons')

    # Load JSON-files with settings and runinfo
    # MANUALLY DEFINE TO 2 REQUIRED JSON FILES HERE !!!
    runsfile = os.path.join(json_path, 'runinfos_11FEB22a.json')  # runinfos_008_medOn2_all
    settfile = os.path.join(json_path, f'settings_v2.1_Feb22.json')

    with open(os.path.join(json_path, settfile)) as f:
        json_settings = json.load(f, )  # dict of group-settings
    with open(os.path.join(json_path, runsfile)) as f:
        runs = json.load(f, )  # list of runinfo-dicts

    settings, groups = dataMng.create_settings_list(json_settings)


    for Run in runs:
        print(f'\nStart Preprocessing Run: {Run}\n')

        # Create Run-DataClass
        runInfo = dataMng.RunInfo(
            sub=Run['sub'],
            ses=Run['ses'],
            task=Run['task'],
            acq=Run['acq'],
            run=Run['run'],
            raw_path=Run['raw_path'],  # to import the source-bids-data
            preproc_sett= getattr(settings, groups[0]).settings_version,
            project_path=Run['project_path'],
        )
        # runrawdata-class defines channels: ECOG / LFP (L / R)
        rawRun = dataMng.RunRawData(bidspath=runInfo.bidspath)

        # To Plot or Not To Plot Figures (True or False)
        if json_settings['plot_figs'] == False:
            runInfo.fig_path = None

        # Create raw mne-object (mne's load_data())
        data = {}
        ch_names = {}  # collects only channel names, no times
        for g in groups:
            data[g] = getattr(rawRun, g).load_data()
            ch_names[g] = ['time'] + data[g].info['ch_names']
            if g[:3] == 'acc': ch_names[g] = [n[:7] for n in ch_names[g]]

        # Actual read in the data (mne's get_data())
        for g in data:
            (ch_arr, ch_t) = data[g].get_data(return_times=True)
            ch_t = np.reshape(ch_t, (1, len(ch_t)))
            data[g] = np.vstack((ch_t, ch_arr))

        # Remove channel with flatlines (based on samples every sec)
        thresh = 0.66  # threshold of fraction flatline in channel
        for g in data:
            flat_chs = []
            for c in np.arange(1, data[g].shape[0]):
                samples = np.arange(0, len(data[g][c]), 4000)
                diffs = np.array([data[g][c][s + 1] -
                            data[g][c][s] for s in samples])
                diff0 = sum(diffs == 0)  # count zeros
                if diff0 / len(diffs) > thresh:
                    flat_chs.append(c)
            if len(flat_chs) > 0:  # if list has content: drop name/ch
                del_names = []
                for f_c in flat_chs: del_names.append(ch_names[g][f_c])
                for c in del_names: ch_names[g].remove(c)
                np.delete(data[g], flat_chs, axis=0)
                print(f'\nFrom {g}, removed: {del_names} '
                      f'due to >{thresh} flatline\n')

        # Delete empty groups
        del_group = []
        for g in data.keys():
            if data[g].shape[1] <= 1: del_group.append(g)
        for group in del_group:
            del(data[group], ch_names[group])
            groups.remove(group)
        print(f'\Empty Group(s) removed: {del_group}')


        # Rereferencing
        # start with deleting existing report-file (if appl)
        reportfile = os.path.join(
            runInfo.data_path, f'reref_report_{groups}.txt')
        if reportfile in os.listdir(
                runInfo.data_path):
            with open(reportfile, 'w') as f:
                # pass  # only overwrites, doesn't fill
                f.write('Empty groups removed after flatline'
                        f'check: {del_group}')
        # Start rereferencing per group
        for g in groups:
            if np.logical_or(g[:4] == 'lfp_', g[:4] == 'ecog'):
                data[g], ch_names[g] = reref.rereferencing(
                    data=data[g],
                    group=g,
                    runInfo=runInfo,
                    lfp_reref=json_settings['lfp_reref'],
                    chs_clean=ch_names[g],
                    reportfile=reportfile,
                )


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
