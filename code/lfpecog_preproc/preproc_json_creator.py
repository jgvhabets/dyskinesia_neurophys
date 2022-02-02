'''
*** PM: Change in notebook to make including different
runs from dir-lists easier. ***

This script/function creates the required JSON-files
to perform the LFP-ECoG-preprocessing as described in
the dyskinesia_neurphys-repo, with the functions in
lfpecog_preproc.

json_creator.py has to be added and run manually for new
settings to create json-files.
JSON-files are called:
    - settings_SETT-VERSION.json:
        containing a list with preprocessing variables
        defining to plot figures or not
        defining which lfp-reref method to use
    - runinfos_DATE_X.json:
        containing a list of dictionaries, each dict
        containing the specific runinfo variables

Both files are written to data/preprocess/preprocess_jsons/

The runinfo dict's are written as a list of dicts which should
be run in one session. These files

See also notebooks for extensive documentation.
Script assumes to be run from cd /.../dyskinesia_neurophys/
'''
import os
import json

# Define json-files to create + if fig's should be plotted
create_settings = True
create_runinfo = False
plot_figures = False
lfp_reref_method = 'levels'  # 'levels' or 'segments'

if __name__ == '__main__':
    if create_settings:
        '''
        MANUAL INPUT FOR PREPROCESS-SETTINGS:
        [win_len, artfct_sd_tresh, bandpass_f, transBW, notchW,
        Fs_origin, Fs_resample, settings_version]
        '''
        sett_version = 'v0.1_Jan22'
        lfp_settings = [1, 4, (1, 120), 10, 2, 4000, 800, sett_version]
        ecog_settings = [1, 4, (1, 120), 10, 2, 4000, 800, sett_version]
    

        # rest of code
        proj_path = os.getcwd()  # projectpath
        data_path = os.path.join(proj_path, 'data')
        json_dir = os.path.join(
            data_path, 'preprocess', 'preprocess_jsons'
        )
        print(f'\nCheck if project_path is correct: {proj_path}\n')
        
        dict_settings = {  # dict to write into json
            'lfp': lfp_settings,
            'ecog': ecog_settings,
            'plot_figs': plot_figures,
            'lfp_reref': lfp_reref_method,
        }

        f = os.path.join(json_dir, f'settings_{sett_version}.json')
        with open(f, 'w') as jsonfile:
            json.dump(dict_settings, jsonfile, indent=4)
    
    if create_runinfo:
        '''
        MANUAL INPUT FOR RUNINFO-SETTINGS:
        [sub, ses, task, acq, run, raw_path, project_path, preproc_sett]

        PM: write combinations manually if more practical,
        or in notebook and import existing runs from directory-lists
        '''
        rawpath = ('/Users/jeroenhabets/OneDrive - Charité - '
                  'Universitätsmedizin Berlin/BIDS_Berlin_ECOG_LFP/rawdata')
        sub = '008'
        ses = 'EphysMedOn01'  # 'EphysMedOn02' / LfpEcogMedOn01
        task = 'Rest'
        acq = 'StimOff'  # 'StimOffLD00'
        run = '01'

        list_dicts = []
        dict_runinfo = {  # dict to write into json
            'sub': sub,
            'ses': ses,  # 'EphysMedOn02'
            'task': task,
            'acq': acq,  # 'StimOffLD00'
            'run': run,
            'raw_path': rawpath,
            'project_path': proj_path,
        }
        list_dicts.append(dict_runinfo)

        acq2 = 'StimOn'
        dict_runinfo = {  # dict to write into json
            'sub': sub,
            'ses': ses,  # 'EphysMedOn02'
            'task': task,
            'acq': acq2,  # 'StimOffLD00'
            'run': run,
            'raw_path': rawpath,
            'project_path': proj_path,
        }
        list_dicts.append(dict_runinfo)

        fname = f'runinfo_{sub}_{ses}_{task}_{acq}_{run}.json'
        fname = 'runinfos_01FEB22.json'
        f = os.path.join(json_dir, fname)
        with open(f, 'w') as jsonfile:
            json.dump(list_dicts, jsonfile, indent=4)


