"""
Run Multivariate Connectivity analysis
via MNE-functions
(https://github.com/vss245/mne-connectivity)

Script has to be ran within active conda-
environment with forked dev mne-connectivity,
without standard mne_connectivity installed
"""
# import public packages and functions
from sys import path
from numpy import where, asarray, isnan
from os.path import join
import json
from numpy.linalg import matrix_rank

# import own functions
from utils.utils_fileManagement import get_project_path

# import forked mne_connectivity (add path to sys before)
codepath = get_project_path('code')
path.append(join(codepath, 'mne-connectivity'))
from mne_connectivity import (
    multivar_seed_target_indices,
    multivar_spectral_connectivity_epochs
)


def run_mne_MVC(
    list_mneEpochArrays,
):
    print('\n\tstart mne mvc function'.upper())

    # set/ extract variables
    ch_names = list_mneEpochArrays[0].info.ch_names
    ecog_side = 'unknown'
    n_name = 0
    while ecog_side == 'unknown':
        if 'ECOG' in ch_names[n_name]:
            ecog_side = ch_names[n_name].split('_')[1]
        else:
            n_name += 1
    # remove non-target/seed indices for mne-functionality
    contra_stn_chnames = [
        n for n in ch_names if f'_{ecog_side}_' not in n
    ]
    for mne_obj in list_mneEpochArrays:
        mne_obj.drop_channels(contra_stn_chnames)
    # update channel names after removal
    ch_names = list_mneEpochArrays[0].info.ch_names

    # define seed and target indices      
    seed_idx = where(
        ['ECOG' in name for name in ch_names]
    )[0]
    target_idx = where(
        [('LFP' in name and ecog_side in name)
        for name in ch_names]
    )[0]


    # load predefined settings
    with open(
        join(
            get_project_path('code'),
            'lfpecog_settings',
            "settings_mne_mvc.json",
        ),
        encoding="utf-8"
    ) as settings_file:
        settings = json.load(settings_file)
    
    # change settings to current data
    settings['seeds'] = [list(seed_idx)]
    settings['targets'] = [list(target_idx)]

    settings['n_seed_components'] = [len(seed_idx)]
    settings['n_target_components'] = [min(
        matrix_rank(
            list_mneEpochArrays[0].get_data()[:, target_idx, :],
            tol=1e-10,
        )
    )]

    print('...settings are set')

    # COMPUTE FEATURES, run mne mvc code
    indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
    
    for n_win, epoch_array in enumerate(list_mneEpochArrays):
        
        print(f'\n...start analysis for window #{n_win}')
        print(f'...array shape: {epoch_array.get_data().shape}')
        # print(f'total ch_names are: {ch_names}')
        # print(f'seeds: {settings["seeds"]}')
        # print(f'targets: {settings["targets"]}')
        # print('...data in array containing NaNs = '
        #     f'{isnan(epoch_array.get_data()).any()}')
        
        win_results = multivar_spectral_connectivity_epochs(
            data=epoch_array,
            indices=indices,
            names=epoch_array.info["ch_names"],
            method=settings["method"],
            mode=settings["mode"],
            tmin=settings["tmin"],
            tmax=settings["tmax"],
            fmin=settings["fmin"],
            fmax=settings["fmax"],
            cwt_freqs=asarray(settings["cwt_freqs"]),
            mt_bandwidth=settings["mt_bandwidth"],
            mt_adaptive=settings["mt_adaptive"],
            mt_low_bias=settings["mt_low_bias"],
            cwt_n_cycles=settings["cwt_n_cycles"],
            n_seed_components=settings["n_seed_components"],
            n_target_components=settings["n_target_components"],
            gc_n_lags=settings["gc_n_lags"],
            n_jobs=settings["n_jobs"],
            block_size=settings["block_size"],
            verbose=settings["verbose"]
        )

        print(type(win_results))
        print(win_results)

