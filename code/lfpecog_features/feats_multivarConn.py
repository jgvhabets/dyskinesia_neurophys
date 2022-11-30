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
from numpy import where, asarray
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
    print('...start mne mvc function')
    # set/ extract variables
    ch_names = list_mneEpochArrays[0].info.ch_names
    seed_idx = where(
        ['ECOG' in name for name in ch_names]
    )[0]
    ecog_side = ch_names[seed_idx[0]].split('_')[1]
    target_idx = where(
        [('LFP' in name and ecog_side in name)
        for name in ch_names]
    )[0]

    # load predefined settings
    with open(
        join(
            get_project_path('code'),
            'hackathon_mne_mvc',
            "Settings",
            "pipeline_settings_mne_mvc.json",
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

    # actual run code
    indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
    
    for n_win, epoch_array in enumerate(list_mneEpochArrays[:3]):
        
        print(f'...start analysis for window #{n_win}')
        print(f'...array shape: {epoch_array.get_data().shape}')
        
        win_results = multivar_spectral_connectivity_epochs(
            data=epoch_array,
            indices=indices,
            names=epoch_array.info["ch_names"],
            method=settings["method"],
            mode=settings["mode"],
            tmin=settings["tmin"],
            tmax=settings["tmax"],
            fmin=settings["fmt_fmin"],
            fmax=settings["fmt_fmax"],
            cwt_freqs=asarray(settings["cwt_freqs"]),
            mt_bandwidth=settings["mt_bandwidth"],
            mt_adaptive=settings["mt_adaptive"],
            mt_low_bias=settings["mt_low_bias"],
            cwt_n_cycles=settings["cwt_n_cycles"],
            n_seed_components=settings["n_seed_components"],
            n_target_components=settings["n_target_components"],
            gc_n_lags=settings["gc_n_lags"],
            n_jobs=settings["n_jobs"],
            block_size=1000,
            verbose=settings["verbose"]
        )

        print(type(win_results))
        print(win_results)

