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
from datetime import date
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

today = date.today()

def run_mne_MVC(
    mvc_method: str,
    list_mneEpochArrays,
    report: bool = False,
    report_path = None,
):
    print('\n\tstart mne mvc function'.upper())
    # create report string to write out at the end
    if report: report = ('\n\n### START OF MNE-MVC '
        f'({today.year} {today.month} {today.day}) ###')

    # set/ extract variables
    ch_names = list_mneEpochArrays[0].info.ch_names
    if report: report += (
        f'\n\n- original ch_names: {ch_names},')

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

    if report: report += (
        f'\n- selected ch_names: {ch_names},'
        f'\n- seed indices: {seed_idx},'
        f'\n- target indices: {target_idx},'
        f'\n- number of windows: {len(list_mneEpochArrays)}'
    )


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
    if report: report += (
        '\n\n- n seed components: '
        f'{settings["n_seed_components"]}'
        '\n\n- n target components (rank):'
        f' {settings["n_target_components"]}'
        f'\n\nMVC method to be calculated: {settings["method"]}'
    )

    print(f'\n\n\tSEED indices: {seed_idx}\n\tTARGET indices: {target_idx}')

    print('...settings are set')

    # COMPUTE FEATURES, run mne mvc code
    indices = multivar_seed_target_indices(settings["seeds"], settings["targets"])
    
    mvc_result_list = []  # store results per window in a list
    for n_win, epoch_array in enumerate(list_mneEpochArrays):
        
        # print(f'\n...start analysis for window #{n_win}')
        # print(f'...array shape: {epoch_array.get_data().shape}')
        
        win_results = multivar_spectral_connectivity_epochs(
            data=epoch_array,
            indices=indices,
            names=epoch_array.info["ch_names"],
            method=mvc_method.lower(),
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
            verbose=False,  # settings["verbose"]
        )
        # returns mne_connectivity.base.SpectralConnectivity
        mvc_result_list.append(win_results)
    
    if report:
        with open(report_path, 'a') as f:

            f.write(report)
            f.close()


    return mvc_result_list

