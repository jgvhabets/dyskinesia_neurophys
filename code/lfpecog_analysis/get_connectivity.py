"""Runs cortex-STN multivariate connectivity analysis"""

import json
import os

import numpy as np

from _compute_connectivity import compute_connectivity
from _connectivity_helpers import (
    load_data,
    get_indices_from_features,
    resample_data,
    save_results,
)


# define project path
project_path = (
    "C:\\Users\\tsbin\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Jeroen_Dyskinesia"
)
results_path = os.path.join(
    project_path,
    "results",
    "features",
    "connectivity",
    "windows_10s_0.5overlap",
)

# get available subjects
info_fpath = os.path.join(
    project_path, "data", "meta_info", "ftExtr_spectral_v6.json"
)
with open(info_fpath, encoding="utf8") as file:
    subjects = json.load(file)["TOTAL_SUBS"]
# take only subjects with ECoG & LFP data
subjects = [sub for sub in subjects if sub.startswith("0")]

METHOD = "mic"

for sub in subjects:
    # load windowed data
    data = load_data(project_path, sub)
    data = resample_data(data, resample_freq=250)
    ch_info = {
        "ch_name": data["ch_name"],
        "ch_type": data["ch_type"],
        "ch_hemisphere": data["ch_hemisphere"],
    }

    # get seed-target indices for connectivity
    indices = get_indices_from_features(
        ch_info=ch_info,
        seed_group={"ch_type": "ecog"},
        target_group={"ch_type": "lfp"},
        split_groups={"ch_hemisphere": data["ch_hemisphere"]},
    )

    # get ranks to project to
    rank = (np.array([4, 4]), np.array([6, 6]))

    # compute connectivity
    connectivity = compute_connectivity(
        method=METHOD,
        data=data["data"],
        sfreq=data["sfreq"],
        indices=indices,
        rank=rank,
        window_times=data["window_times"],
        ch_info=ch_info,
        n_jobs=1,
    )
    save_results(
        os.path.join(results_path, f"sub-{sub}_{METHOD}_ctx-stn.pkl"),
        connectivity,
        {"subject": sub},
    )
