"""Runs cortex-STN multivariate connectivity analysis"""

import json
import os

from _compute_connectivity import compute_connectivity
from _connectivity_helpers import (
    load_data,
    get_indices_from_features,
    resample_data,
)

# define project path
project_path = (
    "C:\\Users\\tsbin\\OneDrive - Charité - Universitätsmedizin Berlin\\"
    "PROJECT ECOG-LFP Coherence\\Jeroen_Dyskinesia"
)

# get available subjects
info_fpath = os.path.join(
    project_path, "data", "meta_info", "ftExtr_spectral_v6.json"
)
with open(info_fpath, encoding="utf8") as file:
    subjects = json.load(file)["TOTAL_SUBS"]
# take only subjects with ECoG & LFP data
subjects = [sub for sub in subjects if sub.startswith("0")]

for sub in subjects[1:]:
    # load windowed data
    data = load_data(project_path, sub)
    data = resample_data(data, sfreq=data["sfreq"], resample_freq=250)
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

    # compute MIC
    mic = compute_connectivity(
        method="mic",
        data=data["data"],
        sfreq=data["sfreq"],
        indices=indices,
        rank=None,
        window_times=data["window_times"],
        ch_info=ch_info,
        n_jobs=1,
    )

    print("jeff")

    # compute GC
