"""Runs cortex-STN multivariate connectivity analysis"""

import json
import os
import sys

import numpy as np

from _compute_connectivity import compute_connectivity
from _connectivity_helpers import (
    load_data,
    get_indices_from_features,
    resample_data,
    save_results,
)


# define project path
project_path = "/fast/work/users/binnsts_c/Analysis/Jeroen_Dyskinesia"
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

SUBJECT = subjects[sys.argv[1]]
METHOD = sys.argv[2]
N_JOBS = sys.argv[3]

# load windowed data
data = load_data(project_path, SUBJECT)
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
rank = (np.array([3, 3]), np.array([6, 6]))

# compute connectivity
connectivity = compute_connectivity(
    method=METHOD,
    data=data["data"],
    sfreq=data["sfreq"],
    indices=indices,
    rank=rank,
    window_times=data["window_times"],
    ch_info=ch_info,
    n_jobs=N_JOBS,
)
save_results(
    os.path.join(results_path, f"sub-{SUBJECT}_{METHOD}_ctx-stn.pkl"),
    connectivity,
    {"subject": SUBJECT},
)
