"""Runs cortex-STN multivariate connectivity analysis"""

import json
import os

from _connectivity_helpers import (
    load_data,
    get_indices_from_features,
    remove_nan_data,
    remove_bads_from_indices,
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
info_fpath = os.path.join(project_path, "data", "meta_info")
with open(
    os.path.join(info_fpath, "ftExtr_spectral_v6.json"), encoding="utf8"
) as file:
    subjects = json.load(file)["TOTAL_SUBS"]
# take only subjects with ECoG & LFP data
subjects = [sub for sub in subjects if sub.startswith("0")]

min_chans = {}
for sub_idx, sub in enumerate(subjects):
    print(f"Processing subject {sub} ({sub_idx + 1} of {len(subjects)})")
    min_chans[sub] = {"seeds": [], "targets": []}
    # load windowed data
    data = load_data(project_path, sub)
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

    for window_idx, window_data in enumerate(data["data"].transpose(1, 0, 2)):
        # find empty channels
        _, empty_chs = remove_nan_data(window_data)
        # remove empty channels from indices
        window_indices, empty_cons = remove_bads_from_indices(
            indices, empty_chs
        )
        # find lowest number of seeds and targets
        min_chans[sub]["seeds"].append([len(con) for con in window_indices[0]])
        min_chans[sub]["targets"].append(
            [len(con) for con in window_indices[1]]
        )
    min_chans[sub]["seeds"] = min_chans[sub]["seeds"]
    min_chans[sub]["targets"] = min_chans[sub]["targets"]

min_chans["all"] = {"seeds": [], "targets": []}
min_chans["all"]["seeds"] = min(
    list(min(min_chans[sub]["seeds"]) for sub in subjects)
)
min_chans["all"]["targets"] = min(
    list(min(min_chans[sub]["targets"]) for sub in subjects)
)

with open(
    os.path.join(info_fpath, "min_n_chans.json"), "w", encoding="utf-8"
) as file:
    json.dump(min_chans, file)
    file.close()
