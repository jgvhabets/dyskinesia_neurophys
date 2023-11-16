"""Process cortex-STN multivariate connectivity results"""

import json
import os

import numpy as np
from matplotlib import pyplot as plt

from _connectivity_helpers import (
    load_results,
    results_dict_to_dataframe,
    add_missing_windows,
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
if METHOD == "mic":
    result_keys = [
        METHOD,
        f"{METHOD}_patterns_seeds",
        f"{METHOD}_patterns_targets",
    ]
else:
    result_keys = [METHOD]

results = []
window_times = []
min_time = np.inf
max_time = -np.inf
for subject_idx, subject in enumerate(subjects):
    subject_result = load_results(
        os.path.join(results_path, f"sub-{subject}_{METHOD}_ctx-stn.pkl")
    )
    freqs = np.arange(3, 101)  # results["freqs"]
    window_times.append(subject_result["window_times"])
    if window_times[subject_idx][0] < min_time:
        min_time = window_times[subject_idx][0]
    if window_times[subject_idx][-1] > max_time:
        max_time = window_times[subject_idx][-1]

    results.append(
        results_dict_to_dataframe(
            results=subject_result,
            method_key=METHOD,
            repeat_keys=["subject"],
            discard_keys=["freqs", "window_times", "sfreq"],
        )
    )

for subject_idx, subject_result in enumerate(results):
    subject_result, final_window_times = add_missing_windows(
        results=subject_result,
        window_times=window_times[subject_idx],
        window_intervals=5,
        result_keys=result_keys,
        start_time=min_time,
        end_time=max_time,
    )

    results[subject_idx] = subject_result

results_array = []
for result in results:
    results_array.append(
        result[METHOD][
            list(result["seed_target_lateralisation"].values).index(
                "ipsilateral"
            )
        ]
    )
results_array = np.array(results_array)

fig, axis = plt.subplots(1, 1)
image = axis.imshow(
    np.nanmean(results_array, axis=0).T,
    origin="lower",
    extent=(
        final_window_times[0] / 60,
        final_window_times[-1] / 60,
        freqs[0],
        freqs[-1],
    ),
    aspect="auto",
    cmap="viridis",
)
axis.set_xlabel("Time (minutes)")
axis.set_ylabel("Frequency (Hz)")
fig.subplots_adjust(right=0.85)
cbar_axis = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(image, cax=cbar_axis, label="Connectivity (A.U.)")
plt.show()

print("jeff")
