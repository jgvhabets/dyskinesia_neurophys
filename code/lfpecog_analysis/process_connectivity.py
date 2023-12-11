"""Process cortex-STN multivariate connectivity results"""

import json
import os

import numpy as np
from matplotlib import pyplot as plt

from lfpecog_analysis._connectivity_helpers import (
    load_coordinates,
    process_results,
    plot_results_timefreqs,
    plot_results_patterns,
)


def get_conn_values_sub_side(
    sub, stn_side, conn_method,
    results_df=False, CONN_FT_PATH=None,
    return_times_freqs=False,
):
    
    if isinstance(results_df, bool) and not results_df:
        (results_df,
         window_times,
         freqs) = process_results(
            method=conn_method, subjects=sub,
            results_path=CONN_FT_PATH
        )
        return_times_freqs = True

    # select data on sub and side
    conn_data = [
        results_df.iloc[i][conn_method]
        for i in np.arange(results_df.shape[0])
        if (results_df.iloc[i]['subject'] == sub and
            results_df.iloc[i]['seed_target_lateralisation'] == stn_side)
    ][0]

    if return_times_freqs: return conn_data, window_times, freqs
    else: return conn_data



if __name__ == '__main__':

    # define project path
    PROJECT_PATH = (
        "C:\\Users\\tsbin\\OneDrive - Charité - Universitätsmedizin Berlin\\"
        "PROJECT ECOG-LFP Coherence\\Jeroen_Dyskinesia"
    )
    RESULTS_PATH = os.path.join(
        PROJECT_PATH,
        "results",
        "features",
        "connectivity",
        "windows_10s_0.5overlap",
    )

    # get available subjects
    INFO_PATH = os.path.join(PROJECT_PATH, "data", "meta_info")
    with open(
        os.path.join(INFO_PATH, "ftExtr_spectral_v6.json"), encoding="utf8"
    ) as file:
        subjects = json.load(file)["TOTAL_SUBS"]
    # take only subjects with ECoG & LFP data
    SUBJECTS = [sub for sub in subjects if sub.startswith("0")]

    METHOD = "mic"

    results, window_times, freqs = process_results(
        method=METHOD, subjects=SUBJECTS, results_path=RESULTS_PATH
    )

    fig, axis = plot_results_timefreqs(
        results=results,
        method=METHOD,
        times=window_times,
        freqs=freqs,
        eligible_entries={"seed_target_lateralisation": "ipsilateral"},
        show=False,
    )

    if METHOD == "mic":
        coordinates = load_coordinates(
            os.path.join(INFO_PATH, "ECoG_LFP_coords.csv")
        )
        coordinates["x"] = np.abs(coordinates["x"])
        fig, axis = plot_results_patterns(
            results=results,
            coordinates=coordinates,
            method=METHOD,
            times=window_times,
            freqs=freqs,
            time_range=None,  # (0, 800),
            freq_range=(12, 35),
            eligible_entries={"seed_target_lateralisation": "ipsilateral"},
            show=False,
        )

    plt.show(block=True)

    print("jeff")
