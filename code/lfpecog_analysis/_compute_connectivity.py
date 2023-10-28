"""Functions for computing multivariate connectivity"""

import numpy as np
from mne_connectivity import spectral_connectivity_time

from _connectivity_helpers import (
    add_missing_patterns_from_indices,
    remove_bads_from_indices,
    remove_nan_data,
)


def compute_connectivity(
    method: str,
    data: np.ndarray,
    indices: tuple[np.ndarray],
    rank: tuple[np.ndarray],
    sfreq: float,
    window_times: list,
    ch_info: dict,
    n_jobs=1,
) -> np.ndarray:
    """Compute multivariate connectivity."""
    accepted_methods = ["mic", "trgc"]
    if method not in accepted_methods:
        raise ValueError(
            f"Method {method} not in accepted methods {accepted_methods}"
        )
    accepted_ch_info = ["ch_name", "ch_type", "ch_hemisphere"]
    if any(key not in accepted_ch_info for key in ch_info.keys()):
        raise ValueError(
            f"Channel information keys {ch_info.keys()} not in accepted keys "
            f"{accepted_ch_info}"
        )

    if method == "trgc":
        mne_method = ["gc", "gc_tr"]
        n_cons = len(indices[0])
        indices = (
            np.concatenate(indices[0], indices[1]),
            np.concatenate(indices[1], indices[0]),
        )
        rank = (
            np.concatenate(rank[0], rank[1]),
            np.concatenate(rank[1], rank[0]),
        )
    else:
        mne_method = method

    connectivity = {
        method: [],
        f"{method}_patterns_seeds": [],
        f"{method}_patterns_targets": [],
        "freqs": None,
        "window_times": window_times,
        "sfreq": sfreq,
    }

    for window_data in data.transpose(1, 0, 2):
        # trim NaN endings
        window_data, empty_chs = remove_nan_data(window_data)
        # remove bad channels from indices
        window_indices = remove_bads_from_indices(indices, empty_chs)

        window_results = spectral_connectivity_time(
            data=window_data[np.newaxis, :, :],
            freqs=np.arange(3, 101),
            method=mne_method,
            indices=window_indices,
            sfreq=sfreq,
            mode="multitaper",
            mt_bandwidth=5.0,
            gc_n_lags=20,
            rank=rank,
            n_jobs=n_jobs,
        )
        if not isinstance(window_results, list):
            window_results = [window_results]

        if method == "trgc":
            gc_st = window_results[0].get_data()[:n_cons]
            gc_ts = window_results[0].get_data()[n_cons:]
            gc_st_tr = window_results[1].get_data()[:n_cons]
            gc_ts_tr = window_results[1].get_data()[n_cons:]
            results = (gc_st - gc_ts) - (gc_st_tr - gc_ts_tr)
            patterns = None
        else:
            results = window_results[0].get_data()[0]
            patterns = np.array(window_results[0].attrs["patterns"])[:, 0]
            patterns = add_missing_patterns_from_indices(
                patterns=patterns,
                current_indices=window_indices,
                new_indices=indices,
            )

        for con_idx in range(results.shape[0]):
            connectivity[method].append(results[con_idx])
            if patterns is not None:
                connectivity[f"{method}_patterns_seeds"].append(
                    patterns[0][con_idx]
                )
                connectivity[f"{method}_patterns_targets"].append(
                    patterns[1][con_idx]
                )

        if connectivity["freqs"] is not None:
            connectivity["freqs"] = window_results[0].freqs
        else:
            if np.array_equal(connectivity["freqs"], window_results[0].freqs):
                raise ValueError(
                    "Frequencies of results for data windows do not match."
                )

    connectivity[method] = np.array(connectivity[method])
    if connectivity[f"{method}_patterns_seeds"] != []:
        connectivity[f"{method}_patterns_seeds"] = np.array(
            connectivity[f"{method}_patterns_seeds"]
        )
        connectivity[f"{method}_patterns_targets"] = np.array(
            connectivity[f"{method}_patterns_targets"]
        )
    else:
        del connectivity[f"{method}_patterns_seeds"]
        del connectivity[f"{method}_patterns_targets"]

    connectivity = add_connectivity_info(
        connectivity=connectivity, indices=indices, ch_info=ch_info
    )


def add_connectivity_info(
    connectivity: dict, indices: tuple[np.ndarray], ch_info: dict
) -> dict:
    """Add optional information to the connectivity results."""
    for info_key in ch_info.keys():
        if info_key == "ch_name":
            connectivity["seed_names"] = [
                [ch_info["ch_name"][ch_idx] for ch_idx in ch_idcs]
                for ch_idcs in indices[0]
            ]
            connectivity["target_names"] = [
                [ch_info["ch_name"][ch_idx] for ch_idx in ch_idcs]
                for ch_idcs in indices[1]
            ]
        elif info_key == "ch_type":
            connectivity["seed_types"] = [
                np.unique(
                    [ch_info["ch_type"][ch_idx] for ch_idx in ch_idcs]
                ).tolist()
                for ch_idcs in indices[0]
            ]
            connectivity["target_types"] = [
                np.unique(
                    [ch_info["ch_type"][ch_idx] for ch_idx in ch_idcs]
                ).tolist()
                for ch_idcs in indices[1]
            ]
        elif info_key == "ch_hemisphere":
            connectivity["seed_hemispheres"] = [
                np.unique(
                    [ch_info["ch_hemisphere"][ch_idx] for ch_idx in ch_idcs]
                ).tolist()
                for ch_idcs in indices[0]
            ]
            connectivity["target_hemispheres"] = [
                np.unique(
                    [ch_info["ch_hemisphere"][ch_idx] for ch_idx in ch_idcs]
                ).tolist()
                for ch_idcs in indices[1]
            ]
            for seed_hemisphere, target_hemisphere in zip(
                connectivity["seed_hemispheres"],
                connectivity["target_hemispheres"],
            ):
                if len(seed_hemisphere) == 1 and len(target_hemisphere) == 1:
                    if seed_hemisphere[0] == target_hemisphere[0]:
                        connectivity["seed_target_hemisphere"] = "ipsilateral"
                    else:
                        connectivity[
                            "seed_target_hemisphere"
                        ] = "contralateral"
                else:
                    connectivity["seed_target_hemisphere"] = "mixed"

    return connectivity
