"""Functions for computing multivariate connectivity"""

from copy import deepcopy

import numpy as np
from mne_connectivity import spectral_connectivity_time

from _connectivity_helpers import (
    add_missing_cons_from_indices,
    add_missing_patterns_from_indices,
    remove_bads_from_indices,
    remove_empty_connections,
    remove_nan_data,
    remove_smalls_from_indices,
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

    connectivity = {
        method: [],
        f"{method}_patterns_seeds": [],
        f"{method}_patterns_targets": [],
        "freqs": None,
        "window_times": window_times,
        "sfreq": sfreq,
    }

    n_cons = len(indices[0])
    if method == "trgc":
        mne_method = ["gc", "gc_tr"]
        indices = (
            np.array(
                [*indices[0].tolist(), *indices[1].tolist()], dtype=object
            ),
            np.array(
                [*indices[1].tolist(), *indices[0].tolist()], dtype=object
            ),
        )
        if rank is not None:
            rank = (
                np.concatenate((rank[0], rank[1])),
                np.concatenate((rank[1], rank[0])),
            )
        connectivity[f"{method}_patterns_seeds"] = None
        connectivity[f"{method}_patterns_targets"] = None
    else:
        mne_method = method

    bad_windows = []
    for window_idx, window_data in enumerate(data.transpose(1, 0, 2)):
        print(
            f"\n--- Processing window {window_idx + 1} of {data.shape[1]} ---"
        )
        # trim NaN endings
        window_data, empty_chs = remove_nan_data(window_data)
        # remove bad channels from indices
        window_indices, empty_cons = remove_bads_from_indices(
            indices=indices, bads=empty_chs
        )
        pad_patterns_indices = deepcopy(window_indices)  # to re-add empty chs
        # remove empty connections from indices & rank
        window_indices, window_rank = remove_empty_connections(
            indices=window_indices, rank=rank, empty_cons=empty_cons
        )
        # remove connections with too few channels from indices
        window_indices, window_rank, small_cons = remove_smalls_from_indices(
            indices=window_indices,
            rank=window_rank,
            min_n_chans=window_rank,
        )
        bad_cons = np.unique(empty_cons + small_cons).tolist()

        if len(bad_cons) < n_cons:
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

            results, patterns = _handle_missing_cons(
                window_results=window_results,
                empty_cons=bad_cons,
                n_cons=n_cons,
            )

            if method == "trgc":
                gc_st = np.array(results[0][:n_cons])
                gc_ts = np.array(results[0][n_cons:])
                gc_st_tr = np.array(results[1][:n_cons])
                gc_ts_tr = np.array(results[1][n_cons:])
                results = (gc_st - gc_ts) - (gc_st_tr - gc_ts_tr)
                patterns = None
            else:
                results = np.abs(results)[0]
                patterns = patterns[0]
                patterns = add_missing_patterns_from_indices(
                    patterns=patterns,
                    current_indices=pad_patterns_indices,
                    new_indices=indices,
                )

            connectivity[method].append(results)
            if patterns is not None:
                connectivity[f"{method}_patterns_seeds"].append(patterns[0])
                connectivity[f"{method}_patterns_targets"].append(patterns[1])

            if connectivity["freqs"] is not None:
                connectivity["freqs"] = window_results[0].freqs
            else:
                if np.array_equal(
                    connectivity["freqs"], window_results[0].freqs
                ):
                    raise ValueError(
                        "Frequencies of results for data windows do not match."
                    )
        else:
            bad_windows.append(window_idx)

    connectivity[method] = np.array(connectivity[method]).transpose(1, 0, 2)
    if connectivity[f"{method}_patterns_seeds"] is not None:
        connectivity[f"{method}_patterns_seeds"] = np.array(
            connectivity[f"{method}_patterns_seeds"]
        ).transpose(1, 0, 2, 3)
        connectivity[f"{method}_patterns_targets"] = np.array(
            connectivity[f"{method}_patterns_targets"]
        ).transpose(1, 0, 2, 3)
    else:
        del connectivity[f"{method}_patterns_seeds"]
        del connectivity[f"{method}_patterns_targets"]

    connectivity = _add_connectivity_info(
        connectivity=connectivity, indices=indices, ch_info=ch_info
    )

    connectivity["window_times"] = [
        time for idx, time in enumerate(window_times) if idx not in bad_windows
    ]

    return connectivity


def _handle_missing_cons(
    window_results: list, empty_cons: list[int], n_cons: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Add entries for missing connections for a given window.

    Parameters
    ----------
    window_results : list
        List of MNE-Connectivity results objects for a given window.

    empty_cons : list of int
        Lists of empty connections for this window.

    n_cons : int
        Total number of connections there should be.

    Returns
    -------
    results : list of np.ndarray
        List of connectivity results as arrays.

    patterns : list of np.ndarray
        List of connectivity patterns as arrays.
    """
    results = []
    patterns = []
    for method_results in window_results:
        results_array = method_results.get_data()[0]
        if empty_cons:
            results_array = add_missing_cons_from_indices(
                results=results_array, empty_cons=empty_cons, n_cons=n_cons
            )
        results.append(results_array)

        if method_results.attrs["patterns"] is not None:
            patterns_array = np.array(method_results.attrs["patterns"])[:, 0]
            if empty_cons:
                filled_seeds = add_missing_cons_from_indices(
                    results=patterns_array[0],
                    empty_cons=empty_cons,
                    n_cons=n_cons,
                )
                filled_targets = add_missing_cons_from_indices(
                    results=patterns_array[1],
                    empty_cons=empty_cons,
                    n_cons=n_cons,
                )
                patterns_array = np.concatenate(
                    (filled_seeds[np.newaxis], filled_targets[np.newaxis])
                )
            patterns.append(patterns_array)

    return results, patterns


def _add_connectivity_info(
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
            connectivity["seed_target_lateralisation"] = []
            for seed_hemisphere, target_hemisphere in zip(
                connectivity["seed_hemispheres"],
                connectivity["target_hemispheres"],
            ):
                if len(seed_hemisphere) == 1 and len(target_hemisphere) == 1:
                    if seed_hemisphere[0] == target_hemisphere[0]:
                        connectivity["seed_target_lateralisation"].append(
                            "ipsilateral"
                        )
                    else:
                        connectivity["seed_target_lateralisation"].append(
                            "contralateral"
                        )
                else:
                    connectivity["seed_target_lateralisation"].append("mixed")

    return connectivity
