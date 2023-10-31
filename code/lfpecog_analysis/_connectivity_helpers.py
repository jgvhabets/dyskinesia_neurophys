"""Helper functions for multivariate connectivity analysis"""

import os
import pickle
from copy import deepcopy

import numpy as np
from mne.filter import resample
from mne_connectivity import seed_target_multivariate_indices
import pandas as pd


def load_data(project_fpath: str, subject: str) -> dict:
    """Load windowed data of a given subject.

    Parameters
    ----------
    project_fpath : str
        Path to the project folder where the data is stored.

    subject : str
        Subject ID.

    Returns
    -------
    data : dict
        Dictionary containing the windowed data, window times, channel names,
        channel types, channel hemispheres, subject ID, and sampling frequency.
        Data has shape (channels, windows, times).
    """

    data_fpath = os.path.join(
        project_fpath,
        "data",
        "windowed_data_classes_10s_0.5overlap",
        "v4.0",
        f"sub-{subject}",
    )

    data = {
        "data": [],
        "window_times": None,
        "ch_name": [],
        "ch_type": [],
        "ch_hemisphere": [],
        "subject": subject,
        "sfreq": None,
    }

    window_times = None
    sfreq = None
    all_files = os.listdir(data_fpath)
    for file in all_files:
        match_start = f"sub-{subject}_windows_10s_v4.0_"
        if file.startswith(match_start):
            file_end = file[len(match_start) :]
            ch_type = file_end[: file_end.index("_")]
            ch_hemisphere = file_end[
                file_end.index("_") + 1 : file_end.index(".P")
            ]
            with open(os.path.join(data_fpath, file), "rb") as data_file:
                file_data = pickle.load(data_file)
                data_file.close()

            ch_names_idcs = {
                key: idx
                for idx, key in enumerate(file_data.keys)
                if key.startswith("ECOG") or key.startswith("LFP")
            }
            data["data"].extend(
                np.array(
                    file_data.data[:, :, list(ch_names_idcs.values())],
                    dtype=np.float64,
                ).transpose(2, 0, 1)
            )
            data["ch_name"].extend(list(ch_names_idcs.keys()))
            data["ch_type"].extend(
                [ch_type for _ in range(len(ch_names_idcs))]
            )
            data["ch_hemisphere"].extend(
                [ch_hemisphere for _ in range(len(ch_names_idcs))]
            )

            if window_times is None:
                window_times = file_data.win_starttimes
            else:
                if not np.array_equal(window_times, file_data.win_starttimes):
                    raise ValueError(
                        "Window times of data files do not match."
                    )

            if sfreq is None:
                sfreq = file_data.fs
            else:
                if sfreq != file_data.fs:
                    raise ValueError(
                        "Sampling frequency of data files does not match."
                    )

    data["data"] = np.array(data["data"])
    data["window_times"] = window_times
    data["sfreq"] = sfreq

    return data


def get_indices_from_features(
    ch_info: dict, seed_group: dict, target_group: dict, split_groups: dict
) -> tuple[np.ndarray]:
    """Group channels into seeds and targets based on channel features.

    Parameters
    ----------
    ch_info : dict
        Dictionary containing channel features.

    seed_group : dict
        Dictionary containing channel features to be used for seeds.

    target_group : dict
        Dictionary containing channel features to be used for targets.

    split_groups : dict
        Dictionary containing channel features to be used for splitting seeds
        and targets for individual connections.

    Returns
    -------
    indices : tuple[np.ndarray]
        Indices of connections in the MNE-Connectivity format.
    """
    seeds = _get_group_from_features(ch_info, seed_group, split_groups)
    targets = _get_group_from_features(ch_info, target_group, split_groups)

    return seed_target_multivariate_indices(seeds, targets)


def _get_group_from_features(
    ch_info: dict, chs_group: dict, split_group: dict
) -> list[list[int]]:
    """Group channels based on channel features.

    Parameters
    ----------
    ch_info : dict
        Dictionary containing channel features.

    chs_group : dict
        Dictionary containing channel features to be used for grouping.

    split_group : dict
        Dictionary containing channel features to be used for splitting groups
        into individual connections.

    Returns
    -------
    chans : list[list[int]], shape of (connections, channels)
        List of channel indices for each connection.
    """
    chans = []
    eligible_chs = []
    for key, value in chs_group.items():
        eligible_chs.extend(
            [
                idx
                for idx, feature in enumerate(ch_info[key])
                if feature == value
            ]
        )
    eligible_chs = np.unique(eligible_chs)

    for key, value in split_group.items():
        ch_values = np.unique(
            [ch_info[key][idx] for idx in eligible_chs]
        ).tolist()
        for ch_value in ch_values:
            chans.append(
                [idx for idx in eligible_chs if ch_info[key][idx] == ch_value]
            )

    return chans


def remove_nan_data(data: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Remove NaN samples from windowed data.

    Parameters
    ----------
    data : np.ndarray, shape of (channels, times)
        Data for a single window.

    Returns
    -------
    data : np.ndarray, shape of (good channels, good times)
        Data with NaN channels removed, and trimmed according to the first
        NaN sample of any remaining channels.

    empty_chs : list[int]
        Indices of channels that were all NaN.
    """
    if not np.any(np.isnan(data)):
        return data, []

    chs_nan_count = np.isnan(data).sum(axis=1)
    nan_chs = np.where(chs_nan_count > 0)[0]
    empty_chs = np.where(chs_nan_count == data.shape[1])[0]

    if np.array_equal(nan_chs, empty_chs):
        return data, empty_chs.tolist()

    nan_chs = [idx for idx in nan_chs if idx not in empty_chs]
    chs_first_nan = np.where(np.isnan(data[nan_chs]).sum(axis=0) > 0)[0][0]
    return data[:, :chs_first_nan], empty_chs.tolist()


def remove_bads_from_indices(
    indices: tuple[np.ndarray], bads: list[int]
) -> tuple[tuple[np.ndarray], list[int]]:
    """Remove bad channels from indices.

    Parameters
    ----------
    indices : tuple[np.ndarray]
        Indices of connections in the MNE-Connectivity format.

    bads : list[int]
        Indices of bad channels to be removed.

    Returns
    -------
    new_indices : tuple[np.ndarray]
        Indices in the MNE-Connectivity format with bad channels removed.

    empty_cons : list[int]
        Indices of connections that are empty after removing bad channels.
    """
    empty_cons = []
    if bads == []:
        return indices, empty_cons

    new_indices = [[], []]
    for group_idx, group in enumerate(indices):
        for con_idx, con_chs in enumerate(group):
            good_entries = [ch for ch in con_chs if ch not in bads]
            if good_entries == []:
                empty_cons.append(con_idx)
            new_indices[group_idx].append(good_entries)

    new_indices = (
        np.array(new_indices[0], dtype=object),
        np.array(new_indices[1], dtype=object),
    )

    return new_indices, empty_cons


def remove_empty_connections(
    indices: tuple[np.ndarray], rank: tuple[np.ndarray], empty_cons: list[int]
) -> tuple[tuple[np.ndarray], tuple[np.ndarray], list[int]]:
    """Remove bad channels from indices.

    Parameters
    ----------
    indices : tuple[np.ndarray]
        Indices of connections in the MNE-Connectivity format.

    rank : tuple[np.ndarray]
        Ranks of connections in the MNE-Connectivity format.

    empty_cons : list[int]
        Indices of empty connections.

    Returns
    -------
    new_indices : tuple[np.ndarray]
        Indices in the MNE-Connectivity format with empty connections removed.

    new_rank : tuple[np.ndarray]
        Ranks in the MNE-Connectivity format with empty connections removed.
    """
    if empty_cons == []:
        return indices, rank

    new_indices = [[], []]
    new_rank = [[], []]
    if empty_cons != []:
        for group_idx, group in enumerate(indices):
            new_indices[group_idx] = np.array(
                [
                    con
                    for con_idx, con in enumerate(group)
                    if con_idx not in empty_cons
                ]
            )
            new_rank[group_idx] = np.array(
                [
                    rank[group_idx][con_idx]
                    for con_idx in range(len(rank[group_idx]))
                    if con_idx not in empty_cons
                ]
            )
        new_indices = tuple(new_indices)
        new_rank = tuple(new_rank)

    return new_indices, new_rank


def remove_smalls_from_indices(
    indices: tuple[np.ndarray],
    rank: tuple[np.ndarray],
    min_n_chans: tuple[np.ndarray],
) -> tuple[tuple[np.ndarray], tuple[np.ndarray], list[int]]:
    """Remove connections with too few channels from indices.

    Parameters
    ----------
    indices : tuple[np.ndarray]
        Indices of connectivity connections in the MNE-Connectivity format.

    rank : tuple[np.ndarray]
        Ranks of connectivity connections in the MNE-Connectivity format.

    min_n_chans : tuple[np.ndarray]
        Minimum number of channels in the seeds and targets for each
        connection in the format of ranks in MNE-Connectivity.

    Returns
    -------
    new_indices : tuple[np.ndarray]
        Indices in the MNE-Connectivity format with small connections removed.

    new_rank : tuple[np.ndarray]
        Ranks in the MNE-Connectivity format with small connections removed.

    small_cons : list[int]
        Indices of connections with too few channels.
    """
    new_indices = list(deepcopy(indices))
    new_rank = list(deepcopy(rank))
    small_cons = []
    for group_idx, group in enumerate(indices):
        for con_idx, con in enumerate(group):
            if len(con) < min_n_chans[group_idx][con_idx]:
                small_cons.append(con_idx)
    small_cons = np.unique(small_cons).tolist()

    if small_cons != []:
        for group_idx in range(2):
            new_indices[group_idx] = [
                new_indices[group_idx][con_idx]
                for con_idx in range(len(new_indices[group_idx]))
                if con_idx not in small_cons
            ]
            new_rank[group_idx] = np.array(
                [
                    rank[group_idx][con_idx]
                    for con_idx in range(len(rank[group_idx]))
                    if con_idx not in small_cons
                ]
            )

    new_indices = (
        np.array(new_indices[0], dtype=object),
        np.array(new_indices[1], dtype=object),
    )

    return new_indices, new_rank, small_cons


def add_missing_cons_from_indices(
    results: np.ndarray, empty_cons: list[int], n_cons: int
) -> np.ndarray:
    """Add entries for missing connections to connectivity results.

    Parameters
    ----------
    results : np.ndarray, shape (n_cons - n_empty_cons, ...)
        Connectivity results to fill.

    empty_cons : list[int]
        Indices of empty connections that should be filled.

    n_cons : int
        Number of connections that should be present.

    Returns
    -------
    filled_results : np.ndarray, shape (n_cons, ...)
        Connectivity results with empty connections filled by np.nan.
    """
    filled_results = np.full((n_cons, *results.shape[1:]), fill_value=np.nan)
    for con_idx in range(n_cons):
        present_con_idx = 0
        if con_idx not in empty_cons:
            filled_results[con_idx] = results[present_con_idx]
            present_con_idx += 1

    return filled_results


def add_missing_patterns_from_indices(
    patterns: np.ndarray,
    current_indices: tuple[np.ndarray],
    new_indices: tuple[np.ndarray],
) -> np.ndarray:
    """Add entries for missing channels to connectivity patterns.

    Parameters
    ----------
    patterns : np.ndarray, shape (2, connections, channels, frequencies)
        Connectivity patterns to fill missing channels in.

    current_indices : tuple[np.ndarray]
        Indices of channels in the current patterns, in the MNE-Connectivity
        format.

    new_indices : tuple[np.ndarray]
        Indices of channels that the patterns should be filled to, in the
        MNE-Connectivity format.

    Returns
    -------
    new_patterns : np.ndarray, shape (2, connections, channels, frequencies)
        Connectivity patterns with missing channels filled by np.nan.
    """
    n_freqs = patterns.shape[-1]
    new_patterns = [[], []]
    max_n_chans = 0
    for group_idx in range(2):
        if len(current_indices[group_idx]) != len(new_indices[group_idx]):
            raise ValueError(
                "Current and new indices must have the same number of "
                "connections."
            )
        if len(current_indices[group_idx]) != patterns[group_idx].shape[0]:
            raise ValueError(
                "Indices and patterns must have the same number of "
                "connections."
            )
        for con_idx in range(len(current_indices[group_idx])):
            new_entry = []
            current_ch_idx = 0
            new_ch_idx = 0
            while new_ch_idx < len(new_indices[group_idx][con_idx]):
                if (
                    current_ch_idx < len(current_indices[group_idx][con_idx])
                    and new_indices[group_idx][con_idx][new_ch_idx]
                    == current_indices[group_idx][con_idx][current_ch_idx]
                ):
                    new_entry.append(
                        patterns[group_idx, con_idx, current_ch_idx]
                    )
                    current_ch_idx += 1
                    new_ch_idx += 1
                else:
                    new_entry.append(np.full((n_freqs,), fill_value=np.nan))
                    new_ch_idx += 1
            max_n_chans = max(max_n_chans, len(new_entry))
            new_patterns[group_idx].append(np.array(new_entry))

    for group_idx in range(2):
        for con_idx, con_patterns in enumerate(new_patterns[group_idx]):
            n_missing_chans = max_n_chans - len(con_patterns)
            if n_missing_chans > 0:
                new_patterns[group_idx][con_idx] = np.concatenate(
                    (
                        new_patterns[group_idx][con_idx],
                        np.full((n_missing_chans, n_freqs), fill_value=np.nan),
                    )
                )
        new_patterns[group_idx] = np.array(new_patterns[group_idx])

    return np.array(new_patterns)


def resample_data(data: dict, resample_freq: float) -> dict:
    """Resamples windowed data.

    Parameters
    ----------
    data : dict
        Dictionary containing the windowed data (key='data') and sampling
        frequency (key='sfreq'; in Hz). Data should have shape (channels,
        windows, times).

    resample_sfreq : float
        Frequency to resample the data to (in Hz).

    Returns
    -------
    data : dict
        Dictionary containing the resampled windowed data, with the sampling
        frequency updated.

    Notes
    -----
    Resampling is performed with MNE's ``filter.resample`` function on the last
    axis of the data array (should be times), with ``npad='auto'``.
    """
    data["data"] = resample(
        data["data"], down=data["sfreq"] / resample_freq, npad="auto", axis=-1
    )
    data["sfreq"] = resample_freq

    return data


def save_results(fpath: str, results: dict, metadata: dict) -> None:
    """Pickle and save connectivity results.

    Parameters
    ----------
    fpath : str
        Path to save results to.

    results : dict
        Results to save.

    metadata : dict
        Metadata to add to the results before saving.
    """
    results.update(metadata)
    with open(fpath, "wb") as file:
        pickle.dump(results, file)
        file.close()


def load_results(fpath: str) -> dict:
    """Load connectivity results dictionary.

    Parameters
    ----------
    fpath : str
        Path to the results file.

    Returns
    -------
    results : dict
        Connectivity results dictionary.
    """
    with open(fpath, "rb") as file:
        results = pickle.load(file)
        file.close()

    return results


def results_dict_to_dataframe(
    results: dict,
    method_key: str,
    repeat_keys: list[str],
    discard_keys: list[str],
) -> pd.DataFrame:
    """Convert connectivity results dictionary to a pandas DataFrame.

    Parameters
    ----------
    results : dict
        Connectivity results dictionary to convert to a DataFrame.

    method_key : str
        Key of the results dictionary that contains the connectivity results.

    repeat_keys : list[str]
        Keys of the results dictionary that should be repeated for each
        connection.

    discard_keys : list[str]
        Keys of the results dictionary that should be discarded.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing the connectivity results.
    """
    n_cons = len(results[method_key])
    for key in repeat_keys:
        results[key] = [results[key] for _ in range(n_cons)]

    for key in discard_keys:
        del results[key]

    return pd.DataFrame.from_dict(results)
