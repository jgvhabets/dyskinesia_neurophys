"""Helper functions for multivariate connectivity analysis"""

import os
import pickle
from copy import deepcopy

import numpy as np
import mne
from mne.filter import resample
from mne_connectivity import seed_target_multivariate_indices
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
import trimesh
from matplotlib import cm, colormaps, colors
from matplotlib.colors import LinearSegmentedColormap


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


def load_coordinates(fpath: str) -> pd.DataFrame:
    """Load channel coordinates.

    Parameters
    ----------
    fpath : str
        Filepath of the coordinates.

    Returns
    -------
    coordinates : pandas.DataFrame
        Channel coordinates as a DataFrame.
    """
    return pd.read_csv(fpath)


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
                ],
                dtype=object,
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

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = list(value)

    return pd.DataFrame.from_dict(results)


def add_missing_windows(
    results: pd.DataFrame,
    window_times: list[int | float],
    window_intervals: int | float,
    result_keys: list[str],
    start_time: int | float | None = None,
    end_time: int | float | None = None,
) -> tuple[pd.DataFrame, list[int | float]]:
    """Add NaN entries between non-consecutive windows.

    Parameters
    ----------
    results : pandas.DataFrame
        Connectivity results.

    window_times : list of int | float
        Times of each window in the connectivity results.

    window_intervals : int | float
        Expected time interval (in units of `window_times`) between each
        window.

    result_keys : list of str
        Keys for the columns in `results` where windows should be added.

    start_time : int | float | None (default None)
        Desired start time of the added windows (in units of `window_times`).
        If < `window_times[0]`, new windows will be added to the start of the
        results. Must be a multiple of `window_intervals`.

    end_time : int | float | None (default None)
        Desired end time of the added windows (in units of `window_times`). If
        > `window_times[-1]`, new windows will be added to the end of the
        results. Must be a multiple of `window_intervals`.

    Returns
    -------
    results : pandas.DataFrame
        Connectivity results with missing windows added.

    window_times : list of int | float
        Times of each window in the new connectivity results.

    Notes
    -----
    If `window_times` has entries [5, 10, 20] and `window_intervals` is 5,
    the missing window at time=15 would be added to the entries of `results`
    given in `result_keys`.
    """
    if len(window_times) == 1:
        return results

    window_time_steps = np.diff(window_times)
    if np.all(window_time_steps == window_intervals):
        return results

    if np.count_nonzero(np.array(window_times) % window_intervals) > 0:
        raise ValueError(
            "`window_times` are expected to be in multiples of "
            "`window_intervals`."
        )

    if start_time is None:
        start_time = window_times[0]
    else:
        if start_time % window_intervals != 0:
            raise ValueError(
                "`start_time` must be a multiple of `window_intervals`."
            )

    if end_time is None:
        end_time = window_times[-1]
    else:
        if end_time % window_intervals != 0:
            raise ValueError(
                "`end_time` must be a multiple of `window_intervals`."
            )

    n_cons = len(results)
    n_windows = int(
        (end_time - start_time + window_intervals) / window_intervals
    )
    new_window_times = np.arange(
        start_time, end_time + window_intervals, window_intervals
    ).tolist()

    for key in result_keys:
        array_to_fill = np.full(
            (n_cons, n_windows, *results[key][0].shape[1:]), fill_value=np.nan
        )
        for con_idx in range(n_cons):
            real_window_idx = 0
            padded_window_idx = 0
            for new_window_time in new_window_times:
                if new_window_time in window_times:
                    array_to_fill[con_idx, padded_window_idx] = results[key][
                        con_idx
                    ][real_window_idx]
                    real_window_idx += 1
                    padded_window_idx += 1
                else:
                    padded_window_idx += 1
        results[key] = list(array_to_fill)

    return results, new_window_times


def process_results(
    method: str, subjects: str | list[str], results_path: str
) -> tuple[pd.DataFrame, list[int | float], list[int | float]]:
    """Process connectivity results.

    Parameters
    ----------
    method : str
        Connectivity method to process results for. Accepts: "mic"; "trgc".

    subjects : str | list of str
        Subjects to process results for. Can be a str for a single subject, or
        a list of str for multiple subjects.

    results_path : str
        Filepath to the folder where results are stored.

    Returns
    -------
    results : pandas.DataFrame
        Connectivity results for all subjects.

    window_times : list of int or float
        Start time of each window in the results.

    freqs : list of int or float
        Frequencies in the results.

    Notes
    -----
    For each subject, the results will be padded with NaN values for
    non-consecutive windows. E.g. if the window times were [5, 10, 20], NaN
    values would be added to the results between the second and third windows
    to mimic a window at time=15.

    If multiple subjects are being processed, the results will be additionally
    padded with NaN values according to the earliest and latest window times
    across all subjects.
    """
    accepted_methods = ["mic", "trgc"]
    if method not in accepted_methods:
        raise ValueError(f"`method` must be one of {accepted_methods}.")

    if method == "mic":
        result_keys = [
            method,
            f"{method}_patterns_seeds",
            f"{method}_patterns_targets",
        ]
    else:
        result_keys = [method]

    if isinstance(subjects, str):
        subjects = [subjects]

    subject_results = []
    subject_window_times = []
    min_window_time = np.inf
    max_window_time = -np.inf
    for subject_idx, subject in enumerate(subjects):
        subject_result = load_results(
            os.path.join(results_path, f"sub-{subject}_{method}_ctx-stn.pkl")
        )

        if subject_idx == 0:
            freqs = subject_result["freqs"]
        else:
            assert np.array_equal(freqs, subject_result["freqs"])

        subject_window_times.append(subject_result["window_times"])
        if subject_window_times[subject_idx][0] < min_window_time:
            min_window_time = subject_window_times[subject_idx][0]
        if subject_window_times[subject_idx][-1] > max_window_time:
            max_window_time = subject_window_times[subject_idx][-1]

        subject_results.append(
            results_dict_to_dataframe(
                results=subject_result,
                method_key=method,
                repeat_keys=["subject"],
                discard_keys=["freqs", "window_times", "sfreq"],
            )
        )

    for subject_idx, subject_result in enumerate(subject_results):
        subject_result, window_times = add_missing_windows(
            results=subject_result,
            window_times=subject_window_times[subject_idx],
            window_intervals=5,
            result_keys=result_keys,
            start_time=min_window_time,
            end_time=max_window_time,
        )

        subject_results[subject_idx] = subject_result

    if len(subject_results) > 1:
        results = pd.concat(subject_results, ignore_index=True)
    else:
        results = subject_results[0]

    return results, window_times, freqs


def _get_eligible_indices(
    results: pd.DataFrame, eligible_entries: dict | None
) -> np.ndarray:
    """Finds indices where a set of conditions are met in a DataFrame.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame to find the eligible entries in.

    eligible_entries : dict | None
        Entries of `results` to plot, where the keys correspond to columns in
        `results`, and the values to the entries of `results[key]` which
        should be plotted. If `None`, all entries are eligible.

    Returns
    -------
    eligible_idcs : numpy.ndarray
        Boolean array of indices (DataFrame rows) where the conditions are met.
    """
    eligible_idcs = np.ones(len(results), dtype=int)

    if eligible_entries is None:
        return eligible_idcs

    for key, value in eligible_entries.items():
        eligible_idcs -= ~(results[key] == value).to_numpy()

    return np.clip(eligible_idcs, 0, 1).astype(bool)


def plot_results_timefreqs(
    results: pd.DataFrame,
    method: str,
    times: list[int | float],
    freqs: list[int | float],
    eligible_entries: dict | None = None,
    show: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot connectivity results in a time-frequency format.

    Parameters
    ----------
    results : pandas.DataFrame
        Results to plot, like that returned from `process_results`.

    method : str
        Connectivity method to plot. Accepts: "mic"; "trgc".

    times : list of int or float
        Times of the results being plotted (in seconds).

    freqs : list of int or float
        Frequencies of the results being plotted (in Hz).

    eligible_entries : dict | None (default None)
        Entries of `results` to plot, where the keys correspond to columns in
        `results`, and the values to the entries of `results[key]` which
        should be plotted. If `None`, all results are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Results figure.

    axis : matplotlib.axes.Axes
        Results axis.

    Notes
    -----
    Whatever results remain after accounting for `eligible_entries` are
    averaged across, leaving a (time x frequency) matrix to be plotted.
    """
    method_mapping = {"mic": "MIC", "trgc": "TRGC"}
    if method not in method_mapping.keys():
        raise ValueError(f"`method` must be one of {method_mapping.keys()}.")

    eligible_idcs = _get_eligible_indices(
        results=results, eligible_entries=eligible_entries
    )
    results_array = np.nanmean(results[method][eligible_idcs].tolist(), axis=0)

    fig, axis = plt.subplots(1, 1)
    image = axis.imshow(
        results_array.T,
        origin="lower",
        extent=(
            times[0] / 60,
            times[-1] / 60,
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

    title = f"Method: {method_mapping[method]}"
    if eligible_entries is not None:
        title = f"{title}\n"
        for key, value in eligible_entries.items():
            title += f"{key} = {value} | "
        title = title[:-3]
    axis.set_title(title)

    if show:
        plt.show()

    return fig, axis


def _gaussianise_data(data: np.ndarray) -> np.ndarray:
    """Gaussianise data to have mean=0 and standard deviation=1.

    Parameters
    ----------
    data : numpy.ndarray
        The data to Gaussianise. Must be a vector.

    Returns
    -------
    data : numpy.ndarray
        The Gaussianised data.

    Notes
    -----
    Follows the approach presented in Van Albada & Robinson (2007). Journal of
    Neuroscience Methods. DOI: 10.1016/j.jneumeth.2006.11.004.
    """
    assert len(data.shape) == 1

    n = np.unique(data, return_inverse=True)[1]
    sorted_n = np.sort(n)
    new_sorted = sorted_n.copy()
    indices = np.argsort(np.argsort(n))

    ties = 0
    for idx, val in enumerate(sorted_n[:-1]):
        if val == sorted_n[idx + 1]:
            ties += 1
        else:
            new_sorted[idx + 1 :] = new_sorted[idx + 1 :] + ties

    rank = new_sorted[indices] + 1

    cdf = rank / len(data) - 1 / (2 * len(data))

    return np.sqrt(2) * sp.special.erfinv(2 * cdf - 1)


def _zscore_data(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to Z-score.

    axis : int (default 0)
        Axis of `data` to Z-score.

    Returns
    -------
    data : numpy.ndarray
        The Z-scored data.
    """
    return (data - np.nanmean(data, axis=axis)) / np.nanstd(data, axis=axis)


def _plot_patterns_ecog(
    data: np.ndarray,
    coordinates: np.ndarray,
    label: str = "",
    colourmap: str | matplotlib.colors.Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    template: str = "mni_icbm152_nlin_asym_09b",
    views: dict | list[dict] | None = None,
    figsize: tuple[float, float] | None = None,
    brain_kwargs: dict | None = None,
    show: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot ECoG connectivity patterns.

    Parameters
    ----------
    data : numpy.ndarray
        Results to plot with shape (channels).

    coordinates : numpy.ndarray
        Coordinates of each channel with shape (channels x 3), where 3
        corresponds to the x-, y-, and z-coordinates, respectively.

    label : str (default "")
        Colourbar label.

    colourmap : str | matplotlib.colors.Colormap (default "viridis")
        Colourmap for the plot.

    vmin : float | None (default None)
        Minimum colourbar value.

    vmax : float | None (default None)
        Maximum colourbar value.

    template : str
        MNE subjects template to use for the plot.

    views : dict | list[dict] | None (default None)
        Viewing parameters for the MNE Brain plot.

    figsize : tuple of float | None (default None)
        Figure size.

    brain_kwargs : dict | None (default None)
        Keyword arguments to pass to the MNE Brain plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Results figure.

    axes : matplotlib.axes.Axes
        Results axes.
    """
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    hemi = "both"

    keys = (str(i) for i in range(coordinates.shape[0]))

    mri_mni_trans = mne.read_talxfm(template, subjects_dir)
    mri_mni_inv = np.linalg.inv(mri_mni_trans["trans"])
    xyz_mri = mne.transforms.apply_trans(mri_mni_inv, coordinates)

    path_mesh = subjects_dir / template / "surf" / f"{template}.glb"
    with open(path_mesh, "rb") as f:
        scene = trimesh.exchange.gltf.load_glb(f)
    mesh: trimesh.Trimesh = trimesh.Trimesh(**scene["geometry"]["geometry_0"])
    xyz_mri = mesh.nearest.on_surface(xyz_mri)[0] * 1.03

    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(keys, xyz_mri, strict=True)), coord_frame="mri"
    )
    info = mne.create_info(
        ch_names=montage.ch_names,
        sfreq=1000,
        ch_types="ecog",
        verbose=None,
    )
    info.set_montage(montage, verbose=False)
    identity = mne.transforms.Transform(fro="head", to="mri", trans=np.eye(4))
    if isinstance(colourmap, str):
        cmap = colormaps[colourmap]
    else:
        cmap = colourmap
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    sensor_colors = mapper.to_rgba(data)
    if brain_kwargs is None:
        brain_kwargs = {
            "surf": "pial",
            "cortex": "low_contrast",
            "alpha": 1.0,
            "background": "white",
        }
    brain = mne.viz.Brain(
        subjects_dir=subjects_dir,
        subject=template,
        hemi=hemi,
        show=False,
        block=False,
        **brain_kwargs,
    )
    brain.add_sensors(
        info,
        trans=identity,
        ecog=True,
        sensor_colors=sensor_colors,
    )

    view_picks = [views]
    view_params = []
    for view in view_picks:
        view_params.append(view)
    if views is None:
        if figsize is None:
            figsize = (6.4, 4.8)
        fig = plt.figure(layout="constrained", figsize=figsize)
        left, right = fig.subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        ax_left = left.add_subplot(111)
        axs_right = right.subplot_mosaic(
            """
            BD
            CD
            """,
            width_ratios=[5, 1],
        )
        axes = [ax_left] + [axs_right[item] for item in ("B", "C")]
        cax = axs_right["D"]
        cbar_kwargs = {
            "ax": cax,
            "fraction": 0.5,
            "shrink": 0.8,
        }
    else:
        if figsize is None:
            figsize = (6.2 * len(view_params) + 0.2, 4.8)
        width_ratios = [1] * len(view_params) + [0.03]
        fig, axes = plt.subplots(
            1,
            len(view_params) + 1,
            width_ratios=width_ratios,
            squeeze=True,
            figsize=figsize,
        )
        axes = (
            [axes] if isinstance(axes, matplotlib.axes.Axes) else axes.tolist()
        )
        cax = axes.pop(-1)
        cbar_kwargs = {"cax": cax}
    for ax, params in zip(axes, view_params, strict=True):
        brain.show_view(**params)
        brain.show()
        im = brain.screenshot(mode="rgb")
        nonwhite_pix = (im != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        im_cropped = im[nonwhite_row][:, nonwhite_col]
        ax.imshow(im_cropped)
        ax.set_axis_off()
    cbar = fig.colorbar(
        mapper,
        location="right",
        **cbar_kwargs,
    )
    cbar.ax.set_ylabel(label)
    cbar.set_ticks((vmin, vmax))
    cbar.set_ticklabels(["Low", "High"])

    if show:
        plt.show(block=True)

    return fig, axes


def plot_results_patterns(
    results: pd.DataFrame,
    coordinates: pd.DataFrame,
    method: str,
    times: list[int | float],
    freqs: list[int | float],
    time_range: tuple | None = None,
    freq_range: tuple | None = None,
    eligible_entries: dict | None = None,
    show: bool = True,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot connectivity patterns.

    Parameters
    ----------
    results : pandas.DataFrame
        Results to plot, like that returned from `process_results`.

    coordinates : pandas.DataFrame
        Coordinates of the channels to plot, like that returned from
        `load_coordinates`.

    method : str
        Connectivity method to plot.

    times : list of int or float
        Times of the results being plotted (in seconds).

    freqs : list of int or float
        Frequencies of the results being plotted (in Hz).

    time_range : tuple | None (default None)
        Time range to average and plot patterns across (in seconds).

    freq_range : tuple | None (default None)
        Frequency range to average and plot patterns across (in Hz).

    eligible_entries : dict | None (default None)
        Entries of `results` to plot, where the keys correspond to columns in
        `results`, and the values to the entries of `results[key]` which
        should be plotted. If `None`, all results are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Results figure.

    axis : matplotlib.axes.Axes
        Results axis.

    Notes
    -----
    Currently only plots ECoG seed patterns.
    """
    eligible_idcs = _get_eligible_indices(
        results=results, eligible_entries=eligible_entries
    )
    results = results.loc[eligible_idcs].reset_index(drop=True)

    seed_types = np.unique(results["seed_types"])
    if len(seed_types) != 1:
        raise ValueError(
            "Currently, only plotting ECoG patterns as seeds is supported."
        )

    if time_range is None:
        time_range = (times[0], times[-1])
    if freq_range is None:
        freq_range = (freqs[0], freqs[-1])

    coordinates[["x", "y", "z"]] /= 1000  # mm to m

    seed_coordinates = []
    seed_patterns = results[f"{method}_patterns_seeds"].tolist()
    missing_patterns = []
    for row_idx, seed_pattern in enumerate(seed_patterns):
        time_start_idx = times.index(time_range[0])
        time_end_idx = times.index(time_range[-1])
        freq_start_idx = freqs.index(freq_range[0])
        freq_end_idx = freqs.index(freq_range[-1])

        seed_pattern = _zscore_data(data=seed_pattern, axis=0)
        seed_pattern = np.nanmean(
            np.abs(
                seed_pattern[
                    time_start_idx : time_end_idx + 1,
                    :,
                    freq_start_idx : freq_end_idx + 1,
                ],
            ),
            axis=(0, 2),
        )

        if np.all(np.isnan(seed_pattern)):
            missing_patterns.append(row_idx)
        else:
            seed_patterns[row_idx] = _gaussianise_data(
                data=seed_pattern[~np.isnan(seed_pattern)]
            )

            subject_coordinate_idcs = coordinates["subject"] == int(
                results["subject"][row_idx]
            )
            subject_coordinates = []
            for seed_name in results["seed_names"][row_idx]:
                subject_coordinates.append(
                    coordinates[["x", "y", "z"]][
                        subject_coordinate_idcs
                        & (coordinates["ch_name"] == seed_name)
                    ].to_numpy()[0]
                )

            n_channels = np.count_nonzero(~np.isnan(seed_patterns[row_idx]))
            assert n_channels == len(subject_coordinates)

            seed_coordinates.append(np.array(subject_coordinates))

    seed_patterns = [
        seed_pattern
        for row_idx, seed_pattern in enumerate(seed_patterns)
        if row_idx not in missing_patterns
    ]
    seed_patterns = np.concatenate(seed_patterns, axis=0)
    seed_coordinates = np.concatenate(seed_coordinates, axis=0)

    colours = plt.get_cmap("Reds")(range(256))
    red_alpha_cmap = LinearSegmentedColormap.from_list(
        name="red_alpha", colors=colours
    )
    height = 4.5 / 2.5
    fig, axis = _plot_patterns_ecog(
        data=seed_patterns,
        coordinates=seed_coordinates,
        label="Connectivity strength (A.U.)",
        colourmap=red_alpha_cmap,
        views={"azimuth": 185.0, "elevation": -50.0},
        figsize=(height * 4 / 3, height),
        brain_kwargs={
            "surf": "pial",
            "cortex": "low_contrast",
            "alpha": 1.0,
            "background": "white",
        },
        show=False,
    )
    title = (
        f"Times: {time_range[0] / 60}-{time_range[1] / 60} minutes | "
        f"Frequencies: {freq_range[0]}-{freq_range[1]} Hz"
    )
    if eligible_entries is not None:
        title = f"{title}\n"
        for key, value in eligible_entries.items():
            title += f"{key} = {value} | "
        title = title[:-3]
    axis[0].set_title(title)

    if show:
        plt.show()

    return fig, axis
