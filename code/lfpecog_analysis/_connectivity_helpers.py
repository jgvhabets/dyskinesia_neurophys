"""Helper functions for multivariate connectivity analysis"""

import os
import pickle

import numpy as np
from mne.filter import resample
from mne_connectivity import seed_target_multivariate_indices


def load_data(project_fpath: str, subject: str) -> dict:
    """Load windowed data of a given subject."""

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
    """Group channels into seeds and targets based on channel features."""
    seeds = _get_group_from_features(ch_info, seed_group, split_groups)
    targets = _get_group_from_features(ch_info, target_group, split_groups)

    return seed_target_multivariate_indices(seeds, targets)


def _get_group_from_features(
    ch_info: dict, chs_group: dict, split_group: dict
) -> list[list[int]]:
    """Group channels based on channel features."""
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
    """Remove NaN samples from windowed data."""
    if not np.any(np.isnan(data)):
        return data, []

    chs_nan_count = np.isnan(data).sum(axis=1)
    nan_chs = np.where(chs_nan_count > 0)[0]
    empty_chs = np.where(chs_nan_count == data.shape[1])[0]

    if np.all(empty_chs == nan_chs):
        return data, empty_chs.tolist()

    nan_chs = [idx for idx in nan_chs if idx not in empty_chs]
    chs_first_nan = np.where(np.isnan(data[nan_chs]).sum(axis=0) > 0)[0][0]
    return data[:, :chs_first_nan], empty_chs.tolist()


def remove_bads_from_indices(
    indices: tuple[np.ndarray], bads: list[int]
) -> tuple[np.ndarray]:
    """Remove bad channels from indices."""
    if bads == []:
        return indices

    new_indices = [[], []]
    for group_idx, group in enumerate(indices):
        for con_chs in group:
            good_entries = [ch for ch in con_chs if ch not in bads]
            if good_entries == []:
                raise ValueError("There are no valid entries left.")
            new_indices[group_idx].append(good_entries)

    return (
        np.array(new_indices[0], dtype=object),
        np.array(new_indices[1], dtype=object),
    )


def add_missing_patterns_from_indices(
    patterns: np.ndarray,
    current_indices: tuple[np.ndarray],
    new_indices: tuple[np.ndarray],
) -> np.ndarray:
    """Add entries for missing channels to connectivity patterns."""
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
                    new_indices[group_idx][con_idx][new_ch_idx]
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


def resample_data(data: dict, sfreq: float, resample_freq: float) -> dict:
    """Resamples windowed data."""
    data["data"] = resample(
        data["data"], down=sfreq / resample_freq, npad="auto", axis=-1
    )
    data["sfreq"] = resample_freq

    return data
