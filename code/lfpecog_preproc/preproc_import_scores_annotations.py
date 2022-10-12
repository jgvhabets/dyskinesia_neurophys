"""
Functions to import and preprocess
clinical scores and behavioral movements
annotations for the dyskinesia-protocol
"""

# Import public packages and functions
import os
import numpy as np
import pandas as pd
import datetime as dt
from dataclasses import dataclass

# Import own functions


def run_import_clinInfo(
    sub: str,
    onedrive_path: str=(
        '/Users/jeroenhabets/Library/CloudStorage'
        '/OneDrive-Charité-UniversitätsmedizinBerlin/'
    )
):
    """
    Main function to run the import and preprocessing
    of clinical dyskinesia scores and movement
    annotations, per subject

    Input:
        - sub (str): three-number code of sub
        - onedrive_path (str): local-path where Charite-
        OneDrive is synced
    """
    clin_fpath = os.path.join(
        onedrive_path, 'dysk_ecoglfp', 'data'
    )

    annot_dict = read_annotations(sub, clin_fpath)
    scores = read_clinical_scores(sub, clin_fpath,)

    dopaTaps = extract_video_tapTimes(annot_dict)

    return scores, dopaTaps, annot_dict   


def read_annotations(
    sub, clin_fpath,
):
    
    annot_fname = f'sub{sub}_recording_annotations.xlsx'
    
    annot_dict = pd.read_excel(
        os.path.join(clin_fpath, annot_fname),
        sheet_name=None,  # sheet None gives all tabs as dict
        header=None, index_col=0,
    )

    return annot_dict


def read_clinical_scores(
    sub, clin_fpath,
):
    scores_fname = 'dyskinesia_recording_scores.xlsx'
    scores = pd.read_excel(
        os.path.join(clin_fpath, scores_fname),
        sheet_name=f'sub-{sub}'
    )

    # delete redundant rows with text-explanation
    row = 0
    while ~ np.isnan(scores['dopa_time'].iloc[row]):
        row += 1
    scores = scores.iloc[:row]

    return scores


def extract_video_tapTimes(
    annot_dict
):
    tap_times = {'left': [], 'right': []}

    for tab in annot_dict.keys():

        # extract xlsl string-time to datetime
        times = {}
        for event in ['video_start', 'dopa_intake']:
            t = annot_dict[tab].loc[
                f'{event}_time'].reset_index(
                drop=True)[0
            ]
            if t[-1] == "'": t = t[:-1]  # delete redundant "'" froim xlsx-string
            times[event] = dt.datetime.strptime(t, '%d-%m-%Y %H:%M')

        # video-start time relative to dopa-intake
        videoStart_dopaTime = times['video_start'] - times['dopa_intake']
        # print(tab, times)

        if 'tap' in annot_dict[tab].loc[
            'video_task'].reset_index(drop=True)[0
        ]:

            for side in ['left', 'right']:
                # extract seconds after video start
                taptimes = annot_dict[tab
                    ].loc[f'tap_{side}_times'
                ].values.astype(float)  # extract as float-array

                # remove nan's from tap-times
                taptimes = [t for t in taptimes if ~np.isnan(t)]

                # convert to dopatime-related instead of videostart-related
                tap_dopaTimes = [
                    videoStart_dopaTime + dt.timedelta(
                        seconds=t_sec,
                    ) for t_sec in taptimes
                ]
                # convert timedelta into seconds
                tap_dopaSecs = [
                    t.total_seconds() for t in tap_dopaTimes
                ]
                # store all taptimes by side
                tap_times[side].extend(tap_dopaSecs)
            
    return tap_times


def get_seconds_of_LID_start():
    """
    LID-start times updated 06.10.2022

    Returns:
        - lid_times (dict): per sub onne class
            containing sub, seconds of LID-start,
            and seconds of LID-peak
    """
    lt_intakes_hhmm = {
        '008': '11:30',
        '012': '09:54',
        '013': '10:55',
        '014': '09:30'
    }
    # check LID-times in video/with Patricia
    lid_start_hhmm = {
        '008': '11:38',
        '012': '09:55',
        '013': '11:17',
        '014': '09:47'
    }
    lid_peak_hhmm = {
        '008': '12:05',
        '012': '10:24',
        '013': '11:40',
        '014': '10:15'  # TO BE RATED PER 10-minutes
    }
    # ADD options for no LID

    lid_times = {}

    for sub in lid_start_hhmm.keys():

        t_dopa = dt.datetime.strptime(
            lt_intakes_hhmm[sub], '%H:%M'
        )
        t_start = dt.datetime.strptime(
            lid_start_hhmm[sub], '%H:%M'
        )
        t_peak = dt.datetime.strptime(
            lid_peak_hhmm[sub], '%H:%M'
        )

        lid_times[sub] = lid_timing(
            sub=sub,
            t_start=(t_start - t_dopa).seconds,
            t_peak=(t_peak - t_dopa).seconds,
        )

    return lid_times


@dataclass(init=True, repr=True, )
class lid_timing:
    """
    Store timing of first LID and
    peak-LID expressed in seconds
    """
    sub: str
    t_start: float
    t_peak: float

