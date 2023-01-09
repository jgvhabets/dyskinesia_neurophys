"""
Functions to import and preprocess
clinical scores and behavioral movements
annotations for the dyskinesia-protocol
"""

# Import public packages and functions
import os
import numpy as np
from pandas import read_excel
import datetime as dt
from dataclasses import dataclass
import json

# Import own functions
from utils.utils_fileManagement import get_onedrive_path

def run_import_clinInfo(
    sub: str, verbose=False,
):
    """
    Main function to run the import and preprocessing
    of clinical dyskinesia scores and movement
    annotations, per subject

    Input:
        - sub (str): three-number code of sub
    
    Returns:
        - scores (df)
        - dopa_taps
        - annot_dict
    """
    data_path = get_onedrive_path('data')
    print(data_path)
    
    try:
        annot_dict = read_annotations(sub, data_path)
    except:
        if verbose: print(f'Read ANNOTATIONS failed (sub {sub})')
        annot_dict = None
    
    try:
        scores = read_clinical_scores(sub, data_path)
    except:
        if verbose: print(f'Read CLINICAL SCORES failed (sub {sub})')
        scores = None
    
    try:
        dopa_taps = extract_video_tapTimes(annot_dict)
    except:
        if verbose: print(f'Read DOPA-TAPS failed (sub {sub})')
        dopa_taps = None
    

    return scores, dopa_taps, annot_dict   


def read_annotations(
    sub, data_path,
):
    
    annot_fname = f'sub{sub}_recording_annotations.xlsx'
    
    annot_dict = read_excel(
        os.path.join(data_path, 'clinical scores', annot_fname),
        sheet_name=None,  # sheet None gives all tabs as dict
        header=None, index_col=0,
    )

    return annot_dict


def read_clinical_scores(
    sub, data_path,
):
    scores_fname = 'dyskinesia_recording_scores_Jeroen.xlsx'
    scores = read_excel(
        os.path.join(data_path, 'clinical scores', scores_fname),
        sheet_name=f'sub-{sub}',
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
        - lid_times (dict): per sub one class
            containing sub, seconds of LID-start,
            and seconds of LID-peak
    """
    # lt_intakes_hhmm = {
    #     '008': '11:30',
    #     '009': '10:30',
    #     '010': '11:12',
    #     '011': '16:50',
    #     '012': '09:54',
    #     '013': '10:55',
    #     '014': '09:30',
    #     '016': '13:51'
    # }
    # # check LID-times in video/with Patricia
    # lid_start_hhmm = {
    #     '008': '11:38',
    #     '012': '09:55',
    #     '013': '11:17',
    #     '014': '09:47'
    # }
    # lid_peak_hhmm = {
    #     '008': '12:05',
    #     '012': '10:24',
    #     '013': '11:40',
    #     '014': '10:15'  # TO BE RATED PER 10-minutes
    # }
    od_path = os.path.join(get_onedrive_path('data'), 'clinical scores')
    json_f = os.path.join(od_path, 'med_info.json')

    with open(json_f, 'w') as jsonfile:
        med_info = json.load(json_f,)
    
    LDOPA_intakes_hhmm = med_info['lt_intakes_hhmm']
    LID_start_hhmm = med_info['lid_start_hhmm']
    LID_peak_hhmm = med_info['lid_peak_hhmm']

    lid_times = {}

    for sub in LDOPA_intakes_hhmm.keys():

        t_dopa = dt.datetime.strptime(
            LDOPA_intakes_hhmm[sub], '%H:%M'
        )
        try:
            t_start = dt.datetime.strptime(
                LID_start_hhmm[sub], '%H:%M'
            )
            t_peak = dt.datetime.strptime(
                LID_peak_hhmm[sub], '%H:%M'
            )
            lid_times[sub] = lid_timing(
                sub=sub,
                t_start=(t_start - t_dopa).seconds,
                t_peak=(t_peak - t_dopa).seconds,
            )
        except:  # if LID is not given or not present
            lid_times[sub] = lid_timing(
                sub=sub,
                t_start=None,
                t_peak=None,
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


def get_ecog_side(sub):

    data_path = get_onedrive_path('data')
    f = 'recording_mainfile.xlsx'

    xldat = read_excel(os.path.join(data_path, f),
                      sheet_name='recording_info',)
    sub_match = [type(s) == str and sub in s for s in xldat['bids_id'].values]
    i_sub = np.where(sub_match)[0][0]
    ecog_side = xldat.iloc[i_sub]['ecog']

    if ecog_side == 1: ecog_side = 'left'
    elif ecog_side == 2: ecog_side = 'right'
    else:
        print(f'No ECoG-side found for sub-{sub}')
        return None

    return ecog_side