"""
Functions to import and preprocess
clinical scores and behavioral movements
annotations for the dyskinesia-protocol
"""

# Import public packages and functions
import os
import numpy as np
from pandas import read_excel, isna, DataFrame
import datetime as dt
from dataclasses import dataclass
import json
import traces

# Import own functions
from utils.utils_fileManagement import get_onedrive_path

def run_import_clinInfo(
    sub: str,
    return_CDRS: bool = True,
    return_annotations: bool = False,
    return_tapTimes: bool = False,
    cdrs_rater: str = 'Patricia',
    verbose=False,
):
    """
    Main function to run the import and preprocessing
    of clinical dyskinesia scores and movement
    annotations, per subject

    Input:
        - sub (str): three-number code of sub
    
    Returns:
        - scores (df from 'dyskinesia_recording_scores_Jeroen.xlsx')
        - dopa_taps
        - annot_dict
    """
    data_path = get_onedrive_path('data')

    if return_CDRS + return_annotations + return_tapTimes > 1:
        return_list = True
        list_out = []
    elif return_CDRS + return_annotations + return_tapTimes == 0:
        raise ValueError('at least one variable must be True to return')
    else:
        return_list = False
    
    if return_annotations:
        try:
            annot_dict = read_annotations(sub, data_path=data_path,)
            if return_list: list_out.append(annot_dict)
            else: return annot_dict
        except:
            if verbose: print(f'Read ANNOTATIONS failed (sub {sub})')
            annot_dict = None
    
    if return_CDRS:
        try:
            scores = read_clinical_scores(sub, data_path=data_path,
                                          rater=cdrs_rater)
            # delete Nan Columns from scoring administration
            for c in ['dopa_time_hhmmss', 'video_starttime',
                      'video_name', 'video_time']:
                if c in scores.keys(): del(scores[c])

            if return_list: list_out.append(scores)
            else: return scores
        except:
            if verbose: print(f'Read CLINICAL SCORES failed (sub {sub})')
            scores = None
    
    if return_tapTimes:
        try:
            dopa_taps = extract_video_tapTimes(annot_dict)
            if return_list: list_out.append(dopa_taps)
            else: return dopa_taps
        except:
            if verbose: print(f'Read DOPA-TAPS failed (sub {sub})')
            dopa_taps = None
    
    if return_list: return list_out   


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
    sub, rater='Patricia', data_path=get_onedrive_path('data'),
):
    assert rater.capitalize() in ['Mean', 'Patricia', 'Jeroen'], (
        'insert correct '
    )
    scores_fname = f'Dyskinesia_Ratings_{rater}.xlsx'
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
            if t[-1] == "'": t = t[:-1]  # delete redundant "'" from xlsx-string
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
    sub_match = [type(s) == str and sub in s for s in xldat['study_id'].values]
    i_sub = np.where(sub_match)[0][0]
    ecog_side = xldat.iloc[i_sub]['ecog']

    if ecog_side == 1: ecog_side = 'left'
    elif ecog_side == 2: ecog_side = 'right'
    elif ecog_side == 0: False

    return ecog_side


def get_cdrs_specific(
    sub, side='both', rater='Patricia',
    regularize=False,
):
    """
    Gives uni- or bilateral CDRS scores
    per subject

    Input:
        - sub
        - side: should be left, right, both,
            or 'contra ecog'
        - rater: should be Patricia, Jeroen or Mean
    
    Returns:
        - times
        - scores
    """
    if rater.capitalize() in ['Patricia', 'Jeroen']:
        scores = read_clinical_scores(sub, rater=rater)
        times = scores['dopa_time']

        if side == 'both':
            # take total (INCL AXIAL SCORES)
            scores = scores['CDRS_total']
        
        elif np.logical_and(
            'contra' in side.lower(),
            'ecog' in side.lower()
        ):
            # take CORRESPONDING BODY SIDE to ECOG
            ecogside = get_ecog_side(sub)
            if ecogside == 'left': side = 'right'
            elif ecogside == 'right': side = 'left'
            scores = scores[f'CDRS_total_{side}']
        
        elif np.logical_and(
            'ipsi' in side.lower(),
            'ecog' in side.lower()
        ):
            # take NONE-CORRESPONDING BODY SIDE to ECOG
            side = get_ecog_side(sub)
            scores = scores[f'CDRS_total_{side}']
        
        elif np.logical_or(
            side.lower() == 'left',
            side.lower() == 'right',
        ):
            side = side.lower()
            scores = scores[f'CDRS_total_{side}']
        
        else:
            raise ValueError('side should be "left", '
                    '"right", "both" or "contra ecog"')

        if regularize:
            og_scores = traces.TimeSeries()
            for t, s in zip(times, scores):
                og_scores[t] = s  # add times and scores

            # interpolate to a score every minute
            reg_scores = np.array(og_scores.sample(sampling_period=1,
                                start=int(min(times)), end=int(max(times)),
                                interpolate='linear'))
            nan_sel = isna(reg_scores[:, 1])  # remove None scores
            reg_scores = reg_scores[~nan_sel]

            times = reg_scores[:, 0]
            scores = reg_scores[:, 1]
        
        return times, scores

    elif rater.capitalize() == 'Mean':

        ext_times, ext_scores = {}, {}
        ext_times['J'], ext_scores['J'] = get_cdrs_specific(
            sub=sub, rater='Jeroen', side=side, regularize=True)
        ext_times['P'], ext_scores['P'] = get_cdrs_specific(
            sub=sub, rater='Patricia', side=side, regularize=True)

        min_idx = np.arange(
            min(min(ext_times['P']), min(ext_times['J'])),
            max(max(ext_times['P']), max(ext_times['J'])) + 1
        )

        times_df = DataFrame(index=min_idx, columns=['P', 'J'])

        for r in ['P', 'J']:
            for i, t in enumerate(ext_times[r]):
                times_df[r].loc[i] = ext_scores[r][i]
        
        times = times_df.index.values
        times_nan = isna(times)
        scores_nan = isna(times_df.values)

        scores_nan = [(scores_nan[i, :] == [True, True]).all() for i in np.arange(len(scores_nan))]
        
        sel = np.logical_and(~times_nan, ~np.array(scores_nan))
        times = times[sel]
        scores = times_df.values[sel]
        scores = np.nanmean(scores, axis=1)
    
    else:
        raise ValueError('rater should be: '
                         'Patricia, Jeroen or Mean')

    return times, scores