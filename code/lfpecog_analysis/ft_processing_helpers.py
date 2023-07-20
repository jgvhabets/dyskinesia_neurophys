"""
Functions to help the post-processing
of features in the process of predicting
dyskinesia scores
"""

# import public functions
import numpy as np
from pandas import DataFrame, concat
from itertools import compress, product

# import own custom functions
from lfpecog_preproc.preproc_import_scores_annotations import(
    get_cdrs_specific, get_ecog_side
)
from lfpecog_analysis.load_SSD_features import ssdFeatures

def find_select_nearest_CDRS_for_ephys(
    sub, ft_times, side: str,
    cdrs_rater: str = 'Patricia',
    EXCL_UNILAT_OTHER_SIDE_LID=False,
):
    """
    Get nearest-CDRS values for neurophys-timings
    or select timestamps of ephys-features that
    are closest to moment where there was only
    Dyskinesia in the NONE-ECoG bodyside

    Input:
        - sub: three-letter sub code
        - ft_times: array with all ft_times to
            be selected in MINUTES
        - side: side of CDRS scores, if only 'left',
            or 'right' given, then this refers to BODYSIDE
        - cdrs_rater: should be: Patricia, Jeroen or Mean

    Returns:
        - select_bool: boolean array with length
            of features, True for rows to keep,
            False for rows to discard
        - ecog_cdrs_for_fts: array with CDRS values
            corresponding to ECoG hemisphere, for
            every timestamp in ft_times
    """
    assert max(ft_times) < 120, 'ft_times should be in MINUTES'
    side = side.lower()
    allowed_sides = ['both', 'all', 'bilat',
                    'right', 'left',
                    'contralat ecog', 'ipsilat ecog',
                    'ecog match', 'ecog nonmatch',
                    'lfp_right', 'lfp_left']
    assert side in allowed_sides, (f'side not in {allowed_sides}')

    # SUM OF BILATERAL CDRS WANTED
    if side in ['both', 'all', 'bilat',]:
        # get timestamps and values of clinical CDRS LID assessment
        cdrs_times, cdrs_scores = get_cdrs_specific(
            sub=sub, rater=cdrs_rater, side='both',)
        
        # find closest CDRS value for ft-values based on timestamps
        cdrs_idx_fts = [np.argmin(abs(t - cdrs_times)) for t in ft_times]
        cdrs_for_fts = cdrs_scores[cdrs_idx_fts]
        # no moments need to be excluded
        select_bool = np.array([True] * len(cdrs_for_fts))

    # UNILATERAL CDRS WANTED
    else:
        # get timestamps and values of clinical CDRS LID assessment
        if side.startswith('lfp'):
            if side == 'lfp_right': side = 'left'
            elif side == 'lfp_left': side = 'right'

            if 'ecog' in side:
                ecogside = get_ecog_side(sub)

                if 'nonmatch' in side or 'ipsilat' in side:
                    if ecogside == 'right': side = 'right'
                    if ecogside == 'left': side = 'left'
                elif side == 'ecog match' or 'contralat' in side:
                    if ecogside == 'right': side = 'left'
                    if ecogside == 'left': side = 'right'
                
        # lid_t_ecog, lid_y_ecog = get_cdrs_specific(
        #     sub=sub, rater=cdrs_rater, side='contralat ecog',)
        # lid_t_nonecog, lid_y_nonecog = get_cdrs_specific(
        #     sub=sub, rater=cdrs_rater, side='ipsilat ecog',)
        
        cdrs_times, cdrs_scores = get_cdrs_specific(sub=sub,
                                                    rater=cdrs_rater,
                                                    side=side,)

        # find closest CDRS value for ft-values based on timestamps
        cdrs_idx_fts = [np.argmin(abs(t - cdrs_times)) for t in ft_times]
        cdrs_for_fts = cdrs_scores[cdrs_idx_fts]
        
        # check whether moments without LID in defined side,
        # but WITH LID ON NON-DEFINED SIDE SHOULD BE EXCLUDED
        if EXCL_UNILAT_OTHER_SIDE_LID:
            if side == 'left': otherside = 'right'
            elif side == 'right': otherside = 'left'
            
            (contra_cdrs_times,
             conta_cdrs_scores) = get_cdrs_specific(sub=sub,
                                                    rater=cdrs_rater,
                                                    side=otherside,)
            contra_cdrs_for_fts = conta_cdrs_scores[cdrs_idx_fts]
            select_bool = ~np.logical_and(cdrs_for_fts == 0,
                                          contra_cdrs_for_fts > 0)
            
        else:
            # no moments need to be excluded
            select_bool = np.array([True] * len(cdrs_for_fts))

    if not isinstance(select_bool, np.ndarray):
        select_bool = select_bool.values
    if not isinstance(cdrs_for_fts, np.ndarray):
        cdrs_for_fts = cdrs_for_fts.values
    
    return select_bool, cdrs_for_fts


def categorical_CDRS(
    y_full_scale,
    time_minutes=None,
    preLID_minutes=10,
    preLID_separate=True,
    cutoff_mildModerate=2.5,
    cutoff_moderateSevere=4.5,
    convert_inBetween_zeros=False,
    return_coding_dict=False
):
    """
    Rescales CDRS scores into
        - None [0]: 0 (excl 10 minutes before LID start),
        - pre- and mild-LID [1]: 10 minutes before LID and up to 2,
            also includes 0-moments after onset of LID
        - moderate [2]: 3 to 5,
        - severe [3]: more than 5.

    Cutoff between mild and severe defaults to 4.5.

    Inputs:
        - y_full_scale: original cdrs scores
        - time_minutes: if needed for pre-LID definition,
            corresponding times in minutes to cdrs values
        - preLID_minutes: number of minutes pre-LID
            to be included in MILD or in SEPERATE CATEGORY
        - preLID_separate: categorize pre-LID in sep category
        - cutoff_mildModerate: defaults to 2.5
        - cutoff_moderateSevere: defaults to 4.5
        - convert_inBetween_zeros: change zero-values in between
            dyskinetic moments to e.g. MILD instead of NONE
        - return_coding_dict: return coding dict
    """
    coding_dict = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
    if preLID_separate:
        for cat in coding_dict.keys():
            if cat == 'none': continue
            coding_dict[cat] = coding_dict[cat] + 1
        coding_dict['pre'] = 1    


    new_y = np.zeros_like(y_full_scale)

    sel_mild = np.logical_and(y_full_scale > 0,
                              y_full_scale < cutoff_mildModerate)
    
    if not preLID_separate and preLID_minutes > 0 and any(y_full_scale > 0):
        # select x-minutes before LID start
        i_start = np.where(y_full_scale)[0][0]
        
        if isinstance(i_start, np.int64) and i_start > 0:
            t_start = time_minutes[i_start]
            sel_pre_min = np.logical_and(
                time_minutes > (t_start - preLID_minutes),
            #     time_minutes < t_start
            # )
            # sel_pre_min = np.logical_and(
            #     sel_pre_min,
                y_full_scale == 0
            )
            sel_mild += sel_pre_min
    
    # convert MILD and pre-LID
    new_y[sel_mild] = coding_dict['mild']
    
    if preLID_separate and any(y_full_scale > 0):
        if preLID_minutes == 0: print(
            'WARNING from categorical CDRS: pre-LID minutes is ZERO'
        )
        new_y[sel_mild] = coding_dict['mild']

        i_start = np.where(y_full_scale)[0][0]        
        t_start = time_minutes[i_start]
        sel_pre = np.logical_and(
            time_minutes > (t_start - preLID_minutes),
            time_minutes < t_start
        )
        new_y[sel_pre] = coding_dict['pre']
    
    if convert_inBetween_zeros and any(y_full_scale > 0):
        i_first_LID = np.where(y_full_scale)[0][0]
        i_last_LID = np.where(y_full_scale)[0][-1]
        sel = np.logical_and(np.arange(len(y_full_scale)) > i_first_LID,
                             np.arange(len(y_full_scale)) < i_last_LID)
        sel = np.logical_and(y_full_scale == 0, sel)

        new_y[sel] = coding_dict[convert_inBetween_zeros]


    sel_moderate = np.logical_and(y_full_scale > cutoff_mildModerate,
                              y_full_scale < cutoff_moderateSevere)
    new_y[sel_moderate] = coding_dict['moderate']

    sel_severe = y_full_scale >= cutoff_moderateSevere
    new_y[sel_severe] = coding_dict['severe']

    if return_coding_dict:
        return new_y, coding_dict
    else:
        return new_y


def load_feature_df_for_pred(
    sub,
    LATERALITY: str,
    INCL_PSD = ['mean_psd',],
    INCL_COH: bool = True,
    INCL_BURSTS: bool = True,
    sel_bandwidths = 'all',
    sel_source = 'all',
    settings_json: str = 'ftExtr_spectral_v1.json',
    preproc_data_version='vX',
    verbose: bool = False,
):
    """
    Inputs:
        - sub
        - LATERALITY: has to be unilat or bilat (all/STN only, or one-sided-ECOG)
        - INCL_PSD: list with 'max_psd' and/or 'mean_psd',
            'variation', 'peak_freq'; False if no powers included
        - INCL_COH: bool, always include both squared and imag
        - INCL_BURSTS_ bool
        - sel_bandwidths: 'all', otherwise selects out features at end
        - sel_source: 'all', otherwise selects out features at end (lfp or ecog)
    """
    # load all features
    fts = ssdFeatures(sub_list=[sub],
                      settings_json=settings_json,
                      data_version=preproc_data_version,
                      incl_bursts=INCL_BURSTS,
                      )
    incl_bws = list(fts.ftExtract_settings['SPECTRAL_BANDS'].keys())
    sub_fts = getattr(fts, f'sub{sub}')  # get sub-specific feature-class

    # create dataframe to store feature in
    features = DataFrame()

    # check bilateral or unilateral feature inclusion
    assert LATERALITY.upper() in ['BILAT', 'UNILAT'], 'LATERALITY should be bilat or unilat'

    if LATERALITY.upper() == 'BILAT':
        SIDE = 'ALL'
    elif LATERALITY.upper() == 'UNILAT':
        if sub.startswith('1'):
            raise ValueError(f'sub-{sub} DOESNOT HAVE ECOG')
        SIDE = get_ecog_side(sub)

    
    # select power-features
    if isinstance(INCL_PSD, list):
        # create bool for sides
        if SIDE == 'ALL': 
            col_side_sel = np.array([True] * sub_fts.powers.shape[1])
        else:
            col_side_sel = np.array([SIDE in k for k in sub_fts.powers.keys()])

        # select on featue type and bw
        ft_sel = np.array([False] * sub_fts.powers.shape[1])
        for f, b in product(INCL_PSD, incl_bws):
            temp_sel = np.array([f in k and b in k
                                 for k in sub_fts.powers.keys()])
            ft_sel += temp_sel  # change every feature-bandwidth combination to True
        
        column_sel = col_side_sel * ft_sel  # take only columns with selected side and ft-band
        power_fts = sub_fts.powers.iloc[:, column_sel]
        
        # if necessary: convert to minutes to agree with CDRS score
        if max(power_fts.index) > 120: power_fts.index = power_fts.index / 60
        
        if verbose: print(f'\tsub-{sub}, POWER FEATS SHAPE INCLUDED: {power_fts.shape}')
        features = concat([features, power_fts], axis=1, ignore_index=False)

    # LOAD COHERENCES
    if INCL_COH:
        if sub.startswith('0'):  # skip STN-only patients
            coh_fts = select_coh_feats(sub_fts=sub_fts, coh_sides='STN_ECOG',
                                    incl_bandws=incl_bws,)
            if verbose: print(f'\tsub-{sub}, ECoG-STN COH FEATS SHAPE INCLUDED: {coh_fts.shape}')
            features = concat([features, coh_fts], axis=1, ignore_index=False)
    
    if INCL_COH and LATERALITY == 'BILAT':
        coh_fts = select_coh_feats(sub_fts=sub_fts, coh_sides='STN_STN',
                                   incl_bandws=incl_bws,)
        if verbose: print(f'\tsub-{sub}, STN-STN COH FEATS SHAPE INCLUDED: {coh_fts.shape}')
        features = concat([features, coh_fts], axis=1, ignore_index=False)


    # INCL BURSTS
    if INCL_BURSTS:
        burst_df = sub_fts.bursts
        if max(burst_df.index) > 120: burst_df.index = burst_df.index / 60

        if SIDE != 'ALL':
            col_sel = [SIDE in col for col in burst_df.keys()]
            burst_df = burst_df.iloc[:, col_sel]
        features = concat([features, burst_df], axis=1, ignore_index=False)
        

    # select specific bandwidths if defined
    if sel_bandwidths != 'all' and sel_bandwidths != ['all']:
        sel_array = np.array([False] * features.shape[1])
        for bw in sel_bandwidths:
            # select based on current bandwidth and add to total selector
            sel = [bw in k for k in features.keys()]
            sel_array += np.array(sel)

        # keep selected columns in feature df
        features = DataFrame(data=features.values[:, sel_array],
                    columns=features.keys()[sel_array],
                    index=features.index)
        if verbose: print(f'\tsub-{sub}, MERGED FEATS SHAPE after {sel_bandwidths} selection: {features.shape}')
    
    # select specific feature sources
    if sel_source != 'all' and sel_source != 'both':
        assert sel_source in ['lfp', 'stn', 'ecog', 'all / both'], 'incorrect source'

        # select based on current bandwidth and add to total selector
        if sel_source.lower() == 'ecog':
            sel_array = np.array(['ecog' in k.lower() and
                                  'coh' not in k for k in features.keys()])
        
        elif sel_source.lower() in ['stn', 'lfp']:
            sel_array = np.array(['ecog' not in k.lower() for k in features.keys()])
       
        # keep selected columns in feature df
        features = DataFrame(data=features.values[:, sel_array],
                    columns=features.keys()[sel_array],
                    index=features.index)
        if verbose: print(f'\tsub-{sub}, MERGED FEATS SHAPE after {sel_source} selection: {features.shape}')
        if verbose: print(f'\tIncluded feats for {sel_source}: {features.keys()}')

    # check incorrect sub-103 timings in v4
    if sub == '103' and max(features.index) > (2e6 / 60):
        corr_times = np.array(features.index) - np.float64(27 * 24 * 60)
        features = DataFrame(data=features.values,
                             index=corr_times,
                             columns=features.keys())
        print('corrected feature timings for sub-013 due to incorrect day')

        
    return features


def select_coh_feats(
    sub_fts, incl_bandws, coh_sides = 'STN_ECOG'
):

    for i_bw, bw in enumerate(incl_bandws):

        for i_coh, coh_type in enumerate(['sq_coh', 'imag_coh']):
            
            coh_sel = getattr(sub_fts.coherences, coh_sides)
            coh_sel = getattr(coh_sel, bw)
            coh_means = getattr(coh_sel, coh_type).mean(axis=1)
            # coh_maxs = getattr(coh_sel, coh_type).max(axis=1)
            temp_index = coh_means.index
            # temp_values = np.concatenate((np.atleast_2d(coh_means.values).T,
            #                               np.atleast_2d(coh_maxs.values).T), axis=1)
            temp_values = coh_means
            coh_labels = [f'{coh_type}_{coh_sides}_{bw}']

            if i_bw == i_coh == 0:
                coh_values = DataFrame(index=temp_index,
                                          data=temp_values,
                                          columns=coh_labels)

            else:
                new_values = DataFrame(index=temp_index,
                                          data=temp_values,
                                          columns=coh_labels)
                
                
                coh_values = concat([coh_values, new_values],
                                    axis=1, ignore_index=False)
                
    # if necessary: convert to minutes to agree with CDRS score
    if max(coh_values.index) > 120: coh_values.index = coh_values.index / 60

    return coh_values


