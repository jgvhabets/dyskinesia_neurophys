"""
Functions to help the post-processing
of features in the process of predicting
dyskinesia scores
"""

# import public functions
import numpy as np
from pandas import DataFrame, concat
from itertools import compress

# import own custom functions
from lfpecog_preproc.preproc_import_scores_annotations import(
    get_cdrs_specific, get_ecog_side
)
from lfpecog_analysis.load_SSD_features import ssdFeatures

def get_idx_discardNonEcogLid(
    sub, ft_times, cdrs_rater: str = 'Patricia'
):
    """
    Select timestamps of epühys-features that
    are closest to moment where there was only
    Dyskinesia in the NONE-ECoG bodyside

    Returns:
        - select_bool: boolean array with length
            of features, True for rows to keep,
            False for rows to discard
        - ecog_cdrs_for_fts: array with CDRS values
            corresponding to ECoG hemisphere, for
            every timestamp in ft_times
    """
    # get timestamps and values of clinical CDRS LID assessment
    lid_t_ecog, lid_y_ecog = get_cdrs_specific(
        sub=sub, rater=cdrs_rater, side='contralat ecog',)
    lid_t_nonecog, lid_y_nonecog = get_cdrs_specific(
        sub=sub, rater=cdrs_rater, side='ipsilat ecog',)
    
    # find closest CDRS value for ft-values based on timestamps
    cdrs_idx_fts = [np.argmin(abs(t - lid_t_ecog)) for t in ft_times]
    ecog_cdrs_for_fts = lid_y_ecog[cdrs_idx_fts]
    nonecog_cdrs_for_fts = lid_y_nonecog[cdrs_idx_fts]

    select_bool = ~np.logical_and(ecog_cdrs_for_fts == 0,
                                  nonecog_cdrs_for_fts > 0)
    
    if not isinstance(select_bool, np.ndarray):
        select_bool = select_bool.values
    if not isinstance(ecog_cdrs_for_fts, np.ndarray):
        ecog_cdrs_for_fts = ecog_cdrs_for_fts.values
    
    return select_bool, ecog_cdrs_for_fts

def load_feature_df_for_pred(
    sub, INCL_POWER: bool, INCL_COH_UNILAT: bool,
    sel_bandwidths = 'all', sel_source = 'all',
    settings_json: str = 'ftExtr_spectral_v1.json',
    verbose: bool = False,
):
    # load all features
    fts = ssdFeatures(sub_list=[sub],
                      settings_json=settings_json,)
    incl_bws = list(fts.ftExtract_settings['SPECTRAL_BANDS'].keys())
    sub_fts = getattr(fts, f'sub{sub}')  # get sub-specific feature-class

    # create dataframe to store feature in
    feat_sel = DataFrame()

    # select ECoG-side power-features
    if INCL_POWER:
        s = get_ecog_side(sub)
        col_sel = [
            (f'ecog_{s}' in c or f'lfp_{s}' in c) and
            ('peak_freq' not in c and 'broad_gamma' not in c)
            for c in list(sub_fts.powers.keys())
        ]
        power_fts = sub_fts.powers.iloc[:, col_sel]
        # if necessary: convert to minutes to agree with CDRS score
        if max(power_fts.index) > 120: power_fts.index = power_fts.index / 60
        
        if verbose: print(f'\tsub-{sub}, POWER FEATS SHAPE INCLUDED: {power_fts.shape}')
        feat_sel = concat([feat_sel, power_fts], axis=1, ignore_index=False)

    # LOAD COHERENCES
    if INCL_COH_UNILAT:
        coh_fts = select_coh_feats(sub_fts=sub_fts, coh_sides='STN_ECOG',
                                   incl_bandws=incl_bws,)
        if verbose: print(f'\tsub-{sub}, COH FEATS SHAPE INCLUDED: {coh_fts.shape}')
        
        feat_sel = concat([feat_sel, coh_fts], axis=1, ignore_index=False)
    
    if verbose: print(f'\tsub-{sub}, MERGED FEATS SHAPE INCLUDED: {feat_sel.shape}')

    # if necessary: convert to minutes to agree with CDRS score
    if max(feat_sel.index) > 120: feat_sel.index = feat_sel.index / 60

    # select specific bandwidths
    if sel_bandwidths != 'all' and sel_bandwidths != ['all']:
        sel_array = np.array([False] * feat_sel.shape[1])
        for bw in sel_bandwidths:
            # select based on current bandwidth and add to total selector
            sel = [bw in k for k in feat_sel.keys()]
            sel_array += np.array(sel)

        # keep selected columns in feature df
        feat_sel = DataFrame(data=feat_sel.values[:, sel_array],
                    columns=feat_sel.keys()[sel_array],
                    index=feat_sel.index)
        if verbose: print(f'\tsub-{sub}, MERGED FEATS SHAPE after {sel_bandwidths} selection: {feat_sel.shape}')
    
    # select specific feature sources
    if sel_source != 'all' and sel_source != 'both':
        assert sel_source in ['lfp', 'ecog'], 'incorrect source'

        # select based on current bandwidth and add to total selector
        sel_array = np.array([sel_source in k for k in feat_sel.keys()])
        # keep selected columns in feature df
        feat_sel = DataFrame(data=feat_sel.values[:, sel_array],
                    columns=feat_sel.keys()[sel_array],
                    index=feat_sel.index)
        if verbose: print(f'\tsub-{sub}, MERGED FEATS SHAPE after {sel_source} selection: {feat_sel.shape}')
        if verbose: print(f'\tIncluded feats for {sel_source}: {feat_sel.keys()}')

    return feat_sel


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
            coh_labels = [f'{coh_type}_{bw}_mn']  # f'{coh_type}_{bw}_mx'  # max coh left out, too similar to mean

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

