"""
Functions to help the post-processing
of features in the process of predicting
dyskinesia scores
"""

# import public functions
import numpy as np

# import own custom functions
from lfpecog_preproc.preproc_import_scores_annotations import(
    get_cdrs_specific
)


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