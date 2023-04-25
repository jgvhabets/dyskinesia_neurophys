"""
Main Script to Run multivariate feature Extract (not MIC)

Run on WIN from repo/code:
    python -m lfpecog_features.run_ssd_ftExtr 'ftExtr_spectral_v1.json'
"""
# import functions
import sys
import json
from os.path import join

# import own functions
from utils.utils_fileManagement import (
    get_onedrive_path,
    load_ft_ext_cfg
)
from lfpecog_features.get_ssd_data import get_subject_SSDs
import lfpecog_features.extract_ssd_features as ssd_feats


if __name__ == '__main__':

    json_fname = sys.argv[1]
    ### load settings from given json-file
    settings = load_ft_ext_cfg(json_fname)
    
    # create SSD'd signals per freq-band, per window
    for sub in settings['SUBS_INCL']:
        print(f'start SSD feature extraction: sub-{sub}...')
        # load or create SSD windows for subject
        sub_SSD = get_subject_SSDs(sub=sub, settings=settings,
                                   incl_ecog=True, incl_stn=True)

        # Extract local spectral power features
        if settings['FEATS_INCL']['to_extract_powers']:
            ssd_feats.extract_local_SSD_powers(
                sub=sub, sub_SSD=sub_SSD, settings_dict=settings
            )
        
        # create local PAC features
        if settings['FEATS_INCL']['to_extract_local_PACs']:
            ssd_feats.extract_local_PACs(
                sub=sub, sub_SSD=sub_SSD, settings_dict=settings
            )

        print(f'\tstart local-PAC sub-{sub}...')

        # extract multi-location connectivity features

        
    


