"""
Main Script to Run multivariate feature Extract (not MIC)

Run on WIN from repo/code:
    python -m lfpecog_features.run_multivar_ftExtr 'ftExtr_spectral_v1.json'
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
from lfpecog_features.feats_ssd import get_subject_SSDs

if __name__ == '__main__':

    json_fname = sys.argv[1]
    ### load settings from given json-file
    settings = load_ft_ext_cfg(json_fname)
    
    # create SSD'd signals per freq-band, per window
    for sub in settings['SUBS_INCL']:
        print(f'\tstart SSD + spectral creation, sub-{sub}...')
        sub_SSD = get_subject_SSDs(sub=sub, settings=settings)
    
        # create local PAC features
         
        print(f'\tstart local-PAC sub-{sub}...')

    # create multi-location connectivity features


        
    


