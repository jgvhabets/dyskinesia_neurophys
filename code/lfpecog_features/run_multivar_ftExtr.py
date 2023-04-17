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
from lfpecog_features.feats_extract_multivar import create_SSDs_and_spectral_features
from utils.utils_fileManagement import get_onedrive_path


if __name__ == '__main__':

    json_fname = sys.argv[1]
    ### load settings from json
    json_path = join(get_onedrive_path('data'),
                     'featureExtraction_jsons',
                     json_fname)
    with open(json_path, 'r') as json_data:
        settings = json.load(json_data)
    
    # create SSD'd signals per freq-band, per window
    for sub in settings['SUBS_INCL']:
        print(f'\tstart SSD + spectral creation, sub-{sub}...')
        create_SSDs_and_spectral_features(sub=sub, settings=settings)
    
    # create local PAC features    
    for sub in settings['SUBS_INCL']:
        print(f'\tstart local-PAC sub-{sub}...')

    # create multi-location connectivity features


        
    


