"""
Main Script to Run multivariate feature Extract (not MIC)

Run on WIN from repo/code:
    python -m lfpecog_features.run_ssd_ftExtr 'ftExtr_spectral_v1.json'
"""
# import functions
import sys
import json
from os.path import join, exists
from os import makedirs

# import own functions
from utils.utils_fileManagement import (
    get_project_path,
    load_ft_ext_cfg
)
from lfpecog_preproc.preproc_import_scores_annotations import get_ecog_side

from lfpecog_features.get_ssd_data import get_subject_SSDs
import lfpecog_features.extract_ssd_features as ssd_feats


if __name__ == '__main__':

    json_fname = sys.argv[1]
    ### load settings from given json-file
    INCL_STN = True
    INCL_ECOG = True
    SETTINGS = load_ft_ext_cfg(json_fname)

    # adjust SSD destination folder to SSD-flank-definition
    try: ssd_flanks = SETTINGS['SSD_flanks']
    except: ssd_flanks = 'narrow'

    if ssd_flanks == 'broadband': ssd_folder = 'SSD_feats_broad'
    else: ssd_folder = 'SSD_feats'

    ### Define paths
    feat_path = join(get_project_path('results'),
                    'features',
                    ssd_folder,
                    SETTINGS["DATA_VERSION"],
                    f'windows_{SETTINGS["WIN_LEN_sec"]}s_'
                    f'{SETTINGS["WIN_OVERLAP_part"]}overlap')
    if not exists(feat_path): makedirs(feat_path)
    
    # create SSD'd signals per freq-band, per window
    for sub in SETTINGS['SUBS_INCL']:
        print(f'\n\tSTART SSD feature extraction: sub-{sub}...')

        # define ephys_sources
        ephys_sources = []
        if INCL_ECOG:
            ecog_side = get_ecog_side(sub)
            ephys_sources.append(f'ecog_{ecog_side}')
        if INCL_STN:
            ephys_sources.extend(['lfp_left', 'lfp_right'])

        # load or create SSD windows for subject
        sub_SSD = get_subject_SSDs(sub=sub,
                                   settings=SETTINGS,
                                   incl_ecog=INCL_ECOG,
                                   incl_stn=INCL_STN)

        # Extract local spectral power features
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_POWERS']:

            print(f'\n\tSTART local powers sub-{sub}...')
            ssd_feats.extract_local_SSD_powers(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
            )
        
        # Extract local spectral power features
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_BURSTS']:

            print(f'\n\tSTART local powers sub-{sub}...')
            ssd_feats.extract_local_SSD_powers(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
            )
        
        # create local PAC features
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_LOCAL_PAC']:
            print(f'\n\tSTART local-PAC sub-{sub}...')

            ssd_feats.extract_local_SSD_PACs(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,   
            )
        
        # create Coherence features
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_COH_STN_STN']:

            print(f'\n\tSTART bi-hemispheric-coherence sub-{sub}...')
            ssd_feats.extract_SSD_connectivity(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
                sources='STN_STN',  connectivity_metric='COH',
            )

        if SETTINGS['FEATS_INCL']['TO_EXTRACT_COH_STN_ECOG']:
            
            print(f'\n\tSTART stn-cortex-coherence sub-{sub}...')
            ssd_feats.extract_SSD_connectivity(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
                sources='STN_ECOG', connectivity_metric='COH',
            )
        
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_PSI_STN_ECOG']:
            
            print(f'\n\tSTART stn-cortex-phase-synch sub-{sub}...')
            ssd_feats.extract_SSD_connectivity(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
                sources='STN_ECOG', connectivity_metric='PSI',
            )
        
        if SETTINGS['FEATS_INCL']['TO_EXTRACT_PSI_STN_STN']:
            
            print(f'\n\tSTART bi-hemispheric-phase-synch sub-{sub}...')
            ssd_feats.extract_SSD_connectivity(
                sub=sub, sub_SSD=sub_SSD, settings_dict=SETTINGS,
                feat_path=feat_path, ephys_sources=ephys_sources,
                sources='STN_STN', connectivity_metric='PSI',
            )
        
        
    


