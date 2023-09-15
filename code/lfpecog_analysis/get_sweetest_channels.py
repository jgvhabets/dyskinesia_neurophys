# import functions
from os.path import join
from os import listdir
import json
from pandas import read_csv, read_excel, DataFrame
import numpy as np

from utils.utils_fileManagement import get_onedrive_path

STN_bids_codes = {'101': 'L005',
                  '102': 'L006',
                  '103': 'L007',
                  '105': 'L013',
                  '107': 'L015',
                  '108': 'L016',
                  '109': 'L017'}

def get_contact_coords(sub, ch_names):

    bids_path = get_onedrive_path('bids_rawdata')

    if sub[0] == '0':
        sub_path = join(bids_path, f'sub-EL{sub}')
    elif sub[0] == '1':
        
        sub_path = join(bids_path, f'sub-{STN_bids_codes[sub]}')

    coord_df = False

    for ses in listdir(sub_path):
        if isinstance(coord_df, DataFrame): continue
        files = listdir(join(sub_path, ses, 'ieeg'))
        for f in files:
            if isinstance(coord_df, DataFrame): continue
            if 'MNI' in f and f.endswith('electrodes.tsv'):
                coord_df = read_csv(join(sub_path, ses, 'ieeg', f),
                                    sep='\t')
    
    assert not all(coord_df['x'] == 0), f'loca missing for sub-{sub}'

    # print(f'coord df for sub-{sub}', coord_df)



    
    return coord_df


def find_closest_to_sweetspot(coords, source, chs_present='all',):
    """
    sweetspots stored as x, y, z MNI coordinates
    (STN sweetspots: Dembek et al: https://onlinelibrary.wiley.com/doi/10.1002/ana.25567)
    (ECoG sweetspots based on Humain Brain Map and
    are within defined ranges in Mayka, NeuroImage https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2034289/.
    )
    """
    swspots = {'LFP_L': (-12.68, -13.53, -5.38),
               'LFP_R': (12.5, -12.72, -5.38),
               'ECOG_L': (-28, -22, 69),  # -37	-21	58
               'ECOG_R': (34, -17, 69)}
    
    source_coords = [source in n for n in coords['name']]
    coords = coords.iloc[source_coords, :].reset_index(drop=True)

    if chs_present != 'all':
        # find which channels in coords are mentioned in present channels
        chs_sel = [any([cp in c for cp in chs_present]) for c in coords['name']]
        chs_sel = coords.index[~np.array(chs_sel)]
        coords.iloc[chs_sel, [1, 2, 3]] = np.nan
    
    # print(coords)


    if coords['manufacturer'][0] == 'Medtronic' and coords.shape[0] == 8:
        print(f'SenSight assumed')
        xyz_levels = {0: coords[['x', 'y', 'z']].values[0],
                      1: np.nanmean(coords[['x', 'y', 'z']].values[1:4], axis=0),
                      2: np.nanmean(coords[['x', 'y', 'z']].values[4:7], axis=0),
                      3: coords[['x', 'y', 'z']].values[7]}
        
    return xyz_levels


    # print(coords)
    

