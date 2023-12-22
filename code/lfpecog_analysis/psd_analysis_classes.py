"""
Store and Load PSD data for plotting
"""

# import libraries and funcitons
import numpy as np
import os
import json
from dataclasses import dataclass
from itertools import compress, product

from utils.utils_fileManagement import (
    get_project_path, get_avail_ssd_subs,
    load_ft_ext_cfg
)
from lfpecog_analysis.prep_movement_psd_analysis import (
    create_move_specific_ephys
)

# @dataclass(init=True,)
# class PSD_plot_data:
#     sel_subs: list
#     LAT_OR_SCALE: str
#     datatype: str = 'STN'
#     CDRS_RATER: str = 'Patricia'
#     CREATE_NEW_DATA: bool = False
#     LOG_POWER=True
#     ZSCORE_FREQS=True
#     SMOOTH_PLOT_FREQS=0
#     BASELINE_CORRECT=False
#     BREAK_X_AX=False
#     STD_ERR=True
#     plt_ax_to_return=False
#     fsize: int = 12
#     fig_name='PLOT_STN_PSD_vs_DYSK'
#     CALC_FREQ_CORR=False
#     SINGLE_SUB_LINES=False
#     SHOW_ONLY_GAMMA=False
#     SHOW_SIGN=False
#     PROCESS_STATS=False

#     def __post_init__(self,):
#         # get sorted psd data
#         # try to load, either create
#         if not self.CREATE_NEW_DATA:
#             try:
#                 self.psd_data = 
#                 self.tf_freqs=
#                 self.n_subs_incl,
#                 self.cdrs_cat_coding
#             except:
#                 (self.psd_data, self.tf_freqs, self.n_subs_incl,
#                  self.cdrs_cat_coding) = prep_Plot_PSD()
#         else:
#             (self.psd_data, self.tf_freqs, self.n_subs_incl,
#              self.cdrs_cat_coding) = prep_Plot_PSD()



@dataclass(init=True,)
class PSD_vs_Move_sub:
    sub: str
    CDRS_RATER: str = 'Patricia'
    FT_VERSION: str = 'v6'

    def __post_init__(self,):
        self.SETTINGS = load_ft_ext_cfg(FT_VERSION=self.FT_VERSION)

        print(f'adding masks for sub {self.sub}')
        create_move_specific_ephys(sub=self.sub,
                                    FT_VERSION=self.FT_VERSION,
                                    ADD_TO_CLASS=True,
                                    self_class=self)
        # put bands in 3d structure
        self.band_names = list(self.SETTINGS['SPECTRAL_BANDS'].keys())
        self.ephys_sources = self.ssd_sub.ephys_sources
        # loop over available lfp/ecog-left/right
        for src in self.ephys_sources:
            # loop over freq bands
            for i_band, band in enumerate(self.band_names):
                print(band)
                if i_band == 0:
                    src_bands = getattr(getattr(self.ssd_sub, src), band)
                else:
                    src_bands = np.dstack(
                        [src_bands, getattr(getattr(self.ssd_sub, src), band)]
                    )
            # add as 3d array
            setattr(self, f'{src}_3d', src_bands)
        delattr(self, 'ssd_sub')
    
                
            