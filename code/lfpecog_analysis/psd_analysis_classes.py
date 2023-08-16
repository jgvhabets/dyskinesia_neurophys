"""
Store and Load PSD data for plotting
"""

# import libraries and funcitons
import numpy as np
import os
import json
from dataclasses import dataclass
from itertools import compress, product

@dataclass(init=True,)
class PSD_plot_data:
    sel_subs: list
    LAT_OR_SCALE: str
    datatype: str = 'STN'
    CDRS_RATER: str = 'Patricia'
    CREATE_NEW_DATA: bool = False
    LOG_POWER=True
    ZSCORE_FREQS=True
    SMOOTH_PLOT_FREQS=0
    BASELINE_CORRECT=False
    BREAK_X_AX=False
    STD_ERR=True
    plt_ax_to_return=False
    fsize: int = 12
    fig_name='PLOT_STN_PSD_vs_DYSK'
    CALC_FREQ_CORR=False
    SINGLE_SUB_LINES=False
    SHOW_ONLY_GAMMA=False
    SHOW_SIGN=False
    PROCESS_STATS=False

    def __post_init__(self,):
        # get sorted psd data
        # try to load, either create
        if not self.CREATE_NEW_DATA:
            try:
                self.psd_data = 
                self.tf_freqs=
                self.n_subs_incl,
                self.cdrs_cat_coding
            except:
                (self.psd_data, self.tf_freqs, self.n_subs_incl,
                 self.cdrs_cat_coding) = prep_Plot_PSD()
        else:
            (self.psd_data, self.tf_freqs, self.n_subs_incl,
             self.cdrs_cat_coding) = prep_Plot_PSD()