# Importing Python and external packages
import os
import sys
import importlib
import json
import csv
from dataclasses import dataclass, field, fields
from itertools import compress
import pandas as pd
import numpy as np
from itertools import product

import matplotlib.pyplot as plt

def get_project_path_in_notebook(
    subfolder=False,
):
    """
    Finds path of projectfolder from Notebook.
    Start running this once to correctly find
    other modules/functions
    """
    path = os.getcwd()

    while path[-20:] != 'dyskinesia_neurophys':

        path = os.path.dirname(path)
    
    if isinstance(subfolder, str):
        if os.path.exists(os.path.join(path,
                                       subfolder)):
            path = os.path.join(path, subfolder)
    
    return path

# define local storage directories
codepath = get_project_path_in_notebook('code')
print(codepath)
sys.path.append(codepath)
os.chdir(codepath)

# own utility functions
import utils.utils_fileManagement as utilsFiles
print('utils')
# own data exploration functions
# import lfpecog_features.feats_read_proc_data as read_data
# import lfpecog_preproc.preproc_import_scores_annotations as importClin
import lfpecog_analysis.ft_processing_helpers as ftProc
print('ftproc')
# import lfpecog_analysis.import_ephys_results as importResults
# import lfpecog_analysis.stats_fts_lid_corrs as ftLidCorr
import lfpecog_analysis.load_SSD_features as load_ssdFts
print('load ssd')
# import lfpecog_features.feats_helper_funcs as ftHelp
from lfpecog_features.get_ssd_data import get_subject_SSDs
import lfpecog_predict.prepare_predict_arrays as prep_pred_arrs

# from lfpecog_plotting.plotHelpers import get_colors
# import lfpecog_plotting.plotHelpers as pltHelp
# import lfpecog_plotting.plot_FreqCorr as plotFtCorrs
# import lfpecog_plotting.plot_SSD_feat_descriptives as plot_ssd_descr

import lfpecog_features.get_ssd_data as ssd


ex_sub = '022'
# import dataclass containing SSD data

ft_setting_path = os.path.join(
    utilsFiles.get_project_path('data', USER='timon'),
    'meta_info'
)
FT_VERSION = 'v6'
ssdSub = ssd.get_subject_SSDs(sub=ex_sub,
                             incl_stn=True,
                             incl_ecog=True,
                             ft_setting_fname=f'ftExtr_spectral_{FT_VERSION}.json',
                             ft_setting_path=ft_setting_path,
                             USER='timon',
                             )
