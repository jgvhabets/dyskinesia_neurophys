"""
Plot Connectivity metrics against Dyskinesia
scores (CDRS)

run on WIN as:
xxx\dyskinesia_neurophys\code> python -m lfpecog_plotting.plot_Spectrals_vs_LID
"""
# import public packages
from dataclasses import dataclass, field
from itertools import product
import os
import matplotlib.pyplot as plt

# import custom functions
from lfpecog_analysis.process_connectivity import (
    get_conn_values_to_plot
)
from utils.utils_fileManagement import load_ft_ext_cfg
from lfpecog_analysis.get_SSD_timefreqs import get_all_ssd_timeFreqs

from lfpecog_plotting.plot_descriptive_SSD_PSDs import (
    plot_scaling_LID
)


@dataclass(init=True, repr=True,)
class plot_Spectrals_vs_LID:
    """
    Class to plot spectral features in PSDs style
    """
    PLOT_DATE: str = '000'
    CDRS_ORIGIN: str = 'bilat'  # use unilateral (contra lat ecog) or bilater ecog scores
    INCL_CORE_CDRS: bool = True  # include core/axial CDRS scores (neck-trunk-face)
    DATA_TYPE: str = 'mic'  # Connect metric mic / trgc
    INCL_CONN_SIDES: list = field(default_factory=lambda: ['ipsilateral', 'contralateral'])
    SELECT_MOVEMENT: bool or str = False  # should be False, INCL_MOVE, EXCL_MOVE
    BASELINE_CORRECT: bool = True
    BASELINE_WO_MOVE: bool = True
    SMOOTH_PLOT_FREQS: int = 4  # defaults zero
    FT_VERSION: str = 'v6'
    VERBOSE: bool = False

    def __post_init__(self,):
        # settings autom. extracted based on defined variables
        self.SETTINGS = load_ft_ext_cfg(FT_VERSION=self.FT_VERSION)
        self.DATA_VERSION = self.SETTINGS['DATA_VERSION']
        
        # take only subjects with ECoG & LFP data
        self.SUBJECTS = [sub for sub in self.SETTINGS['TOTAL_SUBS']
                            if sub.startswith("0")]
        
        # show absolute PSDs for TRGC (default no baseline correct)
        if self.DATA_TYPE == 'trgc': self.BASELINE_CORRECT = False
        
        # add figure name based on alle variables
        self.create_figname()

        
        ### PERPARE AND EXECUTE PLOT ###
        
        ### get values
        TFs = get_all_ssd_timeFreqs(
            SUBS=self.SUBJECTS,
            FT_VERSION=self.FT_VERSION,
            DATA_VERSION=self.DATA_VERSION,
            GET_CONNECTIVITY=self.DATA_TYPE,
            verbose=self.VERBOSE,
        )
        print('got tf values')

        ### sort and average values into categories 
        plot_values, freqs, _ = get_conn_values_to_plot(
            TFs,
            BASELINE_CORRECT=self.BASELINE_CORRECT,
            BASELINE_EXCL_MOVE=self.BASELINE_WO_MOVE,
            CDRS_SIDE=self.CDRS_ORIGIN,
            INCL_CORE_CDRS=self.INCL_CORE_CDRS,
            incl_conn_sides=self.INCL_CONN_SIDES,
            SELECT_MOVEMENT=self.SELECT_MOVEMENT,
            verbose=self.VERBOSE,
        )
        print('got plot values')

        ### plot metric-freq (PSD style)
        plot_scaling_LID(
            psds_to_plot=plot_values,
            tf_freqs=freqs,
            cdrs_origin=self.CDRS_ORIGIN,
            cdrs_cat_coding={'no': 0, 'mild': 1,
                            'moderate':2 , 'severe': 3},
            datatype=self.DATA_TYPE,
            BASELINE_CORRECT=self.BASELINE_CORRECT,
            SELECT_MOVEMENT=self.SELECT_MOVEMENT,
            SMOOTH_PLOT_FREQS=self.SMOOTH_PLOT_FREQS,
            fig_name=self.FIG_NAME,
            FT_VERSION=self.FT_VERSION,
            DATA_VERSION=self.DATA_VERSION,
        )
        print(f'plotted {self.FIG_NAME}')  
        
        
    # functions that are used within class post-init 
    def create_figname(self):
        # generate figure name
        if len(self.INCL_CONN_SIDES) == 2: mic_type = 'both'
        else: mic_type = self.INCL_CONN_SIDES[0].split('lat')[0]

        self.FIG_NAME = (f'{self.PLOT_DATE}_{self.DATA_TYPE.upper()}'
                            f'{mic_type}_{self.CDRS_ORIGIN}LID')
        
        if self.INCL_CORE_CDRS: self.FIG_NAME += 'core'
        if self.SELECT_MOVEMENT: self.FIG_NAME += f'_{self.SELECT_MOVEMENT}'
        if self.BASELINE_CORRECT:
            self.FIG_NAME += f'_blCorr'
            if self.BASELINE_WO_MOVE: self.FIG_NAME += f'WoMove'
        if self.SMOOTH_PLOT_FREQS > 0: self.FIG_NAME += f'_smooth{self.SMOOTH_PLOT_FREQS}'



# CALL EXECUTION OF PLOTTING CLASS
if __name__ == '__main__':
    """
    Considerations for plotting:
        - baseline is now only NO-DYSKINESIA (no movement),
            consider to restrict the baseline selection to
            a max. amount of minutes.
    """
    
    for DATA_TYPE, SELECT_MOVEMENT in product(
        ['trgc',],
        [False, 'INCL_MOVE', 'EXCL_MOVE']
    ):
        print(f'start {type(DATA_TYPE)} x {type(SELECT_MOVEMENT)}')
        plot_Spectrals_vs_LID(
            DATA_TYPE=DATA_TYPE,
            SELECT_MOVEMENT=SELECT_MOVEMENT,
            PLOT_DATE='DECv1',
        )        
