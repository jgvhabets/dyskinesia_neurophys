"""
Plot Spectrals against Dyskinesia scores (CDRS) (depr.)

Only scatter plot used

run on WIN as:
xxx\dyskinesia_neurophys\code> python -m lfpecog_plotting.plot_Spectrals_vs_LID
"""
# import public packages
from dataclasses import dataclass, field
from itertools import product
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# import custom functions
# from lfpecog_analysis.process_connectivity import (
#     get_conn_values_to_plot
# )
from utils.utils_fileManagement import (
    load_ft_ext_cfg, get_project_path
)
from lfpecog_analysis.get_SSD_timefreqs import get_all_ssd_timeFreqs
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side
)
from lfpecog_plotting.plot_descriptive_SSD_PSDs import (
    plot_scaling_LID
)
from lfpecog_features.feats_spectral_helpers import (
    add_mean_gamma_column
)
from lfpecog_plotting.plotHelpers import get_plot_jitter
from lfpecog_analysis.ft_processing_helpers import remove_ipsiECOG_unilatLID


def scatter_Feats_LID_MOVE(
    FeatClass, AccClass=False,
    SAVE_FIG = True,
    SRC: str = 'lfp',  # lfp or ecog
    FIG_NAME = '00_PowerScatter_LID_MOVE',
    POW_or_COH = 'POW',
    power_feats = {'ecog': ['ecog_theta_mean_psd',
                            'ecog_lo_beta_mean_psd',
                            'ecog_gammaBroad_mean_psd'],
                    'lfp': [['lfp_left_theta_mean_psd',
                             'lfp_right_theta_mean_psd'],
                            ['lfp_left_lo_beta_mean_psd',
                             'lfp_right_lo_beta_mean_psd'],
                            ['lfp_left_gammaBroad_mean_psd',
                             'lfp_right_gammaBroad_mean_psd']]},
    coh_feats = {'ecog': ['sq_coh_STN_ECOG_theta',
                          'sq_coh_STN_ECOG_lo_beta',
                          'sq_coh_STN_ECOG_gammaBroad'],
                 'lfp': ['sq_coh_STN_STN_theta',
                         'sq_coh_STN_STN_lo_beta',
                         'sq_coh_STN_STN_gammaBroad']},
    pow_labels = ['THETA power', 'BETA power', 'GAMMA power'],
    gamma_mean_or_peakband: str = 'mean',  # peak or mean
    ZSCORE_POW: bool = True,
    SHUFFLE_SCATTERS: bool = True,
    shareX = True,
    ZERO_SPACE: bool = True,
    EXCL_FREE: bool = True,
    remove_uniLID_ipsiECOG: bool = False,
    task_minutes = None,
):
    if POW_or_COH == 'POW':
        feat_lists = power_feats
    elif POW_or_COH == 'COH':
        feat_lists = coh_feats
        FIG_NAME = FIG_NAME.replace('Power', 'Cohsq')
        pow_labels = [l.replace('power', 'coherence') for l in pow_labels]
    else:
        raise ValueError('POW_or_COH should be "POW" or "COH"')
    
    if shareX: sharex='col'
    else: sharex = 'none'
    
    fig, axes = plt.subplots(len(feat_lists[SRC]), 1,
                            figsize=(6, 3 * len(feat_lists[SRC])),
                            sharex=sharex,)

    shuf_subs = list(FeatClass.FT_LABELS.keys())
    np.random.seed(27)
    np.random.shuffle(shuf_subs)
    fsize=14

    for i_ft, pow_ft in enumerate(feat_lists[SRC]):
        x_values, y_values, pow_values = [], [], []

        for sub in shuf_subs:
            # add mean gamma column or change gamma bandkey
            add_mean_gamma_column(FeatClass.FEATS[sub], MAX=False,)
            
            # get data to plot
            x_temp = FeatClass.FT_LABELS[sub].copy()  # X axis: Dyskinesia (CDRS sum)
            if AccClass: y_temp = AccClass.ACC_RMS[sub].copy()  # X-axis: Movement (ACC RMS)
            else: y_temp = FeatClass.ACC_RMS[sub].copy()  # X-axis: Movement (ACC RMS)
            times = FeatClass.FEATS[sub].index.values.copy()
            scores = FeatClass.FT_LABELS[sub].copy()
            # exclude FREE
            if EXCL_FREE:
                # select feat minutes that appear in FeatSet with only Rest and Tap
                task_sel = [round(t) in task_minutes[sub][0] for t in times]
                x_temp = x_temp[task_sel]
                y_temp = y_temp[task_sel]
                times = times[task_sel]
                scores = scores[task_sel]
            
            # correct for ecog side
            if SRC == 'ecog':
                if gamma_mean_or_peakband == 'peak': pow_ft = pow_ft.replace('gammaBroad', 'gammaPeak')

                # Remove ipsilat ECoG to unilat LID
                if remove_uniLID_ipsiECOG:
                    keep_idx = remove_ipsiECOG_unilatLID(sub=sub, ft_times=times.copy())
                    x_temp = x_temp[keep_idx]
                    y_temp = y_temp[keep_idx]
                    times = times[keep_idx]
                    scores = scores[keep_idx]
                    print(f'...sub-{sub} kept {sum(keep_idx)} / {len(keep_idx)} samples for IPSI-unilat ECoG')
                     

                ecog_side = get_ecog_side(sub)
                pow_temp = FeatClass.FEATS[sub][pow_ft.replace('ecog', f'ecog_{ecog_side}')].copy()
                if EXCL_FREE: pow_temp = pow_temp[task_sel]
                if remove_uniLID_ipsiECOG: pow_temp = pow_temp[keep_idx]
                # indiv zscore powers
                if ZSCORE_POW:
                    base_sel = np.logical_and(scores == 0, times < 5)
                    base_m = np.mean(pow_temp[base_sel])
                    base_sd = np.std(pow_temp[base_sel])
                    pow_temp = (pow_temp - base_m) / base_sd
                # get jitter
                x_jitter, y_jitter = get_plot_jitter(x_temp, y_temp, ZERO_SPACE=ZERO_SPACE,)
                # add to overall lists
                x_values.extend(x_temp + x_jitter)
                y_values.extend(y_temp + y_jitter)
                pow_values.extend(pow_temp)

            elif SRC == 'lfp':
                # create list with only STN-STN feature for parallel structure of coh and powers
                if POW_or_COH == 'COH' and isinstance(pow_ft, str): pow_ft = [pow_ft,]
                # loop over left and right stn
                for i_lfp, ft_temp in enumerate(pow_ft):
                    # make copy from x_temp, bcs x_temp can be changed (extra x-space for CDRS==0)
                    if i_lfp == 0: x_temp_store = x_temp.copy()  # save original x_temp
                    elif i_lfp == 1: x_temp = x_temp_store  # reset x_temp

                    if gamma_mean_or_peakband == 'peak': ft_temp = ft_temp.replace('gammaBroad', 'gammaPeak')

                    pow_temp = FeatClass.FEATS[sub][ft_temp].copy()
                    if EXCL_FREE: pow_temp = pow_temp[task_sel]
                    # indiv zscore powers
                    if ZSCORE_POW:
                        base_sel = np.logical_and(scores == 0, times < 5)
                        base_m = np.mean(pow_temp[base_sel])
                        base_sd = np.std(pow_temp[base_sel])
                        pow_temp = (pow_temp - base_m) / base_sd
                    # get jitter
                    x_jitter, y_jitter = get_plot_jitter(x_temp, y_temp,
                                                         ZERO_SPACE=ZERO_SPACE,)
                    # add to overall lists
                    x_values.extend(x_temp + x_jitter)
                    y_values.extend(y_temp + y_jitter)
                    pow_values.extend(pow_temp)


        # scatter plot data
        if SRC == 'lfp': a = .3
        else: a = .5
        v_range = (-3, 3)

        if SHUFFLE_SCATTERS:
            np.random.seed(27)
            idx = np.arange(len(x_values))
            np.random.shuffle(idx)
            x_values = np.array(x_values)[idx]
            y_values = np.array(y_values)[idx]
            pow_values = np.array(pow_values)[idx]

        scat = axes[i_ft].scatter(x_values, y_values, c=pow_values,
                                  alpha=a, cmap='viridis',
                                  vmin=v_range[0], vmax=v_range[1],)
        cbar = axes[i_ft].scatter([], [], c=[], cmap='viridis',
                                  vmin=v_range[0], vmax=v_range[1],)  # colorbar without transparency
        if POW_or_COH == 'POW':
            cbar_lab = rf"$\bf{pow_labels[i_ft].split(' ')[0]}$" + " " + rf"$\bfPower$" + "\n(indiv. z-score)"
        else:
            cbar_lab = rf"$\bf{pow_labels[i_ft].split(' ')[0]}$" + " " + rf"$\bfsq.$" + rf"$\bfCoherence$" + "\n(indiv. z-score)"

        plt.colorbar(cbar, ax=axes[i_ft], label=cbar_lab,)

        # plot meta info
        # axes[i_ft].set_title(pow_labels[i_ft], size=fsize,
        #                      weight='bold',)
        ylab = rf"$\bfMovement$" + " " + rf"$\bfpresence$" + "\n(indiv. z-scored acc-rms)"
        xlab = rf"$\bfDyskinesia$" + " " + rf"$\bfseverity$" + "\n(total CDRS score)"
        if not shareX: axes[i_ft].set_xlabel(xlab, size=fsize,)
        axes[i_ft].set_ylabel(ylab, size=fsize,)
        axes[i_ft].set_ylim(-2, 4)

        xticks = np.arange(0, 15, 2)
        xticklabels = np.arange(0, 15, 2)
        if ZERO_SPACE: xticks[0] = -1.5
        axes[i_ft].set_xticklabels(xticklabels, size=fsize,)
        axes[i_ft].set_xticks(xticks, size=fsize,)

        axes[i_ft].axvline(x=.3, lw=5, color='lightgray',
                           alpha=.7,)

    if shareX: axes[-1].set_xlabel(xlab, size=fsize,)
    

    if SRC == 'lfp' and POW_or_COH == 'POW': src = 'Subthalamic Power'
    elif SRC == 'lfp' and POW_or_COH == 'COH': src = 'Inter-subthalamic Coherence'
    elif SRC == 'ecog' and POW_or_COH == 'COH': src = 'Cortico-subthalamic Coherence'
    elif SRC == 'ecog' and POW_or_COH == 'POW': src = 'Cortical Power'
    plt.suptitle(f'{src}:\nMovement and Dyskinesia dependence',
                 size=fsize+4, weight='bold',)

    plt.tight_layout(pad=.5,)

    FIG_PATH = os.path.join(
        get_project_path('figures'),
        'final_Q1_2024',
        'feat_scatter_LID_MOVE',
    )
    if SAVE_FIG:   
        plt.savefig(os.path.join(FIG_PATH, FIG_NAME),
                    dpi=300, facecolor='w',)
        print(f'saved plot {FIG_NAME} in {FIG_PATH}!')
        plt.close()
    else: plt.show()



def plot_ratio_biomarker(
    ratio_arr, t_new, 
    Z_SCORE_RATIOS = True,
    MIN_SUBS = 6, fsize = 14,
    SMOOTH_WIN : int = 0,
    SAVE_FIG: bool = False,
):
    # subjects contributing to mean, per window
    n_hemisf_present = np.sum(~np.isnan(ratio_arr), axis=0)
    n_subs_wins = np.sum(~np.isnan(ratio_arr), axis=0) / 2

    # original arr
    if not Z_SCORE_RATIOS:
        ratio_error = np.nanstd(ratio_arr, axis=0) / np.sqrt(n_hemisf_present)
        ratio_std = np.nanstd(ratio_arr, axis=0)
        ratio_mean = np.nanmean(ratio_arr, axis=0)

    # use zscored arr
    elif Z_SCORE_RATIOS:
        z_ratios = np.array([(row - np.nanmean(row)) / np.nanstd(row)
                        for row in  ratio_arr])

        ratio_error = np.nanstd(z_ratios, axis=0) / np.sqrt(n_hemisf_present)
        ratio_std = np.nanstd(z_ratios, axis=0)
        ratio_mean = np.nanmean(z_ratios, axis=0)


    fig, ax = plt.subplots(1,1, figsize=(8, 4))

    x = t_new[n_subs_wins >= MIN_SUBS] / 60
    y = ratio_mean[n_subs_wins >= MIN_SUBS]
    var = ratio_error[n_subs_wins >= MIN_SUBS]

    if SMOOTH_WIN > 0:
            y = savgol_filter(y, window_length=SMOOTH_WIN, polyorder=3,)

    ax.plot(x, y)
    ax.fill_between(x, y1=y - var, y2=y + var, alpha=.5,)
    ax.axvline(x=0, color='gray', alpha=.5, lw=3, ls='--',)

    if Z_SCORE_RATIOS: r = 'Z-scored Ratio'
    else: r = 'Ratio'
    ax.set_ylabel(f'STN Theta * Gamma / Beta {r}',  #\n(4-8 / 12-20 Hz)',
            size=fsize, weight='bold',)
    ax.set_xlabel('Time vs Dyskinesia Onset (minutes)',
            size=fsize, weight='bold',)

    ax.tick_params(axis='both', size=fsize, labelsize=fsize,)

    if Z_SCORE_RATIOS:
            for y in [-2, -1, 0, 1, 2, 3]:
                    ax.axhline(y=y, color='gray', alpha=.3, lw=.5,)
            ax.set_ylim(-2, 2)
    else:
            for y in [-.1, 0, 0.1]:
                    ax.axhline(y=y, color='gray', alpha=.3, lw=.5,)
            ax.set_ylim(-.2, .2)
            ax.axhline(y=0, color='gray', alpha=.5, lw=1,)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if SAVE_FIG:
            plt.savefig(os.path.join(get_project_path('figures'),
                                        'final_Q1_2024',
                                        'prediction', 'ratio',
                                        f'theta_beta_ratio_v1_10sec_min{MIN_SUBS}subs_smooth{SMOOTH_WIN}'),
                        dpi=300, facecolor='w',)
            plt.close()
    else:

            plt.show()



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
    # SELECT_MOVEMENT: bool or str = False  # should be False, INCL_MOVE, EXCL_MOVE
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
