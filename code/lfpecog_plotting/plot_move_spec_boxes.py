"""
Plots Boxplots of Spectral Changes selected
on millisecond precise movement selection
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, compress
from pandas import read_csv

from utils.utils_fileManagement import (
    get_project_path, load_ft_ext_cfg
)
from lfpecog_features.feats_helper_funcs import (
    get_indiv_peak_freqs,
)
from lfpecog_plotting.plot_descriptive_SSD_PSDs import (
    break_x_axis_psds_ticks
)
from lfpecog_analysis.specific_ephys_selection import (
    get_hemisphere_movement_location
)
from lfpecog_analysis.psd_analysis_classes import (
    get_baseline_arr_dict
)
from lfpecog_analysis.psd_lid_stats import run_mixEff_wGroups

# cond_colors = {
#     'nolid': 'green',
#     'nolidbelow30': 'limegreen',
#     'nolidover30': 'darkgreen',
#     'alllid': 'blue', 'mildlid': 'orange',
#     'moderatelid': 'red', 'severelid': 'purple',
# }

lid_labels = ["None\n<30'", "None\n>30'", 'None',
                'Mild', 'Moderate', 'Severe']

lid_clrs = ['limegreen', 'darkgreen', 'darkgreen',
            'orange', 'firebrick', 'darkorchid']  # 'mediumblue

ind_peak_bands = ['lo_beta', 'hi_beta', 'narrow_gamma']

band_labels = ['Theta', 'Alpha', 'low Beta',
            'high Beta', 'Gamma', 'peak Gamma']

lid_states = ['nolidbelow30', 'nolidover30', 'nolid',
                  'mildlid', 'moderatelid', 'severelid']


def get_canonical_bands(SETTINGS=None, FT_VERSION='v6'):
    # get freq bands (with correct naming)
    if not SETTINGS:
        SETTINGS = load_ft_ext_cfg(FT_VERSION=FT_VERSION)

    f_bands = {}

    for k, v in SETTINGS['SPECTRAL_BANDS'].items():
        if k != 'delta' and not 'gamma' in k: f_bands[k] = v
        else: f_bands['theta'] = v
    f_bands['broad_gamma'] = [60, 90]
    f_bands['narrow_gamma'] = ['tba', 'tba']

    return f_bands


def calc_fband_boxlists(
    psd_dict, baselines,
    SRC_SEL: str = 'lfp',  
    STATE_SEL: str = 'rest',
    SPLIT_MOVEMENT: bool = False,
    MEAN_EPOCH = 1,  # n 1-sec windows to average per value
    verbose: bool = False,
):
    """
    
    Arguments:
        - SPLIT_MOVEMENT: returns data to split voluntary
            and involuntary movement
    """
    # check and process defined
    allowed_state_sels = [['rest', 'dyskmove'],
                          ['tap', 'dyskmove'],
                          'rest', 'tap', 'dyskmove', 'movement', ]
    assert STATE_SEL in allowed_state_sels, 'incorrect STATE_SEL'
    if STATE_SEL == 'movement': STATE_SEL = ['tap', 'dyskmove']
    else: SPLIT_MOVEMENT = False  # default false if not movement

    # get baseline arrays
    BL_arrs = get_baseline_arr_dict(BLs_1s=baselines)

    # get freq bands and indiv peaks
    f_bands = get_canonical_bands(list(psd_dict.values())[0].SETTINGS)
    IND_PEAKS = get_indiv_peak_freqs(psd_dict=psd_dict, STATE='all')

    # create dicts and lists to store
    psd_lists = {b: {l: [] for l in lid_states}
                for b in f_bands.keys()}
    subid_lists = {l: [] for l in lid_states}
    if SPLIT_MOVEMENT: movetype_lists = {l: [] for l in lid_states}

    # loop over all states
    for state in psd_dict.keys():
        # only continue with selected condition data
        if isinstance(STATE_SEL, str):
            if not STATE_SEL in state.lower():
                if verbose: print(f'...SKIP state (SEL): {state}')
                continue
        elif isinstance(STATE_SEL, list):
            if not any([s in state.lower() for s in STATE_SEL]):
                if verbose: print(f'...SKIP state (SEL): {state}')
                continue
        if not any([l in state for l in lid_states]):
            if verbose: print(f'...SKIP state (lid state): {state}')        
            continue
        
        if verbose: print(f'...include state: {state}')
        
        LID_CODE = np.where([l in state for l in lid_states])[0][0]
        LID_CODE = lid_states[LID_CODE]
        if verbose: print(f'\ncontinue with {state}  (LID code: {LID_CODE})')
        freqs = psd_dict[state].freqs

        # only continue with subject data
        for k in vars(psd_dict[state]).keys():

            s = k.split(state)[0]
            if not s.endswith('_'): continue
            src = s[:-5]
            sub = s[-4:-1]

            if not SRC_SEL in src: continue
                    
            # get psd array (samples, freqs)
            psx = getattr(psd_dict[state], k)

            # Z-SCORE against baseline (mean and stddev)
            try:
                psx = (psx - BL_arrs[f'{sub}_{src}'][0]) / BL_arrs[f'{sub}_{src}'][1]
            except:
                if verbose: print(f'\tno baseline found for {sub} {src}, SKIPPED')
                continue

            # select indiv peaks
            sub_peaks = IND_PEAKS[sub]

            for bw, f_range in f_bands.items():
                # select relevant freq range from psd (indiv-PEAK or RANGE)
                if bw in list(sub_peaks.keys()):
                    peak_f = sub_peaks[bw]
                    f_range = [peak_f - 2, peak_f + 3]
                f_sel = np.logical_and(freqs >= f_range[0],
                                    freqs < f_range[1])
                # get power mean over freq-range, per 1-sec epoch
                powers = np.mean(psx[:, f_sel], axis=1)

                # if bw == 'theta': print(f'n 1-sec epochs: {len(powers)} ({k})')            
                
                if MEAN_EPOCH > 1:
                    powers = [np.mean(powers[i:i+MEAN_EPOCH])
                              for i in np.arange(0, len(powers), MEAN_EPOCH)]
                    if bw == 'theta' and verbose:
                        print(f'n {MEAN_EPOCH}-sec epochs: {len(powers)} ({k})')            
                
                psd_lists[bw][LID_CODE].extend(powers)

            # add sub-id code ONLX ONCE for every added sample (outside bw loop)
            subid_lists[LID_CODE].extend([sub] * len(powers))

            if SPLIT_MOVEMENT:
                if 'dyskmove' in k: movetype = 'involuntary'
                elif 'tap' in k: movetype = 'voluntary'
                else: raise ValueError('no movetype found')
                if verbose: print(f'...added "{movetype}", n={len(powers)} ({k})')
                movetype_lists[LID_CODE].extend([movetype] * len(powers))

    if not SPLIT_MOVEMENT: return psd_lists, subid_lists
    elif SPLIT_MOVEMENT: return psd_lists, subid_lists, movetype_lists


def plot_moveSpec_boxplots(
    psd_box, ids_box,
    PLOT_SUB_MEANS: bool = True,
    ADD_SUB_DOTS: bool = False,
    STATE_SEL = 'rest',
    SOURCE_SEL = 'lfp',
    ADD_LMM_COEF: bool = False,
    fsize = 14,
    FIG_SAVE: bool = False,
    fig_name_start: str = '',
):
    ALPHA = .01
    # PLOT BOXPLOTS PER FREQ BAND; FOR SELECTED CONDITIONS(S)
    print(f'included conditions: {STATE_SEL}')
    print(f'selected data-source: {SOURCE_SEL}')


    fig, axes = plt.subplots(len(psd_box), 1,
                            # figsize=(6, 9),
                            figsize=(5, 7),
                             sharex='col',
                            )

    for i_ax, bw in enumerate(psd_box.keys()):

        values = psd_box[bw]

        if PLOT_SUB_MEANS:
            assert len(values) == len(ids_box), 'IDs dont match values lists'

            mean_values, dot_ids = [], []  # mean ids is stored to connect subject dots
            for lid_key in ids_box.keys():
                
                lid_temp, lid_sub_temp = [], []  # store mean values per lid
                
                lid_values = values[lid_key]
                lid_subs = ids_box[lid_key]

                for sub in np.unique(lid_subs):
                    sub_sel = np.array(lid_subs) == sub
                    sub_mean = np.mean(np.array(lid_values)[sub_sel])
                    lid_temp.append(sub_mean)
                    lid_sub_temp.append(sub)

                mean_values.append(lid_temp)
                dot_ids.append(lid_sub_temp)
            
            value_lists = mean_values
            
        else:
            value_lists = psd_box[bw].values

        # BOX PLOT values per LID category (plot only present categories)
        BOX_SPACING, BOX_WIDTH = .25, .15
        box_sel = [len(l) > 0 for l in value_lists]
        boxxticks = np.arange(0, BOX_SPACING*sum(box_sel), BOX_SPACING)
        boxvalues = list(compress(value_lists, box_sel))  # lists with present boxes
        boxplot = axes[i_ax].boxplot(
            boxvalues,
            positions=boxxticks, widths=BOX_WIDTH,
            patch_artist=True, showfliers=False,
        )
        # make boxplots pretty
        box_clrs = list(compress(lid_clrs, box_sel))
        for patch, clr in zip(boxplot['boxes'], box_clrs):
            patch.set_facecolor(clr)
            patch.set_alpha(.4)

        for median in boxplot['medians']:
            median.set_color('black')

        # SCATTER PLOT sub means and connect individual dots
        if ADD_SUB_DOTS and PLOT_SUB_MEANS:
            np.random.seed(42)
            for i_box, (xtick, box_dots) in enumerate(zip(
                boxxticks, boxvalues
            )):
                xjitter = np.random.uniform(low=xtick-.08, high=xtick+.08,
                                            size=len(box_dots))
                axes[i_ax].scatter(xjitter, box_dots, color=box_clrs[i_box],
                                   alpha=.5, s=25,)
            # connect sub-dots
            box_ids = list(compress(dot_ids, box_sel))
            for sub in np.unique([s for l in box_ids for s in l ]):
                sub_sels = [np.array(l) == sub for l in box_ids]  # gives sub bool per box
                sub_y = [np.array(v)[s] for v, s in zip(boxvalues, sub_sels)]  # takes only sub values
                y_sel = [len(l) == 1 for l in sub_y]
                sub_y = [l[0] for l in np.array(sub_y, dtype='object')[y_sel]]  # extract only values
                sub_x = boxxticks[y_sel]
                if len(sub_y) < 2: continue
                axes[i_ax].plot(sub_x, sub_y, c='gray', lw=.5,
                                alpha=.3,)

        if ADD_LMM_COEF:
            # prepare equal lists with all stat info
            all_powers = list(compress(list(psd_box[bw].values()), box_sel))
            stat_powers = [v for l in all_powers for v in l]
            all_subids = list(compress(list(ids_box.values()), box_sel))
            stat_subids = [s for l in all_subids for s in l]
            stat_scores = [[i] * len(l) for i, l in enumerate(all_subids)]
            stat_scores = [s for l in stat_scores for s in l]
            
            assert len(stat_powers) == len(stat_scores) == len(stat_subids), (
                '(BOX, LMM) given powers, scores and sub-ids differ in length'    
            )
            
            # run linear mixed effect model
            coef, pval = run_mixEff_wGroups(
                dep_var=np.array(stat_powers),
                indep_var=np.array(stat_scores),
                groups=np.array(stat_subids), TO_ZSCORE=False,
            )
            SIGN = pval < (ALPHA / len(band_labels))

            # plot only for legend
            if SIGN: leg_props = {'weight':'bold'}
            else:  leg_props = {}
            if 'beta' in bw:
                leg_loc = {'loc': 'upper right',
                            'bbox_to_anchor': ((.95, .98))}
            else:
                leg_loc = {'loc': 'lower right',
                            'bbox_to_anchor': ((.95, .02))}
            axes[i_ax].scatter(
                [], [], lw=.1, c='w',
                label=f'Coef.: {round(coef, 3)} (p={round(pval, 3)})'
            )
            axes[i_ax].legend(frameon=False, fontsize=fsize - 2,
                              prop=leg_props, **leg_loc)
                            #   loc='lower right', bbox_to_anchor=(.99, .01),

        # if bw in ind_peak_bands: title = f'{bw} (indiv peak)'
        # else: title = f'{bw} (broad mean)'
        # axes[i_ax].set_title(title, weight='bold', size=fsize,)

        axes[i_ax].set_xticks(boxxticks)
        axes[i_ax].set_xticklabels(list(compress(lid_labels, box_sel)),
                                size=fsize-2,)
        axes[i_ax].set_xlim(-BOX_WIDTH, boxxticks[-1] + BOX_WIDTH)
        axes[i_ax].set_yticks(np.arange(-2, 2.1, .5))
        axes[i_ax].set_yticklabels(np.arange(-2, 2.1, .5), size=fsize,)
        
        if not PLOT_SUB_MEANS: axes[i_ax].set_ylim(-2, 2)
        elif PLOT_SUB_MEANS: axes[i_ax].set_ylim(-.75, .75)


        axes[i_ax].set_ylabel(
            # rf"$\bf{band_labels[i_ax]}$" + "\n(z-scores)",  # two separate string parts to save \n function
            band_labels[i_ax].replace(' ', '\n'),  # two separate string parts to save \n function
            size=fsize, weight='bold',
        )
    # pretty axes
    for ax in axes:
        ax.tick_params(size=fsize, axis='both',)
        ax.spines[['right', 'top']].set_visible(False)

    # overall title
    # T = f'{SOURCE_SEL.upper()}: Spectral changes during {STATE_SEL.upper()}'
    T = (rf"$\bf{SOURCE_SEL.upper()}$" + " " +
         rf"$\bf changes $" + " " rf"$\bf during $" + " " +
         rf"$\bf{STATE_SEL.upper()}$"
         "\n(indiv. z-scored powers)")
    T = T.replace('LFP', 'STN')
    plt.suptitle(T, x=.55, y=.98, ha='center', size=fsize)  # weight='bold', 

    plt.tight_layout(h_pad=.001)

    if FIG_SAVE:
        figname = f'{fig_name_start}SpecBandBoxes_{SOURCE_SEL}_{STATE_SEL}'
        if PLOT_SUB_MEANS:
            figname += '_submeans'
            if ADD_SUB_DOTS:
                figname += 'dots'
        if ADD_LMM_COEF: figname += '_lmmCf'
        plt.savefig(os.path.join(get_project_path('figures'),
                                 'final_Q1_2024', figname),
                    dpi=150, facecolor='w',)
        plt.close()

    else:
        plt.show()


# def calc_lmm_boxFeats(
#     dep_values, indep_values, group_values,
#     ZSCORE_VALUES: bool = False,
# ):





    