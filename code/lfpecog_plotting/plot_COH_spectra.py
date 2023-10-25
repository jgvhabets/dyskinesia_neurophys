"""
Plot overview Coherences Spectra
"""

# import public functions
from os.path import join, exists
from os import makedirs
import numpy as np
import matplotlib.pyplot as plt

# import own functions
from lfpecog_analysis.get_SSD_timefreqs import (
    get_coh_tf_per_sub
)
from lfpecog_analysis.ft_processing_helpers import (
    find_select_nearest_CDRS_for_ephys,
    categorical_CDRS, FeatLidClass
)
from lfpecog_plotting.plotHelpers import get_colors
from utils.utils_fileManagement import (get_project_path,
                                        load_class_pickle,
                                        get_avail_ssd_subs)
from lfpecog_plotting.plotHelpers import remove_duplicate_legend
from lfpecog_plotting.plot_descriptive_SSD_PSDs import (
    break_x_axis_psds_ticks
)
from lfpecog_analysis.psd_lid_stats import (
    process_mean_stats, get_binary_p_perHz
)
from lfpecog_analysis.get_acc_task_derivs import(
    select_tf_on_movement_10s
)


def get_COHs_to_plot(COH_type='imag_coh',
                     COH_source='STN_ECOG',
                     SELECT_ON_ACC_RMS = 'EXCL_MOVE',
                     EXCL_UNILAT_IPSI = True,
                     BASELINE_CORRECT=True,
                     DATA_VERSION='v4.0',
                     FT_VERSION='v6',
                     IGNORE_PTS=['011', '104', '106']):
    """
    """
    # get available SUBS
    SUBS = get_avail_ssd_subs(DATA_VERSION=DATA_VERSION,
                          FT_VERSION=FT_VERSION,
                          IGNORE_PTS=IGNORE_PTS)
    
    if BASELINE_CORRECT: cohs_to_plot = {1: [], 2: [], 3: []}
    else: cohs_to_plot = {0: [], 1: [], 2: [], 3: []}

    # if movement selection on RMS is defined, load pickled class for all subjects
    if SELECT_ON_ACC_RMS:
        featLabPath = join(get_project_path('data'),
                           'prediction_data',
                           'featLabelClasses',
                           f'featLabels_ft{FT_VERSION}_Cdrs_StnOnly.P')
        feat10sClass = load_class_pickle(featLabPath,
                                        convert_float_np64=True)
        print(f'10s-feature class loaded for RMS-ACC selection ({featLabPath})')

    for sub in SUBS:
        sub_cats = {0: [], 1: [], 2: [], 3: []}

        (coh_values,
        coh_times,
        coh_freqs) = get_coh_tf_per_sub(
            sub=sub, COH_type=COH_type, COH_source=COH_source,
            DATA_VERSION=DATA_VERSION, FT_VERSION=FT_VERSION)
        if len(coh_times) == 0:
            print(f'skip sub-{sub}')
            continue
        
        ### SELECT OUT ON ACC RMS IF DEFINED HERE ###
        if SELECT_ON_ACC_RMS:
            coh_values, coh_times = select_tf_on_movement_10s(
                sub=sub, feat10sClass=feat10sClass,
                tf_values_arr=coh_values,
                tf_times_arr=coh_times,
                SELECT_ON_ACC_RMS=SELECT_ON_ACC_RMS
            )
            # rms = feat10sClass.ACC_RMS[sub]  # std zscored mean-rms
            # rms_move = (rms > -0.5).astype(int)
            # rms_time = feat10sClass.FEATS[sub].index.values * 60  # in minutes, convert to seconds
            # move_mask = np.zeros_like(coh_times)

            # for i_t, coh_t in enumerate(coh_times):
            #     try: i_rms = np.where(rms_time == coh_t)[0][0]
            #     except: i_rms = np.argmin(abs(rms_time - coh_t))
            #     move_mask[i_t] = rms_move[i_rms]
            # if 'INCL' in SELECT_ON_ACC_RMS: sel_move = move_mask.astype(bool)
            # elif 'EXCL' in SELECT_ON_ACC_RMS: sel_move = ~move_mask.astype(bool)
            # coh_values = coh_values[sel_move, :]
            # coh_times = coh_times[sel_move]

        # get CDRS (non-categroized) (contralat, to getipsilat-excl bool)
        ipsilat_excl_sel, contra_cdrs = find_select_nearest_CDRS_for_ephys(
            sub=sub, ft_times=coh_times / 60,
            side='contralat ecog',  #'contralat ecog'
            cdrs_rater='Jeroen',
            EXCL_UNILAT_OTHER_SIDE_LID=EXCL_UNILAT_IPSI,
        )
        # get CDRS bilat for actual scoring
        bilat_cdrs = find_select_nearest_CDRS_for_ephys(
            sub=sub, ft_times=coh_times / 60,
            side='bilat',  INCL_CORE_CDRS=True,
            cdrs_rater='Jeroen',
        )
        coh_values = coh_values[ipsilat_excl_sel, :]
        coh_times = coh_times[ipsilat_excl_sel]
        coh_cdrs = bilat_cdrs[ipsilat_excl_sel]

        coh_cdrs = categorical_CDRS(
            coh_cdrs, preLID_separate=False,
            preLID_minutes=0,
            cutoff_mildModerate=4,
            cutoff_moderateSevere=8
        )

        # store values in categories dict
        for i_c, c in enumerate(coh_cdrs):
            sub_cats[c].append(coh_values[i_c, :])

        # add mean categories per sub to main dict
        for c in sub_cats:
            if len(sub_cats[c]) == 0: continue
            cat_values = np.array(sub_cats[c]).astype(np.float64)

            if BASELINE_CORRECT and c == 0:
                base_m = np.mean(cat_values, axis=0)
                # base_sd = np.std(cat_values, axis=0)
            elif BASELINE_CORRECT:
                # BASELINE CORRECT AGAINST NO-LID
                cat_values = (np.mean(cat_values, axis=0) - base_m)  / base_m * 100
                cohs_to_plot[c].append(cat_values)
            else:
                cohs_to_plot[c].append(cat_values)
    
    return cohs_to_plot, coh_freqs


def plot_COH_spectra(COH_source, COH_type,
                     cohs_to_plot=False, coh_freqs=False,
                     SELECT_ON_ACC_RMS=False,
                     DATA_VERSION='v4.0',
                     FT_VERSION='v6',
                     PLOT_STD_ERR = True,
                     SMOOTH_PLOT_FREQS = 8,
                     BREAK_X_AX = True,
                     SAVE_PLOT = True,
                     SHOW_PLOT = False,
                     RETURN_AX=False, given_ax=None,):
    """
    """
    if not cohs_to_plot and not coh_freqs:
        cohs_to_plot, coh_freqs = get_COHs_to_plot(
            COH_type=COH_type, COH_source=COH_source,
            SELECT_ON_ACC_RMS=SELECT_ON_ACC_RMS,
        )
    
    fig_name = f'00{COH_source}_{COH_type}_Scaling_bilat_LID'
    ax_title = f'{COH_source}_{COH_type.upper()} during dyskinesia'

    if SELECT_ON_ACC_RMS:
        fig_name += f'_{SELECT_ON_ACC_RMS}'
        ax_title += f'\n({SELECT_ON_ACC_RMS.lower()})'


    cat_names = {1: 'mild', 2: 'moderate', 3: 'severe'}

    if RETURN_AX:
        ax = given_ax
    else:
        fig, ax = plt.subplots(1, 1, figsize=(1 * 6, 6))


    colors = get_colors('Jacoba')
    fsize=14

    for cat in cohs_to_plot:
        # skip empty categories
        if len(cohs_to_plot[cat]) == 0: continue

        COHs = {}
        cohs = np.array(cohs_to_plot[cat])
        COHs['mean'] = np.nanmean(cohs, axis=0)
        COHs['sd'] = np.nanstd(cohs, axis=0)
        if PLOT_STD_ERR: COHs['sd'] = COHs['sd'] / np.sqrt(cohs.shape[0])
        
        # blank freqs irrelevant after SSD
        blank_sel = np.logical_and(coh_freqs > 35, coh_freqs < 60)
        for k in COHs: COHs[k][blank_sel] = np.nan

        # smoothen signal for plot (both mean and stddev)
        if SMOOTH_PLOT_FREQS > 0:
            for k in COHs:
                idx = np.arange(len(COHs[k]))
                s_win = SMOOTH_PLOT_FREQS // 2
                COHs[k] = [np.nanmean(COHs[k][
                    np.logical_and(idx > (i_v - s_win),
                                idx < (i_v + s_win))
                ]) for i_v in idx]
        # n-subjects to add to legend-label
        n_subs_cat = cohs.shape[0]

        # BREAK X AXIS and adjust xticks and labels
        if BREAK_X_AX:
            (COHs,
            xticks,
            xlabels) = break_x_axis_psds_ticks(coh_freqs, COHs)
            # if SHOW_SIGN: ps, _, _ = break_x_axis_psds_ticks(p_freqs, ps)
            x_axis = xticks

        # plot full line
        ax.plot(x_axis, COHs['mean'], lw=5, alpha=.5,
                        label=f'{cat_names[cat]} dyskinesia'
                        f' (n={n_subs_cat})',
                        color=colors[int(cat)], )
        # PLOT VARIANCE SHADING
        ax.fill_between(x=x_axis, y1=COHs['mean'] - COHs['sd'],
                        y2=COHs['mean'] + COHs['sd'],
                        alpha=.3, color=colors[int(cat)], )

    # PLOT pretty
    ax.set_xticks(xticks[::8], size=fsize)
    ax.set_xticklabels(xlabels[::8], fontsize=fsize)

    ax.hlines(y=0, xmin=x_axis[0], xmax=x_axis[-1],
            color='gray', lw=1, alpha=.5,)

    # set LABELS, LEGEND, AXES
    ax.set_title(ax_title, size=fsize, weight='bold',)
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ylabel = 'Coherence (a.u.)'
    ylabel = f'{ylabel[:-6]} %-change vs bilat-no-LID (a.u.)'
    ax.set_ylabel(ylabel, size=fsize,)
    # plot legend without duplicates
    leg_info = ax.get_legend_handles_labels()
    hands, labs = remove_duplicate_legend(leg_info)
    ax.legend(hands, labs, frameon=False,
            fontsize=fsize, loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)

    plt.tight_layout()

    if RETURN_AX:
        return ax

    if SAVE_PLOT:
        path = join(get_project_path('figures'),
                    'ft_exploration',
                    f'data_{DATA_VERSION}_ft_{FT_VERSION}',
                    'descr_COHs')
        if not exists(path): makedirs(path)
        plt.savefig(join(path, fig_name),
                    facecolor='w', dpi=300,)

    if SHOW_PLOT: plt.show()
    else: plt.close()