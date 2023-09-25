"""
Analyze movement voluntary effect
"""
# import functions and packages
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# import own functions
from utils.utils_fileManagement import (get_project_path,
                                        load_ft_ext_cfg)
from lfpecog_plotting.plotHelpers import get_colors
from lfpecog_plotting.plot_descriptive_SSD_PSDs import break_x_axis_psds_ticks
from lfpecog_analysis.prep_movement_psd_analysis import custom_tap_finding


def load_movement_psds(data_version='v4.0', ft_version='v4',):

    # load ft settings
    SETTINGS = load_ft_ext_cfg(f'ftExtr_spectral_{ft_version}.json')
    freqs = SETTINGS['SPECTRAL_BANDS']
    winlen = SETTINGS['WIN_LEN_sec']
    winoverlap = SETTINGS['WIN_OVERLAP_part']
    # define main directory with stored merged data
    results_path = os.path.join(get_project_path('results'), 'features',
                                  f'SSD_feats_broad_{ft_version}',
                                  data_version,
                                  f'windows_{winlen}s_{winoverlap}overlap_tapRest')
    Fs = 2048
    # states select to plot
    states_to_plot = ['rest', 'tap', 'freeMove', 'freeNoMove']

    # get available SUBS
    files = os.listdir(results_path)
    SUBS = np.unique([f.split('_')[0] for f in files])
    n_subs = len(SUBS)
    print(f'available subjects (n={n_subs}): {SUBS}')

    freq_arr = np.arange(4, 91)

    total_psds = {state: [] for state in states_to_plot}

    for sub in SUBS:
        print(f'load sub-{sub}')
        # define lateralities
        for lfp_side, acc_side in zip(['left', 'right',],
                                      ['right', 'left']):
            # empty psd array to fill
            psd_arr_template = np.array([np.nan] * len(freq_arr))
            sub_lfp_psds = {}
            # sub_lfp_psds = {'rest': np.array([np.nan] * len(freq_arr)),
            #                 'tap': np.array([np.nan] * len(freq_arr))}

            try:
                base_lfp = np.load(os.path.join(results_path,
                                            f'{sub}_lfp_{lfp_side}_restSigs.npy')
                )[:, :Fs * 60 * 3]  # base first 3 minutes rest, all bands
            except FileNotFoundError:
                base_lfp = np.load(os.path.join(results_path,
                                            f'{sub}_lfp_{lfp_side}_restNoMoveSigs.npy')
                )[:, :Fs * 60 * 3]  # base first 3 minutes rest, all bands

            # get lfp data
            for state in states_to_plot:
                sub_lfp_psds[state] = psd_arr_template.copy()

                try:
                    lfp_data = np.load(os.path.join(results_path,
                                            f'{sub}_lfp_{lfp_side}_{state}Sigs.npy'))
                except FileNotFoundError:
                    lfp_data = np.load(os.path.join(results_path,
                                            f'{sub}_lfp_{lfp_side}_{state}NoMoveSigs.npy'))
                            # tap_lfp = np.load(os.path.join(results_path,
            #                            f'{sub}_lfp_{lfp_side}_tapSigs.npy'))
                if len(lfp_data) == 0:
                        print(f'ZERO data loaded: sub-{sub}, lfp-{lfp_side}, {state}')
                        continue
                # calculate psds per band
                for i_bw, bw in enumerate(['delta', 'alpha', 'lo_beta',
                                            'hi_beta', 'gamma']):
                    # take first 3 rest-minutes as baseline
                    f, ps_base = welch(base_lfp[i_bw], fs=Fs, nperseg=Fs)
                    f, ps_state = welch(lfp_data[i_bw], fs=Fs, nperseg=Fs)
                    # f, ps_tap =  welch(tap_lfp[i_bw], fs=Fs, nperseg=Fs)

                    if len(ps_state) == 0:
                        print(f'no {bw} data for sub-{sub}, lfp-{lfp_side}, {state}')
                        continue

                    ps_state = (ps_state - ps_base) / ps_base * 100
                    
                    # ps_tap = (ps_tap - ps_base) / ps_base * 100
                    
                    psx_f_sel = np.logical_and(f > SETTINGS['SPECTRAL_BANDS'][bw][0],
                                            f < SETTINGS['SPECTRAL_BANDS'][bw][1])
                    arr_f_sel = np.logical_and(freq_arr > SETTINGS['SPECTRAL_BANDS'][bw][0],
                                            freq_arr < SETTINGS['SPECTRAL_BANDS'][bw][1])
                    sub_lfp_psds[state][arr_f_sel] = ps_state[psx_f_sel]
                    # sub_lfp_psds['tap'][arr_f_sel] = ps_tap[psx_f_sel]
                
                total_psds[state].append(sub_lfp_psds[state])

            # for task in ['rest', 'tap']: total_psds[task].append(sub_lfp_psds[task])
            
            # for key in sub_lfp_psds.keys():
            #     plt.plot(freq_arr, sub_lfp_psds[key], label=key)
            
            # plt.legend(loc='upper left')
            # plt.title(f'Sub-{sub}, LFP-{lfp_side}')
            # plt.show()
    PSDs = {state: np.array(total_psds[state]) for state in total_psds.keys()}

    return PSDs, freq_arr

    # PSD_REST = np.array(total_psds['rest'])
    # PSD_TAP = np.array(total_psds['tap'])

    # print(PSD_REST.shape, PSD_TAP.shape)

    # plt.plot(freq_arr, PSD_REST.mean(axis=0), label='REST')
    # plt.plot(freq_arr, PSD_TAP.mean(axis=0), label='TAP')

    # plt.show()

    # return PSD_REST, PSD_TAP, freq_arr



def plotPSD_rest_vs_tap(PSDs, freqs, n_subs_incl,
                        fsize = 16, data_version='v4.0',
                        fig_name='tap vs rest PSDs'):
    
    colors = list(get_colors().values())
    c_idx = [0, 1, 4, 5]
    colors = [colors[i] for i in c_idx]
    colors[2] = 'orange'
    # colors = ['yellowgreen', 'darkolivegreen',
    #         'darkviolet', 'pink']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.axhline(y=0, xmin=0, xmax=1, color='gray', alpha=.5, lw=1,)

    for i, state in enumerate(PSDs.keys()):
        # REST: PLOT MEAN PSD LINE, AND STDDEV SHADING
        m = np.nanmean(PSDs[state], axis=0)
        sd = np.nanstd(PSDs[state], axis=0)
        plot_f = freqs.copy()

        m, sd, xticks, xlabs = break_x_axis_psds_ticks(
            PSD=m, PSD_sd=sd, tf_freqs=plot_f
        )

        # plot mean line
        ax.plot(xticks, m, lw=4, alpha=.5,
                label=state, color=colors[i], )
        # PLOT VARIANCE SHADING
        ax.fill_between(x=xticks, y1=m - sd, y2=m + sd,
                        alpha=.25, color=colors[i],)

    plt.legend(loc='upper left')

    ylabel = 'Power (a.u.)'
    ylabel = f'{ylabel[:-6]} %-change vs OFF w/o movements (a.u.)'

    ax.set_xticks(xticks[::8], size=fsize)
    ax.set_xticklabels(xlabs[::8], fontsize=fsize)
    ax.set_xlabel('Frequency (Hz)', size=fsize,)
    ax.set_ylabel(ylabel, size=fsize,)

    ax.legend(frameon=False, fontsize=fsize, loc='upper center',
            ncol=2,)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', size=fsize, labelsize=fsize)

    ax.set_title('Tap vs. Rest vs Free'
                 f'\n subthalamic changes (n={n_subs_incl})',
                fontsize=fsize+2, weight='bold',)

    ax.set_ylim(-100, 150)

    plt.tight_layout()

    plt.savefig(os.path.join(get_project_path('figures'), 'ft_exploration',
                             data_version, 'descr_PSDs',
                             f'{fig_name}_n{n_subs_incl}'))
    plt.close()


def plot_overview_tap_detection(
    acc, fsize=14, SAVE_FIG=False,
):

    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # plot both tap data columns
    if acc.sub != '017':
        ax.plot(acc.times/60, acc.data[:, 5], label=acc.colnames[5])
        ax.plot(acc.times/60, acc.data[:, 6], label=acc.colnames[6])
    else:
        _, _, tap_left = custom_tap_finding_017(
                acc, acc_side='left', move_type='tap',
        )
        _, _, tap_right = custom_tap_finding_017(
                acc, acc_side='right', move_type='tap',
        )
        ax.plot(acc.times/60, tap_left, label='TAP left')
        ax.plot(acc.times/60, tap_right, label='TAP right')
        
    ax.plot(acc.times/60, acc.data[:, 4], lw=5, alpha=.6,
            label='task')

    ax.set_ylim(0, 2.5)

    ax.legend(loc='upper right', ncol=3,
            fontsize=fsize,)

    ax.set_title(f'Sub-{acc.sub}: Acc-Tap-Detection',
            loc='left', weight='bold',
            size=fsize,)
    ax.set_xlabel('Time after L-Dopa intake (minutes)',
                  size=fsize,)

    for sp in ['right', 'top']: ax.spines[sp].set_visible(False)

    plt.tick_params(axis='both', size=fsize,
                    labelsize=fsize)
    plt.tight_layout()
    
    if SAVE_FIG:
        fig_name = f'acc_tap_detect_sub{acc.sub}'
        save_path = os.path.join(get_project_path('figures'),
                                'ft_exploration',
                                'movement', 'tap_detect',
                                fig_name)
        plt.savefig(save_path, facecolor='w', dpi=150,)
        plt.close()

    else:
        plt.show()
