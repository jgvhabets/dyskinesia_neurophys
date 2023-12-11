"""
Plot Spectral Power or Connectivity metrics
over the course after levodopa intake

run on WIN as:
xxx\dyskinesia_neurophys\code> python -m lfpecog_plotting.plot_group_metrics_over_time
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from lfpecog_analysis.ft_processing_helpers import split_lid_subs
from utils.utils_fileManagement import get_project_path
from lfpecog_analysis.get_SSD_timefreqs import get_all_ssd_timeFreqs
from lfpecog_plotting.plot_descriptive_SSD_PSDs import plot_PSD_vs_DopaTime

DATA_VERSION = 'v4.0'
FT_VERSION = 'v6'

SAVE=True
SHOW=False

SAVE_DATE = '00'
SOURCE='CONN_IPSI'  # CONN_IPSI, STN, ECOG
CONN_METRIC='MIC'  # MIC//TRGC
ZSCORE_FREQS=False
SMOOTH_FREQS=4
LOG_POWER=False
BASELINE_CORRECT=True


subs_LID, subs_noLID = split_lid_subs(
    DATA_VERSION=DATA_VERSION,
    FT_VERSION=FT_VERSION
)
n_subs_incl = len(subs_LID) + len(subs_noLID)

if 'CONN' in SOURCE:
    TFs = get_all_ssd_timeFreqs(
        SUBS=subs_LID+subs_noLID,
        FT_VERSION=FT_VERSION,
        DATA_VERSION=DATA_VERSION,
        GET_CONNECTIVITY=CONN_METRIC.lower(),  # either 'trgc', 'mic', or default False for Psds
    )
else:
    TFs = get_all_ssd_timeFreqs(
        SUBS=subs_LID+subs_noLID,
        FT_VERSION=FT_VERSION,
        DATA_VERSION=DATA_VERSION,
    )


fig_name = f'{SAVE_DATE}_{SOURCE}_PSDs_noLID_LID_vs_DopaTime_n{n_subs_incl}'
if 'CONN' in SOURCE: fig_name = (f'{SAVE_DATE}_{SOURCE}_{CONN_METRIC}_'
                                 f'PSDs_noLID_LID_vs_DopaTime_n{n_subs_incl}')

if BASELINE_CORRECT: fig_name += '_blCorrPrc'
if ZSCORE_FREQS: fig_name += '_Z'
if LOG_POWER: fig_name += '_log'
if SMOOTH_FREQS > 0: fig_name += f'_smooth{SMOOTH_FREQS}'


# plot_PSD_vs_DopaTime(TFs['008']['lfp_left'])
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fsize=20

# if None returned, dont use axes[0] to collect returned output
if SOURCE == 'STN': ax_title = 'Bilateral STNs w/o Dyskinesia'
elif SOURCE == 'ECOG': ax_title = 'ECoG w/o Dyskinesia'
elif 'CONN' in SOURCE: ax_title = f'{CONN_METRIC} ({SOURCE[5:]}) w/o Dyskinesia'

axes[0] = plot_PSD_vs_DopaTime(
    TFs, sel_subs=subs_noLID,
    SOURCE=SOURCE,
    CONN_METRIC=CONN_METRIC,
    BASELINE_CORRECT=BASELINE_CORRECT,
    ZSCORE_FREQS=ZSCORE_FREQS,
    LOG_POWER=LOG_POWER,
    SMOOTH_PLOT_FREQS=SMOOTH_FREQS,
    plt_ax_to_return=axes[0], fsize=fsize, BREAK_X_AX=True,
    ax_title=ax_title
)

if SOURCE == 'STN': ax_title = 'Bilateral STNs with Dyskinesia'
elif SOURCE == 'ECOG': ax_title = 'ECoG with Dyskinesia'
elif 'CONN' in SOURCE: ax_title = f'{CONN_METRIC} ({SOURCE[5:]}) with Dyskinesia'

axes[1] = plot_PSD_vs_DopaTime(
    TFs, sel_subs=subs_LID,
    SOURCE=SOURCE,
    CONN_METRIC=CONN_METRIC,
    LOG_POWER=LOG_POWER,
    SMOOTH_PLOT_FREQS=SMOOTH_FREQS,
    BASELINE_CORRECT=BASELINE_CORRECT,
    ZSCORE_FREQS=ZSCORE_FREQS,
    plt_ax_to_return=axes[1], fsize=fsize, BREAK_X_AX=True,
    ax_title=ax_title
)


# equalize axes
ymin = min([min(ax.get_ylim()) for ax in axes])
ymax = max([max(ax.get_ylim()) for ax in axes])
for ax in axes: ax.set_ylim(ymin, ymax)
for ax in axes: ax.tick_params(axis='both', size=fsize,
                               labelsize=fsize)
plt.tight_layout()

if SAVE:
    path = os.path.join(get_project_path('figures'),
                        'ft_exploration',
                        f'data_{DATA_VERSION}_ft_{FT_VERSION}')
    if 'CONN' in SOURCE: path = os.path.join(path, 'connectivity')
    else: path = os.path.join(path, 'descr_PSDs')
    if not os.path.exists(path): os.makedirs(path)
    plt.savefig(os.path.join(path, fig_name),
                facecolor='w', dpi=300,)
if SHOW: plt.show()
else: plt.close()

