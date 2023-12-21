"""
plot general descriptives
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from lfpecog_analysis.ft_processing_helpers import (
    categorical_CDRS, find_select_nearest_CDRS_for_ephys
)
from lfpecog_plotting.plotHelpers import (
    remove_duplicate_legend
)
from utils.utils_fileManagement import (
    get_project_path, get_avail_ssd_subs
)

def plot_CDRS_distributions():

    SUBS = get_avail_ssd_subs(DATA_VERSION='v4.0', FT_VERSION='v6',)

    CAT = False
    time_dummy =np.arange(0, 90, 5)
    cat_scores = [[0], [1,2,3], [4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14]]
    cat_colors = ['green', 'orange', 'blue', 'purple']
    leg_content = [[], []]
    
    fig, axes = plt.subplots(len(SUBS), 1, figsize=(4, 6),)
    FS = 12

    for i_s, sub in enumerate(SUBS):
        
        cdrs = find_select_nearest_CDRS_for_ephys(
            sub=sub, side='bilat',  
            ft_times=time_dummy,
            INCL_CORE_CDRS=True,
            cdrs_rater='Patricia',
        )
        if CAT:
            cdrs = categorical_CDRS(cdrs,
                                    cutoff_mildModerate=3.5,
                                    cutoff_moderateSevere=7.5,)
        uniq_cdrs, cdrs_counts = np.unique(cdrs, return_counts=True)
        cdrs_counts = cdrs_counts / sum(cdrs_counts)
        # print(sub, uniq_cdrs, cdrs_counts, sum(cdrs_counts))

        for i_cat, cat in enumerate(['None', 'Mild',
                                    'Moderate', 'Severe']):
            cat_sel = [s in cat_scores[i_cat] for s in uniq_cdrs]
            if sum(cat_sel) == 0: continue
            axes[i_s].bar(x=uniq_cdrs[cat_sel] + 1,
                        height=cdrs_counts[cat_sel],  # plus one only for plotting (x-axis) reasons
                        align='center', width=1,
                        color=cat_colors[i_cat],
                        label=cat,)
        axes[i_s].set_xticks([])
        axes[i_s].set_yticks([.5])
        axes[i_s].set_yticklabels([sub], rotation=0, size=FS,)
        leg_hnd_lbs = axes[i_s].get_legend_handles_labels()
        for i in [0, 1]: leg_content[i].extend(leg_hnd_lbs[i])

    axes[-1].set_xticks(np.arange(0, 13, 2) + 1)  # plus one for bar plotting x
    axes[-1].set_xticklabels(np.arange(0, 13, 2))
    axes[-1].set_xlabel('CDRS (sum score)', size=FS,
                        weight='bold',)  # plus one for bar plotting x

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 13.5)
        ax.tick_params(size=FS, labelsize=FS,)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    # single legend
    handles, labels = remove_duplicate_legend(leg_content)
    axes[0].legend(handles, labels, ncol=2, frameon=False,
                loc='lower center', bbox_to_anchor=(0.5, 1.02),
                fancybox=False, fontsize=FS,)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.tight_layout()

    plt.savefig(os.path.join(get_project_path('figures'),
                             'clinical_scores',
                            'CDRS_score_distribution'),
                dpi=300, facecolor='w',)

    plt.show()