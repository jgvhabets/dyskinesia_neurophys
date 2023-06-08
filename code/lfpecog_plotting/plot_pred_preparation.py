"""
Plotting prediction preparation process
"""

# import public packages
import numpy as np
from os.path import join
from itertools import product
import matplotlib.pyplot as plt

from utils.utils_fileManagement import get_project_path
from lfpecog_plotting.plotHelpers import remove_duplicate_legend

def plot_ft_covariance_matrix(X_all):

    cov_matrix = np.cov(X_all, rowvar=False)

    variances = np.diag(cov_matrix)
    std_devs = np.sqrt(variances)
    # scale cov matrix
    scaled_cov_matrix = cov_matrix / np.outer(std_devs, std_devs)

    mask = np.abs(scaled_cov_matrix) < .7
    scaled_cov_matrix[mask] = 0

    # Plot the covariance matrix
    plt.imshow(scaled_cov_matrix, cmap='RdYlBu',
            vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Covariance Matrix')
    plt.show()


def boxplot_zscored_LID_features(
    subs_list: list, X_total: list,
    y_total_binary: list, ft_names: list,
    ftLabel_dict: dict,
    TO_SAVE_FIG: bool = False,
    figname = 'LID_ssdFeatures_boxplots_indiv'
):
    """
    make boxplots per subject of z-scored
    features (only LID) used for prediction

    Input:
        - subs_list: list with all sub-string codes
        - X_total: list with all arrays of all features per subject
        - y_total_binary: list with corresponding binary LID-labels
    """

    fig, axes = plt.subplots(len(subs_list), 1, figsize=(12, 16))
    fs = 16
    ##### PLOT BOXPLOT OF FEATURES ######
    for i_s, (sub_fts, sub_y_bin) in enumerate(
        zip(X_total, y_total_binary)
    ):
        sub = subs_list[i_s]
        sub_LID_sel = np.array(sub_y_bin).astype(bool)
        sub_LID_fts = sub_fts[sub_LID_sel, :]
        # make lists for boxplot values (only LID-windows) without NaNs, per features
        bp_LID_values_list = [
            list(sub_LID_fts[~np.isnan(sub_LID_fts[:, i_ft]), i_ft])
            for i_ft in np.arange(sub_LID_fts.shape[1])
        ]
        box = axes[i_s].boxplot(bp_LID_values_list)
        plt.setp(box['fliers'], color='gray')
        # plt.setp(box['whiskers'], color='red')

        axes[i_s].axhline(y=0, xmin=0, xmax=24, color='k', alpha=.3)
        for y_line in [-2, 2]:
            axes[i_s].axhline(y=y_line, xmin=0, xmax=24, color='r', alpha=.3)

        axes[i_s].set_ylim(-6, 6)
        axes[i_s].set_ylabel(f'z-scores\nvs no-LID (a.u.)', fontsize=fs)
        axes[i_s].set_title(f'Sub-{sub} (mean unilat. CDRS '
                            f'{round(np.nanmean(ftLabel_dict[sub]), 2)})',
                            weight='bold', fontsize=fs)
        axes[i_s].set_xticklabels(['mx', 'mn', 'cv'] * int(len(ft_names) / 3),
                                fontsize=fs,)

        for side in ['top','right','bottom']:
            axes[i_s].spines[side].set_visible(False)

        ### fill colors
        colors = {
            'alpha': 'yellow',
            'lo_beta': 'lightblue',
            'hi_beta': 'darkblue',
            'midgamma': 'green'
        }
        hatches = {
            'STN': '',
            'ECoG': '//'
        }

        x_fill_list = []
        for x1 in np.arange(.5, len(ft_names) + .5, 3):
            x2 = x1 + 3
            x_fill_list.append([x1, x2])

        for i_x, (src, bw) in  enumerate(product(hatches.keys(), colors.keys())):
            axes[i_s].fill_betweenx(
                y=np.arange(-6, 6), x1=x_fill_list[i_x][0],
                x2=x_fill_list[i_x][1], color=colors[bw], hatch=hatches[src],
                label=f'{src} {bw}', alpha=.2, edgecolor='gray',)

    leg_content = plt.gca().get_legend_handles_labels()
    handles, labels = remove_duplicate_legend(leg_content)
    plt.legend(handles, labels, ncol=4, frameon=False,
            loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=False,
            prop={'weight': 'bold', 'size': fs})

    plt.suptitle('Individual Feature values during Dyskinesia\n',
                 weight='bold', fontsize=fs+4)
    plt.tight_layout()

    if TO_SAVE_FIG:
        plt.savefig(join(get_project_path('figures'), 'ft_exploration', 'SSD', figname),
                    dpi=300, facecolor='w',)
    plt.show()

    print(f'FEATURES X-AXIS: {ft_names}')
