"""
Standard Predictive Plots
"""

# import public packages
from os.path import join
from itertools import product
import matplotlib.pyplot as plt

def plot_confMatrix(
    cm, to_show=True, to_save=False,
    fig_path=None, fig_name=None,
):
    fs=14

    plt.imshow(cm)
    # cbar = plt.colorbar()
    # cbar.set_label('# observations\n', rotation=270, fontsize=fs)
    for x_txt, y_txt in product([0, 1], [0, 1]):
        if cm[x_txt, y_txt] > 450:
            plt.text(x_txt, y_txt, cm[x_txt, y_txt], color='k',
                    ha='center', fontsize=fs, weight='bold',)
        else:    
            plt.text(x_txt, y_txt, cm[x_txt, y_txt], color='w',
                    ha='center', fontsize=fs, weight='bold',)

    plt.xlabel('Predicted Label', weight='bold', fontsize=fs+2)
    plt.xticks([0, 1], labels=['No LID', 'LID'])
    plt.yticks([0, 1], labels=['No LID', 'LID'])
    plt.ylabel('True Label', weight='bold', fontsize=fs+2)

    plt.tick_params(axis='both', labelsize=fs, size=fs)
    plt.tight_layout()

    if to_save:
        plt.savefig(join(fig_path, 'prediction', fig_name),
                    facecolor='w', dpi=300,)
    if to_show: plt.show()
    else: plt.close()
