"""
Combine PNG images to one image
using matplotlib
"""

# import functions
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image as image

from utils.utils_fileManagement import get_project_path

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

fig_dir = join(get_project_path('figures'),
    	               'ft_exploration',
                       'v4.0', 'descr_PSDs')

fig_name = '2208_ECoG_PSDs_LID_SCALE_n12_blCorr_smooth8_GAMMA_SIGN_onlyMatch.png'
im = image.imread(join(fig_dir, fig_name))
axes[0].imshow(im, aspect='auto',)

fig_name = '2208_STN_PSDs_SCALE_LID_n12_blCorr_smooth8_GAMMA_SIGN_onlyMatch.png'
im = image.imread(join(fig_dir, fig_name))
axes[1].imshow(im, aspect='auto',)

for ax in axes: ax.axis('off')

plt.tight_layout()

plt.savefig(join(fig_dir, 'B04_PSD_vs_LID_STN_ECOG'),
            dpi=300, facecolor='w',)
plt.close()
