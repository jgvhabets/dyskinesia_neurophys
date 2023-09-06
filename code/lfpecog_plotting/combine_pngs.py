"""
Combine PNG images to one image
using matplotlib
"""

# import functions
from os.path import join, exists, dirname
import matplotlib.pyplot as plt
import matplotlib.image as image

from utils.utils_fileManagement import get_project_path, get_onedrive_path

NEW_FIG_NAME = 'JCS_LID_PSD_entrain'

fig, axes = plt.subplots(1, 3, figsize=(24, 6),
                         gridspec_kw={'width_ratios': [2, 1, 1]},
                         )  # 16, 8 for two

fig_dir = get_onedrive_path('onedrive')
fig_dir = join(dirname(fig_dir), 'Bilder')

fig_name = 'oehrn ucsf preprint 2023 cortical gamma entrainment home.png'
im = image.imread(join(fig_dir, fig_name))
axes[0].imshow(im, aspect='auto',)
axes[0].text(.15, .9, 'A', weight='bold',
             fontsize=32, color='k',)


fig_dir = join(get_project_path('figures'),
    	               'ft_exploration',
                       'v4.0', 'descr_PSDs')

fig_name = '2208_ECoG_PSDs_LID_SCALE_n12_blCorr_smooth8_SIGN_onlyMatch.png'
im = image.imread(join(fig_dir, fig_name))
axes[2].imshow(im, aspect='auto',)
axes[2].text(.15, .9, 'C', weight='bold',
             fontsize=32, color='k',)

fig_name = '2208_STN_PSDs_SCALE_LID_n12_blCorr_smooth8_SIGN_onlyMatch.png'
im = image.imread(join(fig_dir, fig_name))
axes[1].imshow(im, aspect='auto',)
axes[1].text(.15, .9, 'B', weight='bold',
             fontsize=32, color='k',)


for ax in axes: ax.axis('off')

plt.tight_layout()

plt.savefig(join(fig_dir, NEW_FIG_NAME),
            dpi=300, facecolor='w',)
plt.close()
