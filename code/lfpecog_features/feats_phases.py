"""
Calculate Phase Locking Features
"""

# import public libraries
import numpy as np
from scipy.signal import hilbert


def get_phase_synchrony_index(sig1, sig2):
    """
    Based on Cagnan et al, PNAS 2019:
    https://www.pnas.org/doi/10.1073/pnas.1819975116

    Calculates the difference in phase over time,
    in absolute pi (0 and 2 pi are equal, -1 and 1 pi are equal),
    0.5 doesnot equal -0.5, but the size of difference
    (phase lag) is similar.

    Parameters:
        - sig1, sig2: two filtered uni-dimensional signals
            coming from same period of time with same sampling freq 
    """
    # perform Hilbert transform
    h1 = hilbert(sig1)
    h2 = hilbert(sig2)
    # take phase part of Hilbert (angle) ,values become between -1pi and 1pi
    phase_rad1 = np.angle(h1)
    phase_rad2 = np.angle(h2)
    # get difference between values (1 pi equals -1 pi)
    phase_diff = phase_rad2 - phase_rad1
    # bring all values between -1 pi and 1 pi (-1.5 pi equals 0.5 pi)
    phase_diff[phase_diff < -1*np.pi] = phase_diff[phase_diff < -1*np.pi] + 2*np.pi
    phase_diff[phase_diff > 1*np.pi] = phase_diff[phase_diff > 1*np.pi] - 2*np.pi

    # calculate Phase Synchrony Index
    psi = np.mean(np.real(np.exp(1j * phase_diff)))

    return psi









#h1 = signal.hilbert(ssd08.lfp_left.gamma[20])
# h2 = signal.hilbert(ssd08.lfp_right.gamma[20])
# env = abs(h1)
# # phase = np.imag(h1)
# phase_rad1 = np.angle(h1) / np.pi
# phase_rad2 = np.angle(h2) / np.pi
# phase_diff = phase_rad2 - phase_rad1
# phase_diff[phase_diff < -1] = phase_diff[phase_diff < -1] + 2
# phase_diff[phase_diff > 1] = phase_diff[phase_diff > 1] - 2

# fig, axes = plt.subplots(2, 2, figsize=(12, 6),
#                          sharex='col', sharey='row')
# x1 = (2000, 2000 + 3 * 2048)
# x2 = (2000, 2000 + .5 * 2048)

# for col in [0, 1]:
#     axes[0, col].plot(phase_rad1, color='blue', alpha=.5,)
#     axes[0, col].plot(phase_rad2, color='green', alpha=.5)
#     axes[1, col].plot(abs(phase_diff))

# for row in [0, 1]:
#     axes[row, 0].set_xlim(x1)
#     axes[row, 1].set_xlim(x2)


# axes[0, 0].set_ylabel('phase (pi)', size=16)
# axes[1, 0].set_ylabel('phase difference\n(absolute pi)', size=16)
# axes[1, 0].set_xlabel('time (3 seconds) (samples)', size=16)
# axes[1, 1].set_xlabel('time (0.5 seconds) (samples)', size=16)

# for r, c in product([0, 1], [0, 1]):
#     axes[r, c].tick_params(labelsize=16, size=16)

# plt.suptitle('Phase Synchrony Index (PSI) example: interhem. STN mid-gamma',
#              y=.98, size=16, weight='bold')
# plt.tight_layout()
# plt.savefig(os.path.join(get_project_path('figures'),
#                          'ft_exploration', 'PSI_example_STNs_midGamma'),
#                          dpi=300, facecolor='w',)

# plt.show()