"""
Plot features over time, based on full windows.
Windows are +/- 2-3 minutes of data.

WORK IN PROGRESS / OBSOLETE?
"""

# import public packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Preparation
for sub in ['008', '012', '013', '014']:

    # if sub != '013': continue
    
    restarr, restkeys, restWinTimes = ftsMain.get_windows(
        sub_dfs[sub], fs=1600, ch='none',
        winLen_sec=180
    )

    for ephysSource in ['LFP_R', 'LFP_L', 'ECOG']:
    
        plotFts, plotTimes = get_segmentFeatures(
            sub = sub,
            rest_arr = restarr,
            restkeys = restkeys,
            restWinTimes = restWinTimes,
            ephyGroup = ephysSource,
            segLen_sec = .5,
            part_overlap = 0,
        )


#### Time-Freq with full windows

segLen_sec = .5
segLen_n = segLen_sec * fs  # n-samples per segment
part_overlap = 0
nOverlap = part_overlap * fs
nHop = (1 - part_overlap) * segLen_n

col = 'LFP_R_1_2'

icol = np.where(restkeys == col)[0][0]

ch_arr = restarr[:, :, icol]
temp_times = restWinTimes.copy()

winPsds, del_wins = [], []

for i_win in np.arange(ch_arr.shape[0]):

    windat = ch_arr[i_win, :]

    if np.isnan(list(windat)).any():

        # print('remove NaNs')

        win_noNan = windat[~np.isnan(list(windat))]
        
        if len(win_noNan) < segLen_n:

            del_wins.append(i_win)
            continue


    f, ps = signal.welch(
        win_noNan, fs=fs, nperseg=segLen_n,
        noverlap=nOverlap,
    )
    # do not add empty psd-lists to winPsds
    if len(ps) == 0:
        del_wins.append(i_win)
        continue

    winPsds.append(abs(ps))

winPsds = np.array(winPsds, dtype=float)

for i_w in sorted(del_wins, reverse=True):
    del(temp_times[i_w])

plt.figure(figsize=(16, 8))
plt.imshow(
    winPsds.T, cmap='viridis',
    vmin=0, vmax=5e-13, aspect=.25,
)
plt.colorbar()

plt.xlim(0, winPsds.shape[0])
plt.xlabel('Time after LT intake (min)')
plt.xticks(
    range(len(temp_times)),
    labels=np.array(temp_times) / 60,
)

plt.yticks(f)
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 90)

plt.title(f'sub-{sub}, {col} - Rest (full window means)')

plt.show()

# plt.plot(f, abs(ps))
# plt.xlim(0, 90)
# plt.ylim(0, 5e-13)
# plt.title('Segment in once, NaNs removed')
# plt.show()