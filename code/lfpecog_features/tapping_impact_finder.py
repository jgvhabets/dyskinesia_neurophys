"""
Finding impact-moments in ACC-traces
"""
# import public packages and function
import numpy as np
from scipy.signal import find_peaks, peak_widths

def find_impacts(uni_arr, fs):
    """
    Function to detect the impact moments in
    (updrs) finger (or hand) tapping tasks.
    Impact-moment is defined as the moment
    where the finger (or hand) lands on the
    thumb (or the leg) after moving down,
    also the 'closing moment'.

    Input:
        - ax_arr: 1d-array of the acc-axis
            which recorded most variation /
            has the largest amplitude range.
    """
    thresh = np.max(uni_arr) * .2
    arr_diff = np.diff(uni_arr)
    df_thresh = np.max(arr_diff) * .35
    pos1 = find_peaks(
        np.diff(uni_arr),
        height=[np.max(uni_arr) * .3, np.max(uni_arr)],
        width=[1, 5],
    )[0]
    
    # v2.0 method
    pos_peaks = find_peaks(
        uni_arr,
        height=(thresh, np.max(uni_arr)),
        # distance=fs * .1,
    )[0]

    # select peaks with surrounding pos- or neg-DIFF-peak
    impact_pos = [np.logical_or(
        any(arr_diff[i - 3:i + 3] < -df_thresh),
        any(arr_diff[i - 3:i + 3] > df_thresh)
    ) for i in pos_peaks]
    impacts = pos_peaks[impact_pos]
    # excl too wide peaks
    impact_widths = peak_widths(
        uni_arr, impacts, rel_height=0.5)[0]
    sel = impact_widths < (fs / 40)  # 25 / 40
    impacts = impacts[sel]
    
    # INCLUDE EXCLUSION OF TOO CLOSE TOPS (PEAKS WITHIN 10 MSEC
    # ): TAKE POINT WITH HIGHEST ACC-PEAK

    # # delete endPeaks which are too close after each other
    # # by starting with std False before np.diff, the diff- 
    # # scores represent the distance to the previous peak
    # tooclose = endPeaks[np.append(
    #     np.array(False), np.diff(endPeaks) < (fs / 6))]
    # for p in tooclose:
    #     i = np.where(endPeaks == p)
    #     endPeaks = np.delete(endPeaks, i)
    #     posPeaks = np.append(posPeaks, p)

    # print(len(pos1))
    # print(len(pos2))
    # print(len(pos2a))

    return pos1, impacts