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
    For NOW (07.07.22) work with v2!

    PM: *** include differet treshold values for good and
    bad tappers; or check for numbers of peaks
    detected and re-do with lower threshold in case
    of too little peaks ***

    Input:
        - ax_arr: 1d-array of the acc-axis
            which recorded most variation /
            has the largest amplitude range.
        - fs (int): sample freq in Hz
    
    Returns:
        - pos1: impact-positions of method v1
        - impacts: impact-positions of method v2
        (USE METHOD v2 FOR NOW)
    """
    thresh = np.nanmax(uni_arr) * .2
    arr_diff = np.diff(uni_arr)
    df_thresh = np.nanmax(arr_diff) * .35
    pos1 = find_peaks(
        np.diff(uni_arr),
        height=[np.nanmax(uni_arr) * .3, np.nanmax(uni_arr)],
        width=[1, 5],
    )[0]
    pos1 = delete_too_close_peaks(
        acc_ax=uni_arr, peak_pos=pos1,
        min_distance=fs / 5
    )
    
    # v2.0 method
    pos_peaks = find_peaks(
        uni_arr,
        height=(thresh, np.nanmax(uni_arr)),
    )[0]

    # select peaks with surrounding pos- or neg-DIFF-peak
    impact_pos = [np.logical_or(
        any(arr_diff[i - 3:i + 3] < -df_thresh),
        any(arr_diff[i - 3:i + 3] > df_thresh)
    ) for i in pos_peaks]
    impacts = pos_peaks[impact_pos]
    
    # impacts = delete_too_wide_peaks(
    #     acc_ax=uni_arr, peak_pos=impacts,
    #     max_width=fs / 40
    # )
    impacts = delete_too_close_peaks(
        acc_ax=uni_arr, peak_pos=impacts,
        min_distance=fs / 10
    )

    return pos1, impacts


def delete_too_wide_peaks(
    acc_ax, peak_pos, max_width
):
    impact_widths = peak_widths(
        acc_ax, peak_pos, rel_height=0.5)[0]
    sel = impact_widths < max_width
    peak_pos = peak_pos[sel]

    return peak_pos


def delete_too_close_peaks(
    acc_ax, peak_pos, min_distance
):
    """
    Deletes tapping peaks which are too close on each
    other.

    Input:
        - acc_ax (array): uni-axial acc-signal
            from which the peaks are detected
        - peak_pos (array): containing the samples
            on which peaks are detected
        - fs (int): sample frequency
        - min_distance (int): peaks closer too each other
            than this minimal distance (in samples)
            will be removed
    Returns:
        - peak_pos: array w/ selected peak-positions
    """
    pos_diffs = np.diff(peak_pos)
    del_impacts = []
    acc_ax = np.diff(acc_ax)  # use diff as decision selection
    for n, df in enumerate(pos_diffs):
        if df < min_distance:
            
            pos1, pos2 = peak_pos[n], peak_pos[n + 1]
            peak1, peak2 = acc_ax[pos1], acc_ax[pos2]
            if peak1 >= peak2:
                del_impacts.append(n + 1)

            else:
                del_impacts.append(n)
            
            for hop in [2, 3]:  # check distances to 2nd, 3rd
                try:
                    pos1, pos2 = peak_pos[n], peak_pos[n + hop]
                    
                    if (pos2 - pos1) < min_distance:
                        peak1, peak2 = acc_ax[pos1], acc_ax[pos2]

                        if peak1 >= peak2:
                            del_impacts.append(n + hop)
                        else:
                            del_impacts.append(n)
                except IndexError:
                    pass
    
    peak_pos = np.delete(peak_pos, del_impacts)

    return peak_pos