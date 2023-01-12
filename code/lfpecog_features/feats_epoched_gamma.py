"""
Run Epoched Gamma Features
"""
# import public packages and functions
from datetime import date
from numpy import array, mean, nan
from scipy.signal import welch

# import own functions
from lfpecog_features.feats_spectral_helpers import select_bandwidths

today = date.today()

def run_epoched_gamma(
    list_mneEpochArrays,
    freq_low: int,
    freq_high: int,
    report: bool = False,
    report_path = None,
):
    """
    Inputs:
        - list_mneEpochArrays: list of 3d-mne-EpochedArrays
            [n-epochs, n-channels, n-times]
    """
    print('\n\tstart epoched gamma function'.upper())
    # create report string to write out at the end
    if report: report = ('\n\n### START OF GAMMA EPOCHING '
        f'({today.year} {today.month} {today.day}) ###')

    # set/ extract variables
    ch_names = list_mneEpochArrays[0].info.ch_names
    print(ch_names)
    if report: report += (
        f'\n\n- present ch_names: {ch_names},')
    
    # empty array for values [n-windows x n-channels]
    values = array([[nan] * len(ch_names)] * len(list_mneEpochArrays))
    print(f'shape empty value array: {values.shape}')

    for w, mne_window in enumerate(list_mneEpochArrays):
        
        for c, ch in enumerate(ch_names):
            epochedSig = mne_window.get_data()[:, c, :]
            f, psd = welch(
                epochedSig, fs=mne_window.info['sfreq'],
                nperseg=512, noverlap=256, axis=1,)
            # get psd of gamma freqs
            ps_gamma, _ = select_bandwidths(psd, f, freq_low, freq_high)
            # take means over all epochs in window
            psd = mean(psd, axis=0)
            ps_gamma = mean(ps_gamma, axis=0)
            # calculate and store relative gamma part
            rel_gamma = sum(ps_gamma) / sum(psd)
            values[w, c] = rel_gamma
    
    if report:
        with open(report_path, 'a') as f:

            f.write(report)
            f.close()


    return values, ch_names

