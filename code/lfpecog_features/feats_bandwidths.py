'''
Functions to calculate bandwidth-based
spectral features from neurophysiology
data (LFP and ECOG) in ReTune's B04 Dyskinesia Project

Containing:
- bandpass filter
- (relative) bandwidth-peaks using fooof to define
    peak-bandwidth frequencies
- Phase Amplitude Coupling (tensorpac)
'''
# Import general packages and functions
import os
import numpy as np
from scipy import signal

or_wd = os.getcwd()  # get original working directory
# import functions for 1/f aperiodic-comp correction
from fooof import FOOOF
from fooof.analysis import get_band_peak_fm
# returns [CenterFreq, Power, Bandwidth]

# if necessary: change working directory to import tensorpac
if or_wd.split('/')[-1] == 'dyskinesia_neurophys':
    os.chdir(os.path.join(or_wd, 'code/PAC/tensorpac'))
# import functions for Phase-Amplitude-Coupling

os.chdir(or_wd)  # set back work dir