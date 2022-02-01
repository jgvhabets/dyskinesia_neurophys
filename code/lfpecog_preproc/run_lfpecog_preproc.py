'''
Python script to run full pre-processing pipeline for
neurphysiology data (ECoG + STN LFP) in once
for one specific recording of a patient.
'''

if __name__ == '__main__':
    # makes sure code is only run when script
    # itself is called; not as imported function
    
    # Import available packages and functions
    import os
    import sys
    import json
    import numpy as np
    import csv

    # Import own functions
    import preproc_reref  # works from dysinesia folder

    ospath = os.getcwd()
    syspath = sys.path
    print(f'Pythons __name__ is {__name__}')
    # with open(os.path.join(
    #     'Users/jeroenhabets/Research/CHARITE/projects/dyskinesia_neurophys',
    #     'code', 'TESTwrite.txt'), 'w') as f:
    # with open('TESTwrite.txt', 'w') as f:
    #     f.write(f'Hello file'
    #             f'\nospath: {ospath}.')

