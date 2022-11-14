# dyskinesia_neurophys

Repo to decode dyskinesia presence and severity from cortical and subcortical electrophysiology signals.
Work within ReTune Project B04, workpackage 2, under supervision of Prof. Kühn, Dep. of Movement
Disorders and Neuromodulation at Charité Berlin, in close collaboration with the ICN Lab
from prof. Neumann.

### Requirements:
- python environment created over conda:
 `conda create --ENVNAME python==3.9 numpy pandas scipy scikit-learn jupyter matplotlib h5py`
- additional packages installed:
  - `conda install mne`
  - `conda install mne-bids`


### Code and Files Structure
To use the repo in its best structure, a project folder has to be created with
a specific file-organisation.
This repo has to be cloned within the desired folder containing other research projects.

|-- General Research folder

|-- |-- dyskinesia_neurophys (this repo's main folder)

|-- |-- |-- code (the code-folder in this repo)

|-- |-- |-- data (ignored by this repo)

|-- |-- |-- figures (ignored by this repo)

To use the repo in it's current functionality, it should be possible to retrieve the
raw-bids-data via a synced (OneDrive-) folder. 
