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
Do this via git-GUI-software (GitLab/ GitKraken), or clone as follows:
- change your working directory in the terminal to the general research folder
- give terminal command: git clone https://github.com/jgvhabets/dyskinesia_neurophys.git
- now this repo is cloned and the folders below which are part of the repo are created

|-- General Research folder

|-- |-- dyskinesia_neurophys (created as described above; this repo's main folder)

|-- |-- |-- code (the code-folder in this repo)

|-- |-- |-- data (ignored by this repo, added manually)

|-- |-- |-- figures (ignored by this repo, added manually)

To use the repo in it's current functionality, it should be possible to retrieve the
raw-bids-data via a synced (OneDrive-) folder. 
