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
  - `conda install mne_connectivity` (NOT FOR MULTIVARIATE CONN ANALYSES)

  for the mne mvc analyses:
  - use an env: without general installed mne_connectivity
  - conda install --channel=conda-forge mne-base
  - conda isntall xarray
  - clone mne multivariate repo (hackathon_mne_mvc by tsbinns)
    run: `sys.path.append(os.path.join(codepath, "hackathon_mne_mvc"))`
  - clone mvc mne fork: https://github.com/vss245/mne-connectivity
    run (in fork-repo): `pip install --proxy=http://proxy.charite.de:8080 -e .`
        (--proxy necessary behind Charite Firewall), `pip install -e .`
    run: `sys.path.append(os.path.join(codepath, "mne-connectivity"))`


- on Charité environments:
    run in terminal command line:
    - set HTTPS_PROXY=http://proxy.charite.de:8080
    - set HTTP_PROXY=http://proxy.charite.de:8080
    - git config --global http.proxy http://proxy.charite.de:8080


### Code and Files Structure
To use optimally use this repo, a project folder has to be created with
a specific file-structure.
This repo has to be cloned within the desired folder containing other research projects.
Do this via git-GUI-software (GitLab/ GitKraken), or clone as follows:
- change your working directory in the terminal to the general research folder
- give terminal command: git clone https://github.com/jgvhabets/dyskinesia_neurophys.git
- now this repo is cloned and the folders below which are part of the repo are created

|-- General Research folder (can consist multiple research projects)

|-- |-- dyskinesia_neurophys (created as described above; this repo's main folder)

|-- |-- |-- code (the code-folder in this repo)

|-- |-- |-- data (ignored by this repo, added manually)

|-- |-- |-- figures (ignored by this repo, added manually)

All sub-folders within repo/data and repo/figures in which figures and processed data
are stored are created automatically by the functions.

To use the repo in it's current functionality, it should be possible to retrieve the
raw-bids-data via a synced (OneDrive-) folder.


### Workflow and Main-Files
To use this repo as efficient as possible, the following main-files are
running the different processing phases of the workflow. Details on
terminal commands are given within the functions. 

- Preprocessing raw-bids-data
    - main file to run: ´code/lfpecog_preproc/run_lfpecog_preproc.py´
    - supporting files as command-arguments: ´preprocSettings_v2.4.json´

- Transforming preprocessed data into merged-dataframes (ephys + ACC)
    - main file to run: ´code/lfpecog_features.run_mergeDataClass.py´
    - supporting strings as command-arguments: ´ "012" "v3.0"´

- Feature Extraction based on merged-dataframes
    - main file to run: ...
    - supporting string-infos to run as arguments: ...


### Specifications of functions and settings

#### Preprocessing versions

- until v2.X: ECoG rereferencing was performed with common average and STN-LFP
was performed as a bipolar-difference between averaged neighbouring levels
- 3.0:
  - ECoG rerefenced as 'bipolar' differences between neighbouring contacts
  - STN-LFP: 'bipolar' differences between single neighbouring-contacts
      (all segments within neighbouring intra-level segments, and neighbouring
      inter-level segments, two edging rings also with each other)

