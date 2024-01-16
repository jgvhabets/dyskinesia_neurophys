"""
Store and Load PSD data for plotting
"""

# import libraries and funcitons
import numpy as np
import os
import json
from dataclasses import dataclass, field
from itertools import compress, product

from utils.utils_fileManagement import (
    get_project_path, get_avail_ssd_subs,
    load_ft_ext_cfg,
    load_class_pickle, save_class_pickle
)
from lfpecog_analysis.prep_movement_psd_analysis import (
    create_ephys_masks
)
from lfpecog_analysis.specific_ephys_selection import (
    select_3d_ephys_moveTaskLid,
    get_ssd_psd_from_array,
    plot_check_ephys_selection
)
from lfpecog_preproc.preproc_import_scores_annotations import (
    get_ecog_side
)

def get_allSpecStates_Psds(
    RETURN_PSD_1sec = True
):
    """
    
    Returns:
        - PSDs: dict with all conditions
        - BLs: baseline class
    """

    

    sources = ['lfp_left', 'lfp_right', 'ecog']
    lid_states = ['no', 'mild', 'moderate', 'severe']

    # define conditions to be loaded as seperate get_selectedEphys classes
    rest_conditions = ['rest_nolidbelow30', 'rest_nolidover30'] + [
        f'rest_{l}lid' for l in lid_states[1:]
    ]
    tap_conditions = [f'tap{s}_{l}lid' for s, l in
                    product(['left', 'right'], lid_states)]
    lidmove_conditions = [
        f'dyskmove{s}only_{l}lid' for s, l in
        product(['left', 'right'], lid_states)
    ] + [f'dyskmoveboth_{l}lid' for l in lid_states]

    conditions = rest_conditions + tap_conditions + lidmove_conditions
    print(f'EXTRACT classes for specific Conditions: {conditions}')

    # get baselines
    BLs = get_selectedEphys(
        STATE_SEL='baseline',
        LOAD_PICKLE=True,
        USE_EXT_HD=True,
        PREVENT_NEW_CREATION=False,
        RETURN_PSD_1sec=RETURN_PSD_1sec,
    )

    PSDs = {
        cond: get_selectedEphys(
            STATE_SEL=cond,
            LOAD_PICKLE=True,
            USE_EXT_HD=True,
            PREVENT_NEW_CREATION=False,
            RETURN_PSD_1sec=RETURN_PSD_1sec,
        ) for cond in conditions
    }

    return PSDs, BLs


@dataclass(init=True,)
class get_selectedEphys:
    STATE_SEL: str
    FT_VERSION: int = 'v6'
    MIN_SEL_LENGTH: int = 5
    RETURN_PSD_1sec: bool = False
    LOAD_PICKLE: bool = True
    SAVE_NEW_PICKLES: bool = True
    PLOT_SEL_DATA_PROCESS: bool = False
    USE_EXT_HD: bool = False
    FORCE_CALC_PSDs: bool = False
    ONLY_CREATE: bool = False
    PREVENT_NEW_CREATION: bool = False
    EXTRACT_FREE: bool = False
    available_states: list = field(default_factory=lambda: [
        f'{s}_{d}lid' for s, d in product(
            ['rest', 'tapleft', 'tapright',
             'dyskmoveboth', 'dyskmoveleftonly',
             'dyskmoverightonly',
             'freenomove', 'freemoverightonly',
             'freemoveleftonly', 'freemoveboth'],
            ['no', 'all', 'mild', 'moderate', 'severe']
        )
    ])
    ecog_sides: dict = field(default_factory= lambda: {})
    DYSK_CUTOFFS: dict = field(default_factory= lambda: 
                               {'mild': (1, 3), 'moderate': (4, 7),
                                'severe': (8, 25)})
    SKIP_NEW_CREATION: list = field(default_factory= lambda: [])
    verbose: bool = False
    """
    Attributes:
        - MIN_SEL_LENGTH: number of seconds required to
            calculate PSDs for a specific selection
    """

    def __post_init__(self,):
        # CHECK AND PREPARE INPUT AND VARIABLES
        self.available_states.extend(['baseline', 'rest_nolidbelow30',
                                      'rest_nolidover30'])
        self.STATE_SEL = self.STATE_SEL.lower()
        assert self.STATE_SEL in self.available_states, (
            f'STATE_SEL "{self.STATE_SEL}" should be in {self.available_states}'
        )
        self.SETTINGS = load_ft_ext_cfg(FT_VERSION=self.FT_VERSION)
        data_path = get_project_path('data')
        if self.USE_EXT_HD:
            data_path = 'D://Research/CHARITE/projects/dyskinesia_neurophys/data/'
        picklepath = os.path.join(
            data_path,
            'windowed_data_classes_10s_0.5overlap',
            'selected_ephys_classes_all'
        )
        states_picklepath = os.path.join(
            data_path,
            'windowed_data_classes_10s_0.5overlap',
            'selected_psd_states'
        )
        
        ### INCLUDE FREE
        if self.EXTRACT_FREE:
            print(f'\n#### ADDED _FREE to FOLDERNAMES')
            picklepath += '_FREE'
            states_picklepath += '_FREE'
        # print(f'\n#### (no FREE folder) TODO: calculate selectedPsdState (MEANS) incl FREE')
        
        if self.RETURN_PSD_1sec: states_picklepath += '_1secArrays'
        
        self.loaded_subs = []

        # GET RELEVANT STATE PSDs per Subject and Source
        for sub in self.SETTINGS['TOTAL_SUBS']:
            # try to load existing states instead of creating/loading full large array files
            if not self.FORCE_CALC_PSDs:
                print(f'...trying to load {self.STATE_SEL, sub} from {states_picklepath}')
                LOADED_BOOL, sub_state_tups = load_sub_states(
                    self.STATE_SEL, sub, states_picklepath,
                    verbose=self.verbose
                )
                if LOADED_BOOL:
                    print(f'...sub class ({sub}) loaded successful')
                    # if succesful, list contains tuples with source, array
                    for src, arr in sub_state_tups:
                        # add adjusted class to current main class
                        setattr(self, f'{src}_{sub}_{self.STATE_SEL}', arr)
                        if 'ecog' in src:
                            self.ecog_sides[sub] = get_ecog_side(sub)

                        if self.verbose: print(f'...added {src}_{sub}_{self.STATE_SEL}')

                    self.loaded_subs.append(sub)
                    if self.verbose: print(f'added states of sub-{sub} to main get class (loaded existing)')

                    self.freqs = np.concatenate([np.arange(4, 35), np.arange(60, 90)])
                    continue

            # CALCULATE PSD PER CONDITION

            # get (load or create) large data-pickle for current subject
            picklename = f'ephys_selections_{sub}.P'
            # load existing data
            if self.LOAD_PICKLE and picklename in os.listdir(picklepath):
                if self.ONLY_CREATE:
                    if self.verbose: print(f'### {picklename} exists in {picklepath}, skipped loading of sub')
                    continue
                if self.verbose: print(f'...loading {picklename} (from {picklepath})')
                sub_class = load_class_pickle(
                    file_to_load=os.path.join(picklepath,
                                              picklename),
                    convert_float_np64=True
                )
                if self.verbose: print(f'...loaded {picklename}')
            # create and save new data
            else:
                if self.verbose: print(f'{picklename} NOT AVAILABLE IN {picklepath}')
                if self.PREVENT_NEW_CREATION: continue  # 
                print(f'...create new PSD_vs_Move_sub class ({picklename})')
                sub_class = PSD_vs_Move_sub(sub=sub,
                                            PLOT_SELECTION_DATA=self.PLOT_SEL_DATA_PROCESS,
                                            SKIP_NEW_CREATION=self.SKIP_NEW_CREATION,  # (currently: create separate FREE files) classes ideally contain ALL; only give list here for debugging
                                            verbose=self.verbose,)
                
                # for saving delete total 3d arrays and save only mean psd arrays
                for src in sub_class.ephys_sources:
                    delattr(sub_class, f'{src}_3d')
                
                if self.SAVE_NEW_PICKLES:
                    save_class_pickle(sub_class,
                                      path=picklepath,
                                      filename=picklename)
                    print(f'...saved new Class {picklename}')


            ### calculate condition-PSDs with loaded SUB_CLASS
            src_INVOL_TODO = {s: True for s in sub_class.ephys_sources}  # bool to calc DYSK MOVE once per source     
            
            # select relevant data (PSDs) for subject
            for src, sel in product(sub_class.ephys_sources,
                                    sub_class.incl_selections):
                # add baseline per source
                # TODO PM ADD sub-012 baseline
                if src in sel and 'baseline' in sel.lower():
                    if 'BASELINE' in self.SKIP_NEW_CREATION: continue
                    # load 2d arr: n-samples, n-bands (only samples for selection)
                    bl_sig = getattr(sub_class, sel).ephys_2d_arr
                    ftemp, psdtemp = get_ssd_psd_from_array(
                        ephys_arr=bl_sig,
                        sfreq=sub_class.fs,
                        SETTINGS=self.SETTINGS,
                        band_names=sub_class.band_names,
                        RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                    )
                    if len(psdtemp) == 0: continue  # array shorter than 1 second
                    np.save(os.path.join(states_picklepath,
                                         f'{sub}_{src}_baseline_'
                                         f'{bl_sig.shape[0]}samples.npy'),
                            psdtemp, allow_pickle=True)
                    setattr(self, f'{src}_{sub}_baseline', psdtemp)

                # all REST WITHOUT DYSKINESIA
                if src in sel and 'REST' in sel and 'lidno' in sel:
                    if 'REST' in self.SKIP_NEW_CREATION: continue
                    
                    for time_SEL, time_code in zip(
                        [getattr(sub_class, sel).time_arr < (30 * 60),
                         getattr(sub_class, sel).time_arr > (30 * 60)],
                        ['below30', 'over30']
                    ):
                        # load 2d arr: n-samples, n-bands (only samples for selection)
                        ftemp, psdtemp = get_ssd_psd_from_array(
                            ephys_arr=getattr(sub_class, sel).ephys_2d_arr[time_SEL.astype(bool), :],
                            sfreq=sub_class.fs,
                            SETTINGS=self.SETTINGS,
                            band_names=sub_class.band_names,
                            RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                        )
                        if len(psdtemp) == 0: continue  # array shorter than 1 second
                        np.save(os.path.join(states_picklepath,
                                            f'{sub}_{src}_rest_nolid{time_code}'
                                            f'_{sum(time_SEL)}samples.npy'),
                                psdtemp, allow_pickle=True)

                
                # get rest with ALL LID
                if src in sel and 'REST' in sel and 'lidlid' in sel:
                    if 'REST' in self.SKIP_NEW_CREATION: continue

                    for lid_code in ['lid', 'mild', 'moderate', 'severe']:
                        if lid_code == 'lid': 
                            lid_sel = np.ones_like(getattr(sub_class, sel).cdrs_arr)
                        else:
                            lid_sel = np.logical_and(
                                getattr(sub_class, sel).cdrs_arr >= self.DYSK_CUTOFFS[lid_code][0],
                                getattr(sub_class, sel).cdrs_arr <= self.DYSK_CUTOFFS[lid_code][1]
                            )
                        lid_sel = getattr(sub_class, sel).ephys_2d_arr[lid_sel.astype(bool), :]
                        ftemp, psdtemp = get_ssd_psd_from_array(
                            ephys_arr=lid_sel,
                            sfreq=sub_class.fs,
                            SETTINGS=self.SETTINGS,
                            band_names=sub_class.band_names,
                            RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                        )
                        if len(psdtemp) == 0: continue  # array shorter than 1 second
                        if lid_code == 'lid': lid_code = 'all'
                        np.save(os.path.join(states_picklepath,
                                            f'{sub}_{src}_dyskRest_{lid_code}lid'
                                            f'_{lid_sel.shape[0]}samples.npy'),
                                psdtemp, allow_pickle=True)
                    
                ## ADD DYSKINETIC MOVEMENTS (only during REST-tasks)
                if src in sel and 'INVOL' in sel and src_INVOL_TODO[src]:
                    if 'INVOLUNTARY' in self.SKIP_NEW_CREATION: continue

                    # set INVOL-todo bool false, only calculate once
                    src_INVOL_TODO[src] = False
                    
                    move_times = {}
                    for s in ['left', 'right']:
                        try:
                            move_times[s] = getattr(sub_class, f'{src}_INVOLUNTARY_move{s}_lidlid').time_arr
                        except:
                            move_times[s] = []  # empty list works in np intersect/in1d
                    # gives boolean array: LEFT time-array NOT IN RIGHT, and viceversa
                    left_only_bool = ~np.in1d(move_times['left'], move_times['right'])  # np.in1d is true if first array value is in second arr
                    right_only_bool = ~np.in1d(move_times['right'], move_times['left'])
                    both_bool_l = ~left_only_bool  # both is based on moveLEFT selection
                    
                    # loop over only-left, only-right and both
                    for move_bool, move_code in zip([both_bool_l, left_only_bool, right_only_bool],
                                                   ['Both', 'LeftOnly', 'RightOnly']):
                        if sum(move_bool) == 0: continue  # skip empty combinations
                        # get corresponding ephys arr and cdrs scores
                        if move_code == 'RightOnly':
                            sel = f'{src}_INVOLUNTARY_moveright_lidlid'
                        else:
                            sel = f'{src}_INVOLUNTARY_moveleft_lidlid'  # both is also based on left
                        temp_ephys = getattr(sub_class, sel).ephys_2d_arr[move_bool, :]
                        temp_cdrs = getattr(sub_class, sel).cdrs_arr[move_bool]
                            
                        # select LID categories based on lidlid (contains all)
                        for lid_code in ['lid', 'mild', 'moderate', 'severe']:
                            if lid_code == 'lid': 
                                lid_sel = np.ones_like(temp_cdrs)
                            else:
                                lid_sel = np.logical_and(
                                    temp_cdrs >= self.DYSK_CUTOFFS[lid_code][0],
                                    temp_cdrs <= self.DYSK_CUTOFFS[lid_code][1]
                                )
                            lid_sel = temp_ephys[lid_sel.astype(bool), :]
                            ftemp, psdtemp = get_ssd_psd_from_array(
                                ephys_arr=lid_sel,
                                sfreq=sub_class.fs,
                                SETTINGS=self.SETTINGS,
                                band_names=sub_class.band_names,
                                RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                            )
                            if len(psdtemp) == 0: continue  # array shorter than 1 second
                            if lid_code == 'lid': lid_code = 'all'
                            np.save(os.path.join(states_picklepath,
                                                f'{sub}_{src}_dyskmove{move_code}_'
                                                f'{lid_code}lid_{lid_sel.shape[0]}samples.npy'),
                                    psdtemp, allow_pickle=True)
                            
                ## add TAPs (voluntary movement)
                if src in sel and '_VOLUN' in sel:
                    if 'VOLUNTARY' in self.SKIP_NEW_CREATION: continue
                    
                    if 'moveboth' in sel or 'lidall' in sel: continue  # skip summary groups
                    
                    tap_side = sel.split('move')[1].split('_')[0]
                    # once add taps without LID and once taps in LID-categories
                    if 'lidno' in sel.lower():
                        sig = getattr(sub_class, sel).ephys_2d_arr
                        ftemp, psdtemp = get_ssd_psd_from_array(
                            ephys_arr=sig,
                            sfreq=sub_class.fs,
                            SETTINGS=self.SETTINGS,
                            band_names=sub_class.band_names,
                            RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                        )
                        if len(psdtemp) == 0: continue  # array shorter than 1 second
                        np.save(os.path.join(states_picklepath,
                                            f'{sub}_{src}_tap{tap_side}_noLid'
                                            f'_{sig.shape[0]}samples.npy'),
                                psdtemp, allow_pickle=True)
                    
                    else:
                        for lid_code in ['mild', 'moderate', 'severe']:
                            lid_sel = np.logical_and(
                                    getattr(sub_class, sel).cdrs_arr >= self.DYSK_CUTOFFS[lid_code][0],
                                    getattr(sub_class, sel).cdrs_arr <= self.DYSK_CUTOFFS[lid_code][1]
                                )
                            lid_sel = getattr(sub_class, sel).ephys_2d_arr[lid_sel.astype(bool), :]
                            ftemp, psdtemp = get_ssd_psd_from_array(
                                ephys_arr=lid_sel,
                                sfreq=sub_class.fs,
                                SETTINGS=self.SETTINGS,
                                band_names=sub_class.band_names,
                                RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                            )
                            if len(psdtemp) == 0: continue  # array shorter than 1 second
                            np.save(os.path.join(states_picklepath,
                                                f'{sub}_{src}_tap{tap_side}_'
                                                f'{lid_code}lid_{lid_sel.shape[0]}samples.npy'),
                                    psdtemp, allow_pickle=True)

                ## extract FREE moments
                if src in sel and 'FREE' in sel:
                    if 'FREE' in self.SKIP_NEW_CREATION: continue
                    if 'lidall' in sel or 'lidlid' in sel: continue  # skip summary groups
                    print(f'\n...new FREE process: {sel}')

                    # take for no move or both: all times, for L/R-moves, take only unilat movement
                    if 'FREENOMOVE' in sel or ('FREEMOVE' in sel and 'moveboth' in sel):
                        move_bool = [True] * len(getattr(sub_class, sel).time_arr)
                        if 'FREENOMOVE' in sel: move_code = 'nomove'
                        elif 'FREEMOVE' in sel: move_code = 'moveboth'

                    else:  # for left, right, both FREEMOVE
                        # get unique side-only times, catch Error if other side is not present
                        own_times = getattr(sub_class, sel).time_arr
                        if 'moveleft' in sel:
                            move_code = 'moveLeftOnly'
                            try:
                                contra_times = getattr(sub_class, sel.replace('moveleft', 'moveright')).time_arr
                            except AttributeError:
                                contra_times = np.array([])
                        elif 'moveright' in sel:
                            move_code = 'moveRightOnly'
                            try:
                                contra_times = getattr(sub_class, sel.replace('moveright', 'moveleft')).time_arr
                            except AttributeError:
                                contra_times = np.array([])
                        # gives boolean array: LEFT time-array NOT IN RIGHT, and viceversa
                        move_bool = ~np.in1d(own_times, contra_times)  # np.in1d is true if first array value is in second arr
                    
                    if sum(move_bool) == 0: continue  # skip empty combinations
                    temp_ephys = getattr(sub_class, sel).ephys_2d_arr[move_bool, :]
                    temp_cdrs = getattr(sub_class, sel).cdrs_arr[move_bool]
                    # CDRS should already be selected in preparation
                    print(f'...FREE CDRS check, LID categs? {sel}: {temp_cdrs}')
                    for templid in ['mild', 'moderate', 'severe']:
                        if templid in sel: lid_code = templid
                    if 'lidno' in sel: lid_code = 'no'
                
                    ftemp, psdtemp = get_ssd_psd_from_array(
                        ephys_arr=temp_ephys,
                        sfreq=sub_class.fs,
                        SETTINGS=self.SETTINGS,
                        band_names=sub_class.band_names,
                        RETURN_PSD_1sec=self.RETURN_PSD_1sec,
                    )
                    if len(psdtemp) == 0: continue  # array shorter than 1 second

                    np.save(os.path.join(states_picklepath,
                                        f'{sub}_{src}_free{move_code}_'
                                        f'{lid_code}lid_{temp_ephys.shape[0]}samples.npy'),
                            psdtemp, allow_pickle=True)

            # ADD SPECIFIC STATE AFTER CREATION
            LOADED_BOOL, sub_state_tups = load_sub_states(
                self.STATE_SEL, sub, states_picklepath
            )
            if LOADED_BOOL:
                # if succesful, list contains tuples with source, array
                for src, arr in sub_state_tups:
                    # add adjusted class to current main class
                    setattr(self, f'{src}_{sub}_{self.STATE_SEL}', arr)
                    print(f'...added {src}_{sub}_{self.STATE_SEL}')
            else:
                print(f'\n##### WARNING: SUBJECT-{sub} could not be '
                      'imported after creating new PSDs')
            self.loaded_subs.append(sub)
            if self.verbose: print(f'CREATED AND ADDED states of sub-{sub} to main get class')




@dataclass(init=True,)
class PSD_vs_Move_sub:
    """
    Loads ephys data and corresponding masks,
    ephys as SOURCE_3d, MASKS as: task_mask,
    coded: 0: rest, 1: tap, 2: free;
    lid_mask, lid_mask_left, lid_mask_right
    (lid is bilat incl axial, l/r are unilat w/o axial)
    """
    sub: str
    CDRS_RATER: str = 'Patricia'
    FT_VERSION: str = 'v6'
    PLOT_SELECTION_DATA: bool = False
    SKIP_NEW_CREATION: list = field(default_factory= lambda: [])
    verbose: bool = False

    def __post_init__(self,):
        self.SETTINGS = load_ft_ext_cfg(FT_VERSION=self.FT_VERSION)

        print(f'adding masks for sub {self.sub}')
        create_ephys_masks(sub=self.sub,
                           FT_VERSION=self.FT_VERSION,
                           ADD_TO_CLASS=True,
                           self_class=self)
        # put bands in 3d structure
        self.band_names = list(self.SETTINGS['SPECTRAL_BANDS'].keys())
        self.ephys_sources = self.ssd_sub.ephys_sources
        self.incl_selections = []
        # loop over available lfp/ecog-left/right
        for src in self.ephys_sources:
            # loop over freq bands
            for i_band, band in enumerate(self.band_names):
                if i_band == 0:
                    src_bands = getattr(getattr(self.ssd_sub, src), band)
                else:
                    src_bands = np.dstack(
                        [src_bands, getattr(getattr(self.ssd_sub, src), band)]
                    )
            # add as 3d array
            setattr(self, f'{src}_3d', src_bands)

            for SEL, move_side, LID_STATE in product(
                ['VOLUNTARY', 'INVOLUNTARY',
                 'REST', 'BASELINE',
                 'FREENOMOVE', 'FREEMOVE'],
                ['both', 'left', 'right'],
                ['no', 'lid', 'all', 'mild', 'moderate', 'severe']
            ):
                if SEL in self.SKIP_NEW_CREATION: continue
                # excl non-relevant combinations
                if SEL == 'INVOLUNTARY' and LID_STATE in ['no', 'all']: continue
                if SEL == 'BASELINE' and not (LID_STATE == 'no' and move_side == 'both'): continue
                if (SEL in ['REST', 'FREENOMOVE']) and move_side != 'both': continue  # no movement only once
                if 'FREE' in SEL and LID_STATE in ['lid', 'all']: continue  # no double summary classes for free
                # if SEL == 'FREEMOVE' and move_side == 'both': continue  # only save per side, not double

                if 'ecog' in src: DYSK_UNI_EXCL = True
                else: DYSK_UNI_EXCL = False

                if 'FREE' in SEL:
                    print(f'\n...START ephys_moveTaskLid for {SEL}, {move_side}, {LID_STATE}')

                (sel_ephys,
                 sel_times,
                 sel_cdrs,
                 sel_task) = select_3d_ephys_moveTaskLid(
                    psdMoveClass=self, ephys_source=src,
                    SEL=SEL, SEL_bodyside=move_side,
                    DYSK_SEL=LID_STATE,
                    DYSK_UNILAT_SIDE=DYSK_UNI_EXCL,
                    EXCL_ECOG_IPSILAT=DYSK_UNI_EXCL,
                    verbose=self.verbose,
                )

                print(
                    f'({src}): selected EPHYS for {SEL}, LID-state: {LID_STATE}, '
                    f'move-side: {move_side}, results in '
                    f'{round(len(sel_times) / self.fs, 1)} secs data\n'
                )
                if len(sel_times) == 0: continue
                if self.PLOT_SELECTION_DATA:
                    plot_check_ephys_selection(
                        sel_times, sel_ephys, sel_cdrs, sel_task
                    )

                # put selected extraction in class
                sel_class = metaSelected_ephysData(
                    sub=self.sub,
                    ephys_source=src,
                    band_names=['theta' if b == 'delta' else b
                                for b in self.band_names],
                    SEL_type=SEL,
                    SEL_move_side=move_side,
                    SEL_LID=LID_STATE,
                    ephys_2d_arr=sel_ephys,
                    time_arr=sel_times,
                    cdrs_arr=sel_cdrs,
                    task_arr=sel_task
                )
                setattr(self,
                        sel_class.SEL_string,
                        sel_class)
                
                print(f'...sub-{self.sub}, {sel_class.SEL_string} added\n')
                self.incl_selections.append(sel_class.SEL_string)

        if 'ssd_sub' in list(vars(self).keys()):  # remove 2d
            delattr(self, 'ssd_sub')
        
        # replace delta band-name with theta (Freq is 4 - 8 Hz)
        self.band_names = ['theta' if b == 'delta' else b
                           for b in self.band_names]
    
                

@dataclass(init=True, repr=True)
class metaSelected_ephysData:
    """
    Class to store selected ephys and meta data,
    only adds name string for selection
    """
    sub: str
    ephys_source: str
    band_names: list
    SEL_type: str
    SEL_move_side: str
    SEL_LID: str
    ephys_2d_arr: any
    time_arr: any
    cdrs_arr: any
    task_arr: any
    

    def __post_init__(self,):

        self.SEL_string = (f'{self.ephys_source}_'
                           f'{self.SEL_type}_'
                           f'move{self.SEL_move_side}_'
                           f'lid{self.SEL_LID}')


def load_sub_states(state, sub, pickled_state_path,
                    verbose: bool = False,):

    sub_files = [f for f in os.listdir(pickled_state_path)
                 if f.endswith('.npy') and f.startswith(sub)]
    sub_state_files = [f for f in sub_files if state in f.lower()]
    if verbose: print(f'for ({sub}, {state}) selected: {sub_state_files}')
    
    # load existing data
    if len(sub_state_files) > 0:
        list_tuples = []

        for f in sub_state_files:
            arr = np.load(os.path.join(pickled_state_path, f),
                          allow_pickle=True)
            
            for s in ['lfp_right', 'lfp_left', 'ecog']:
                if s in f: src = s

            list_tuples.append((src, arr))
    
        return True, list_tuples

    elif len(sub_files) > 0:
        print(f'\n### NO STATE-specific array available '
              f'for {state} in sub-{sub}, other states ALREADY CREATED ###\n')
        return True, []
    
    else:
        print('\n...no available state-sub-arrays found, try new CREATION '
              f'for {state} in sub-{sub}\n')
        return False, None