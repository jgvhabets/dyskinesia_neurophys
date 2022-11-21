"""
Functions to assist reading and processing
priorly preprocessed Ephys-data
"""
# Import public packages and functions
import numpy as np
import pandas as pd


def add_detected_acc_states(
    df, detectedMoves,
):
    """
    Adds motor states to full Ephys-DataFrames.
    A binary coding is used and set positive
    from the beginning till the end of tap or
    movement.

    Inputs:
        - df (dataframe): dataframe from one
            ephys group (e.g. .data in subData)
        - detectedMoves (list): subject specific
            lists of detected taps and moves
    
    Returns:
        - new_df: including 4 binary columns
            for tap/move left/right and one
            last column for 'no_move', if all
            4 move-columns are negative.
    
    Raises:
        - KeyError if 'dopa_time' is no column-
            name, nor index
    """
    try:
        dat = df.set_index('dopa_time')
        datkeys = list(dat.keys())
        dattime = dat.index.values
        dat = dat.values
    except KeyError:
        if df.index.name == 'dopa_time':
            datkeys = list(df.keys())
            dattime = df.index.values
            # memory error if notebook has large active memory opened
            dat = df.values
        else:
            raise KeyError('dopa_time not in keys, nor index')

    print('... adding move states to DataFrame ... -> takes a bit...')

    move_keys = [
        'left_tap',
        'right_tap',
        'left_move',
        'right_move'
    ]

    for key in move_keys:
        print('\t...start', key)
        # empty array to fill
        key_col = np.zeros((dat.shape[0], 1))

        # binary fill move-state epochs
        for times in detectedMoves[f'{key}_t']:
            # create index-selection of full tap
            sel = np.logical_and(
                times[0] < dattime,
                dattime < times[-1]
            )
            key_col[sel] = 1  # set to 1
   
        # add move-state column values and key
        dat = np.concatenate([dat, key_col], axis=1)
        datkeys += [key]
    
    # add no-movement column
    no_move = np.nansum(dat[:, -4:], axis=1) == 0
    no_move = np.array([no_move]).T  #.reshape(len(no_move), 1)
    dat = np.concatenate([dat, no_move], axis=1)
    datkeys += ['no_move']
    print('\t...added NO MOVE column')

    # new_df = pd.DataFrame(data=dat, columns=datkeys, index=dattime)  # dopa time as index
    # dopa time as first column, no index set
    dattime = np.array([dattime]).T  # make 2d array for concatenate
    dat = np.concatenate([dattime, dat], axis=1)
    datkeys = ['dopa_time'] + datkeys
    new_df = pd.DataFrame(data=dat, columns=datkeys,)

    print('\t...created new df')

    return new_df