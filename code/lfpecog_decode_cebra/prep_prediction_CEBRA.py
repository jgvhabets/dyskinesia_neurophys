import pickle
import numpy as np
import os

with open(os.path.join("d_out_NEW_FEATURES_offset_10_dim_4.pickle"), "rb") as f:
    d_out = pickle.load(f)

# read X_cross_val_data.pickle
with open(
    os.path.join("data", "X_cross_val_data_ECOG.pickle"), "rb"
) as f:
    X_cross_val_data = pickle.load(f)

print("")