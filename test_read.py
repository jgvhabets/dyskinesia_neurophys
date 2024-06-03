import pickle
import os


with open(
    os.path.join(
        # "d_out_offset_10_dim_4_including_proba_plus_ECoG_only_v8_2904v1.pickle"
        "d_out_offset_10_dim_4_including_proba_plus_ECoG_only_v8_2904v2_incmov.pickle"
    ),
    "rb",
) as f:
    d_out = pickle.load(f)

print(d_out.keys())
