# read X_cross_val_data.pickle
import pickle
from sklearn import linear_model, metrics
import numpy as np
import os
from scipy import stats
from matplotlib import pyplot as plt
import cebra

with open(os.path.join("data", "ACC_data_STN 1.pickle"), "rb") as f:
    acc_ = np.array(pickle.load(f)["ACC_RMS"])

with open(os.path.join("data", "X_cross_val_data_STN.pickle"), "rb") as f:
    X_cross_val_data = pickle.load(f)

sub_ids = X_cross_val_data["sub_ids"]

acc_subs = []
for sub in np.unique(sub_ids):
    idx_sub = np.where(sub_ids == sub)[0]
    acc_subs.append(acc_[idx_sub])

for sub_idx, acc_sub in enumerate(acc_subs):
    plt.figure()
    plt.hist(acc_sub, bins=100)
    plt.title(f"{np.unique(sub_ids)[sub_idx]}")
plt.suptitle("Histogram of movement accelerometer signals")
plt.tight_layout()


plt.figure(figsize=(20, 10))
for sub_idx, acc_sub in enumerate(acc_subs):
    plt.subplot(4, 6, sub_idx + 1)
    plt.hist(acc_sub, bins=100)
    plt.title(f"{np.unique(sub_ids)[sub_idx]}")
plt.suptitle("Histogram of movement accelerometer signals")
plt.tight_layout()


plt.figure(figsize=(20, 10))
for sub_idx, acc_sub in enumerate(acc_subs):
    plt.subplot(4, 6, sub_idx + 1)
    plt.plot(acc_sub)
    plt.title(f"{np.unique(sub_ids)[sub_idx]}")
plt.suptitle("Time-series of movement accelerometer signals")
plt.tight_layout()
