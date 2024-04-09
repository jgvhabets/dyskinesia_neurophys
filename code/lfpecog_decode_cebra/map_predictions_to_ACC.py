import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cebra

SELECT_ECOG = True

if SELECT_ECOG:
    with open(
        os.path.join("data", "X_cross_val_data_ECOG.pickle"), "rb"
    ) as f:
        X_cross_val_data = pickle.load(f)
else:
    with open(
        os.path.join("data", "X_cross_val_data_STN.pickle"), "rb"
    ) as f:
        X_cross_val_data = pickle.load(f)

with open(os.path.join("d_out_NEW_FEATURES_offset_10_dim_4.pickle"), "rb") as f:
    d_out = pickle.load(f)

if SELECT_ECOG:
    with open(os.path.join("data", "ACC_data_ECOG 1.pickle"), 'rb') as f:
        acc_ = np.array(pickle.load(f)["ACC_RMS"])
else:
    with open(os.path.join("data", "ACC_data_STN 1.pickle"), 'rb') as f:
        acc_ = np.array(pickle.load(f)["ACC_RMS"])


matr_ = []
if SELECT_ECOG:
    model_name = "ECOG_CEBRA_True_binary"
else:
    model_name = "STN_CEBRA_True_binary"
for sub_idx, sub_id in enumerate(np.unique(X_cross_val_data["sub_ids"])):
    idx_ = np.where(X_cross_val_data["sub_ids"] == sub_id)[0]
    label = d_out[model_name]["y_test_true"][sub_idx]
    predict = d_out[model_name]["y_test_pred"][sub_idx]
    true_pr_idx = np.where(label == predict)[0]
    false_pr_idx = np.where(label != predict)[0]

    acc_true = acc_[true_pr_idx].mean()
    acc_false = acc_[false_pr_idx].mean()

    lid_true = label[true_pr_idx].sum() / true_pr_idx.shape[0]
    lid_false = label[false_pr_idx].sum() / false_pr_idx.shape[0]

    matr_.append([[acc_true, acc_false], [lid_true, lid_false]])

matr_ = np.array(matr_)
matr_mean = np.mean(matr_, axis=0)

ax = sns.heatmap(matr_mean, annot=True, cmap="coolwarm", cbar=False)
# set x tick labels: to "True Prediction", "False Prediction"
ax.set_xticklabels(["True Prediction", "False Prediction"])
ax.set_yticklabels(["Activity [z-score]", "LID %"])
if SELECT_ECOG:
    ax.set_title("ECoG+STN Predictions\nMean all subjects")
else:
    ax.set_title("STN Predictions\nMean all subjects")

# make the same plot above but for each subject
plt.figure(figsize=(15, 10))
for sub_idx in range(matr_.shape[0]):

    if SELECT_ECOG:
        ax_ = plt.subplot(3, 5, sub_idx + 1)
    else:
        ax_ = plt.subplot(4, 6, sub_idx + 1)
    ax = sns.heatmap(matr_[sub_idx], annot=True, cmap="coolwarm", cbar=False, ax=ax_)
    # set x tick labels: to "True Prediction", "False Prediction"
    ax.set_xticklabels(["True Prediction", "False Prediction"])
    ax.set_yticklabels(["Activity [z-score]", "LID %"])
    ax.set_title(f"Subject: {sub_idx}")

plt.tight_layout()
plt.show()
 
plt.hist(acc_, bins=200)