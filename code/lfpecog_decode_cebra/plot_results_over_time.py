import os
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import matplotlib.colors

with open(
    os.path.join("lfpecog_decode_cebra", "X_cross_val_data_STN.pickle"), "rb"
) as f:
    X_cross_val_data = pickle.load(f)

with open(os.path.join("lfpecog_decode_cebra", "d_out.pickle"), "rb") as f:
    d_out = pickle.load(f)

sub_ids = np.unique(X_cross_val_data["sub_ids"])


### PLOT INDIVIDUAL PREDICTIONS OVER TIME


def plot_predictions_over_time(model_name: str = "STN_CEBRA_True_categ"):
    plt.figure(figsize=(17, 12))
    for sub_idx in range(len(sub_ids)):
        plt.subplot(6, 4, sub_idx + 1)
        sub = sub_ids[sub_idx]

        idx_sub = np.where(X_cross_val_data["sub_ids"] == sub)[0]
        times = X_cross_val_data["ft_times_all"][idx_sub]
        offset = 0.00 if model_name == "STN_CEBRA_True_categ" else 0.00
        plt.plot(
            times,
            d_out[model_name]["y_test_pred"][sub_idx] - offset,
            "x",
            label="Prediction",
            markersize=4,  # 3
            alpha=0.5,
            # mew=0,
        )
        plt.plot(
            times,
            d_out[model_name]["y_test_true"][sub_idx],
            "o",
            label="Label",
            markersize=4,
            alpha=0.5,
            mew=0,
        )
        plt.xlabel("Time [min]")
        plt.ylabel("Prediction")
        plt.legend()
        plt.title(sub)
    plt.suptitle("STN CEBRA Model Predictions")
    plt.tight_layout()
    if model_name == "STN_CEBRA_True_categ":
        plt.savefig(
            os.path.join(
                "lfpecog_decode_cebra", "figures", "STN_CEBRA_CATEG_Predictions.pdf"
            )
        )
    else:
        plt.savefig(
            os.path.join(
                "lfpecog_decode_cebra", "figures", "STN_CEBRA_BIN_Predictions.pdf"
            )
        )


plot_predictions_over_time(model_name="STN_CEBRA_True_binary")
plot_predictions_over_time(model_name="STN_CEBRA_True_categ")

# Regression plots
model_name = "STN_CEBRA_True_scale"

plt.figure(figsize=(17, 12))
for sub_idx in range(len(sub_ids)):
    plt.subplot(6, 4, sub_idx + 1)

    df_plt = pd.DataFrame()
    df_plt["y_test_true"] = d_out[model_name]["y_test_true"][sub_idx]
    df_plt["y_test_pred"] = d_out[model_name]["y_test_pred"][sub_idx]

    sb.regplot(
        y="y_test_true",
        x="y_test_pred",
        scatter_kws={"alpha": 0.5},
        data=df_plt,
    )
    plt.title(sub)
    plt.ylabel("True Label")
    plt.xlabel("Prediction")
plt.suptitle("Regression Plots STN CEBRA Model")
plt.tight_layout()

# plot predictions all subjects
plt.figure(figsize=(6, 6), dpi=300)
df_plt = pd.DataFrame()
df_plt["y_test_true"] = np.concatenate(d_out[model_name]["y_test_true"])
df_plt["y_test_pred"] = np.concatenate(d_out[model_name]["y_test_pred"])
sb.regplot(
    y="y_test_true",
    x="y_test_pred",
    scatter_kws={"alpha": 0.5},
    data=df_plt,
)
plt.title("STN CEBRA Model Cobined Regression Predictions")

model_name = "STN_CEBRA_True_categ"


def get_labels(arr_):
    l_return = []
    for i in arr_:
        if i == 0:
            l_return.append("none")
        elif i == 1:
            l_return.append("mild")
        elif i == 2:
            l_return.append("moderate")
        elif i == 3:
            l_return.append("severe")
    return l_return


# plot individual confusion matrices
def get_font_color(value):
    """Get font color based on the contrast with the cell color"""
    norm_value = matplotlib.colors.Normalize(vmin=-3, vmax=3)(value)
    cell_color = plt.cm.viridis(norm_value)
    perceived_brightness = 1 - (
        0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
    )
    return "black" if perceived_brightness < 0.1547 else "white"


def plot_cm(model_name, labels="categ"):
    plt.figure(figsize=(8, 8), dpi=100)  # figsize=(17, 12),
    for sub_idx in range(len(sub_ids)):
        plt.subplot(6, 4, sub_idx + 1)
        cm = d_out[model_name]["cm"][sub_idx]
        plt.imshow(cm)
        # plt.xlabel("Prediction", fontsize=5)
        # plt.ylabel("True", fontsize=5)
        if labels == "categ":
            labels_ = get_labels(np.unique(d_out[model_name]["y_test_pred"][sub_idx]))
        else:
            labels_ = ["none", "dysk"]
        plt.xticks(np.arange(len(labels_)), labels_, rotation=90, fontsize=5)
        plt.yticks(np.arange(len(labels_)), labels_, fontsize=5)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=5)
        plt.title("\n" + sub_ids[sub_idx], fontsize=8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                font_color = get_font_color(cm[i, j])
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color=font_color,
                    fontsize=5,
                )

        if (sub_idx // 4) == 5:  # Adjust this condition based on the number of rows
            plt.xlabel("Prediction", fontsize=5)
            plt.xticks(np.arange(len(labels_)), labels_, rotation=90, fontsize=5)
        else:
            plt.xticks(
                np.arange(len(labels_)), [""] * len(labels_)
            )  # Show ticks but not labels

        if (sub_idx % 4) == 0:  # Adjust this condition based on the number of columns
            plt.ylabel("True", fontsize=5)
            plt.yticks(np.arange(len(labels_)), labels_, fontsize=5)
        else:
            plt.yticks(
                np.arange(len(labels_)), [""] * len(labels_)
            )  # Show ticks but not labels

    plt.suptitle("STN CEBRA Model Confusion Matrices")
    if labels == "categ":
        plt.savefig(
            os.path.join("lfpecog_decode_cebra", "figures", "STN_CEBRA_CATEG_CM.pdf")
        )
    else:
        plt.savefig(
            os.path.join("lfpecog_decode_cebra", "figures", "STN_CEBRA_BIN_CM.pdf")
        )


plot_cm("STN_CEBRA_True_binary", labels="binary")

plot_cm("STN_CEBRA_True_categ", labels="categ")

# mean / summed CM
model_name = "STN_CEBRA_True_binary"
plt.figure(figsize=(8, 4), dpi=300)

plt.subplot(121)
cm_ = np.mean(d_out[model_name]["cm"], axis=0)
plt.imshow(cm_)
plt.xticks([0, 1], ["None", "Dysk"], rotation=90)
plt.yticks([0, 1], ["None", "Dysk"])
plt.xlabel("Prediction")
plt.ylabel("True")
plt.colorbar()
plt.title("Mean confusion matrix")
plt.subplot(122)

cm_ = np.sum(d_out[model_name]["cm"], axis=0)
plt.imshow(cm_)
plt.xticks([0, 1], ["None", "Dysk"], rotation=90)
plt.yticks([0, 1], ["None", "Dysk"])
plt.xlabel("Prediction")
plt.ylabel("True")
plt.colorbar()
plt.title("Summed confusion matrix")
plt.suptitle("STN CEBRA Group Confusion Matrices")


# 0: none,
# 1: mild,
# 2: moderate,
# 3: severe

plt.figure(figsize=(8, 8), dpi=100)  # figsize=(17, 12),
for sub_idx in range(len(sub_ids)):
    plt.subplot(6, 4, sub_idx + 1)

    # plot individual confusion matrices
    plt.imshow(d_out[model_name]["cm"][sub_idx])
    plt.xlabel("Prediction")
    plt.ylabel("True")
    labels_ = get_labels(np.unique(d_out[model_name]["y_test_pred"][sub_idx]))
    plt.xticks(np.arange(len(labels_)), labels_, rotation=90)
    plt.yticks(np.arange(len(labels_)), labels_)
    plt.colorbar()
    plt.title(sub_ids[sub_idx])
plt.suptitle("STN CEBRA Model Confusion Matrices")
plt.tight_layout()
