# read d_out.pickle

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cebra


def compute_balance(cnts):
    # 0 if the classes are perfectly balanced
    # get's higher if unbalanced
    classes, counts = np.unique(cnts, return_counts=True)
    return np.var([c / cnts.shape[0] for c in counts])


if __name__ == "__main__":
    GET_PREDICTION_DYS = True

    with open(
        os.path.join(
            "d_out_offset_10_dim_4_including_proba_plus_ECoG_only_v8_2904v2_incmov.pickle"
        ),
        "rb",
    ) as f:
        d_out = pickle.load(f)

    with open(os.path.join("data", "X_cross_val_data_ECOG_v8.pickle"), "rb") as f:
        X_cross_val_data = pickle.load(f)

    sub_ids = np.unique(X_cross_val_data["sub_ids"])

    df = pd.DataFrame(
        columns=[
            "subject",
            "location",
            "label_method",
            "MovPrediction",
            "performance_metric",
            "performance",
            "model_name",
        ]
    )

    for idx, key in enumerate(d_out.keys()):
        if key.startswith("ECOG_"):
            loc = "ECOG"
        elif key.startswith("STN_"):
            loc = "STN"
        elif key.startswith("ECOGSTN_"):
            loc = "ECOGSTN"

        if key.endswith("AddMovementLabels_Mov"):
            mov_prediction = "Mov"
            if GET_PREDICTION_DYS:
                continue
        elif key.endswith("AddMovementLabels_Ephys"):
            mov_prediction = "Ephys"
            if not GET_PREDICTION_DYS:
                continue
        elif key.endswith("AddMovementLabels_EphysMov"):
            mov_prediction = "EphysMov"

        if "CEBRA_True" not in key:
            LM_Flag = True
        else:
            LM_Flag = False

        label_method = "binary"
        performance_metric = "balanced accuracy"

        model_name_ = "CEBRA" if LM_Flag is False else "LM"

        if mov_prediction == "Mov" or mov_prediction == "Ephys":
            per_metric = d_out[key]["performances"]
        if mov_prediction == "EphysMov":
            if GET_PREDICTION_DYS:
                per_metric = d_out[key]["performances_dys"]
            else:
                per_metric = d_out[key]["performances_mov"]

        for sub_idx, sub in enumerate(sub_ids):
            # check here also the label ratio per subject
            y_test = d_out[key]["y_test_true"][sub_idx]
            balance = compute_balance(y_test)

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "subject": sub,
                                "location": loc,
                                # "model": model,
                                "label_method": label_method,
                                "performance_metric": performance_metric,
                                "performance": per_metric[sub_idx],
                                "model_name": model_name_,
                                "CEBRA": not LM_Flag,
                                "mov_prediction": mov_prediction,
                                "balance": balance,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    plt.figure(
        figsize=(10, 6),
    )
    # df_plt = df.query("CEBRA == False")
    if GET_PREDICTION_DYS:
        plt.suptitle("Dyskinesia prediction")
    else:
        plt.suptitle("Movement prediction")
    df_plt = df
    for idx, model_name in enumerate(["LM", "CEBRA"]):
        plt.subplot(1, 2, idx + 1)
        sns.boxplot(
            df_plt.query("model_name == @model_name"),
            palette="viridis",
            x="location",
            y="performance",
            showmeans=True,
            hue="mov_prediction",
            legend=True,  # if idx == 2 else False
            meanprops={
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },  # Change mean marker properties
        )
        # plt.ylabel("Balanced Accuracy performance")
        plt.title(f"{model_name}")
        plt.ylabel("Balanced Accuracy performance")

        if idx == 2:
            plt.title("Regression")
        plt.xlabel("Location")

        plt.ylim(0.4, 0.95)
    plt.tight_layout()
    plt.savefig("Difference four-class vs two-class classification.pdf")

    data_plt = df[df["model_name"] == "CEBRA"].query("location == 'STN'")
    sns.regplot(x="balance", y="performance", data=df_plt, order=2)
