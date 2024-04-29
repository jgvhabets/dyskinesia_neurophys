# read d_out.pickle

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cebra

if __name__ == "__main__":
    with open(
        os.path.join(
            "d_out_offset_10_dim_4_including_proba_plus_ECoG_only_v8_2904v1.pickle"
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

        if "CEBRA_True" not in key:
            LM_Flag = True
        else:
            LM_Flag = False

        if "categ" in key:
            label_method = "categ"
            performance_metric = "mean_absolute_error"
        elif "scale" in key:
            label_method = "mean_absolute_error"
        else:
            label_method = "binary"
            performance_metric = "balanced accuracy"

        model_name_ = "CEBRA" if LM_Flag is False else "LM"

        for sub_idx, sub in enumerate(sub_ids):
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
                                "performance": d_out[key]["performances"][sub_idx],
                                "model_name": model_name_,
                                "CEBRA": not LM_Flag,
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
    df_plt = df
    for idx, label_method in enumerate(["binary", "categ", "mean_absolute_error"]):
        plt.subplot(1, 3, idx + 1)
        sns.boxplot(
            df_plt.query("label_method == @label_method"),
            palette="viridis",
            x="location",
            y="performance",
            showmeans=True,
            hue="model_name",
            legend=True,  # if idx == 2 else False
            meanprops={
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },  # Change mean marker properties
        )
        # plt.ylabel("Balanced Accuracy performance")
        plt.title(f"{label_method}")
        if idx == 0:
            plt.ylabel("Balanced Accuracy performance")
        else:
            plt.ylabel("Mean Absolute Error")

        if idx == 2:
            plt.title("Regression")
        plt.xlabel("Location")
    plt.suptitle("Performance comparison ECOG/STN/ECOG+STN")
    plt.tight_layout()
    plt.savefig("comp_per_ECOG_STN_ECOGSTN.pdf")
