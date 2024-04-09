# read d_out.pickle

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cebra

#"lfpecog_decode_cebra", 
with open(os.path.join("d_out_NEW_FEATURES_offset_10_dim_4.pickle"), "rb") as f:
    d_out = pickle.load(f)

# read X_cross_val_data.pickle
with open(
    os.path.join("data", "X_cross_val_data_ECOG.pickle"), "rb"
) as f:
    X_cross_val_data = pickle.load(f)

# get unique subject ids
sub_ids = np.unique(X_cross_val_data["sub_ids"])


def plot_consistency_maps():
    l1 = d_out["STN_CEBRA_True_scale"]["X_test_emb"]
    l2 = d_out["STN_CEBRA_True_scale"]["y_test_true"]
    l1.pop(7)
    l2.pop(7)
    scores, paris, subjects = cebra.sklearn.metrics.consistency_score(
        embeddings=d_out["STN_CEBRA_True_scale"]["X_test_emb"],
        labels=d_out["STN_CEBRA_True_scale"]["y_test_true"],
        between="datasets",
    )

    fig = plt.figure(figsize=(11, 4))
    ax1 = plt.subplot(121)
    ax1 = cebra.plot_consistency(
        scores,
        pairs=paris,
        datasets=subjects,
        ax=ax1,
        title="ECOG_CEBRA_True_scale",
        colorbar_label=None,
    )


def plot_embedding(plt_train: bool = False):
    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(231, projection="3d")
    ax2 = plt.subplot(232, projection="3d")
    ax3 = plt.subplot(233, projection="3d")
    axs_up = [ax1, ax2, ax3]

    ax1 = plt.subplot(234, projection="3d")
    ax2 = plt.subplot(235, projection="3d")
    ax3 = plt.subplot(236, projection="3d")
    axs_down = [ax1, ax2, ax3]

    for ax, embedding_name in zip(
        axs_up,
        ["STN_CEBRA_True_categ", "STN_CEBRA_True_binary", "STN_CEBRA_True_scale"],
    ):
        if plt_train is True:
            X_emb = d_out[embedding_name]["X_train_emb"][1]
            y_emb = d_out[embedding_name]["y_train_true_l"][1]
        else:
            X_emb = np.concatenate(d_out[embedding_name]["X_test_emb"])
            y_emb = np.concatenate(d_out[embedding_name]["y_test_true"])
        ax = cebra.plot_embedding(
            ax=ax,
            embedding=X_emb,
            embedding_labels=y_emb,
            markersize=10,
            cmap="viridis",
            title=embedding_name,
        )
        ax.axis("off")

    for ax, embedding_name in zip(
        axs_down,
        ["ECOG_CEBRA_True_categ", "ECOG_CEBRA_True_binary", "ECOG_CEBRA_True_scale"],
    ):
        if plt_train is True:
            X_emb = d_out[embedding_name]["X_train_emb"][1]
            y_emb = d_out[embedding_name]["y_train_true_l"][1]
        else:
            X_emb = np.concatenate(d_out[embedding_name]["X_test_emb"])
            y_emb = np.concatenate(d_out[embedding_name]["y_test_true"])
        ax = cebra.plot_embedding(
            ax=ax,
            embedding=X_emb,
            embedding_labels=y_emb,
            markersize=10,
            cmap="viridis",
            title=embedding_name,
        )
        ax.axis("off")

    plt.show()

# get through keys of d_out and store them in a dataframe
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

LM_Flag = False
add_str = ""

for model_name in ["offset_10_dim_4", "offset_model_with_movement"]:  # offset_model ist 

    if model_name == "offset_10_dim_4":
        with open(
            #os.path.join("lfpecog_decode_cebra", ),
            "d_out_NEW_FEATURES_offset_10_dim_4.pickle",
            "rb",
        ) as f:
            d_out = pickle.load(f)
            add_str = "_without_movement_feature"
    elif model_name == "offset_model_with_movement":
        with open(
            #os.path.join("lfpecog_decode_cebra", ),
            "d_out_NEW_FEATURES_REDUCED_WITH_MOVEMENT_offset_10_dim_4.pickle",
            "rb",
        ) as f:
            d_out = pickle.load(f)
            add_str = "_with_movement_feature"

    for idx, key in enumerate(d_out.keys()):
        if "ECOG" in key:
            loc = "ECOG_STN"
        else:
            loc = "STN"
        if "CEBRA_True" not in key:
            LM_Flag = True
        else:
            LM_Flag = False

        if "categ" in key:
            label_method = "categ"
            performance_metric = "balanced accuracy"
        elif "scale" in key:
            label_method = "scale"
            label_method = "mean_absolute_error"
        else:
            label_method = "binary"
            performance_metric = "balanced accuracy"

        model_name_ = "CEBRA" if LM_Flag is False else "LM"
        model_name_ = model_name_ + "_" + add_str
        for sub_idx, sub in enumerate(sub_ids):
            
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "subject": sub,
                                "location": loc,
                                #"model": model,
                                "label_method": label_method,
                                "performance_metric": performance_metric,
                                "performance": d_out[key]["performances"][sub_idx],
                                "model_name": model_name_,
                                "CEBRA" : not LM_Flag,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

# plot comparison s.t. the LM is also included as a model architecture

plot_embedding(plt_train=False)

plt.figure(figsize=(10, 6),)
#df_plt = df.query("CEBRA == False")
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
        legend = True# if idx == 2 else False
    )
    #plt.ylabel("Balanced Accuracy performance")
    plt.title(f"{label_method}")
    if idx == 0:
        plt.ylabel("Balanced Accuracy performance")
    else:
        plt.ylabel("Mean Absolute Error")
    
    if idx == 2:
        plt.title("Regression")
    plt.xlabel("Location")
plt.tight_layout()


# try to order them by mean performances
# Calculate mean performance for each 'model_name' within each 'label_method'
mean_performance = df.groupby(['label_method', 'model_name'])['performance'].mean()

# Sort by 'performance' within each 'label_method'
sorted_order_largest = mean_performance.groupby(level=0, group_keys=False).nlargest(10)
sorted_order_smallest = mean_performance.groupby(level=0, group_keys=False).nsmallest(10)

# Get the sorted 'model_name' for each 'label_method'
sorted_order_largest = sorted_order_largest.reset_index().groupby('label_method')['model_name'].apply(list)
sorted_order_smallest = sorted_order_smallest.reset_index().groupby('label_method')['model_name'].apply(list)

plt.figure(figsize=(10, 6))
for idx, label_method in enumerate(["binary", "categ", "mean_absolute_error"]):
    plt.subplot(1, 3, idx + 1)
    if label_method == "mean_absolute_error":
        sorted_order = sorted_order_smallest
    else:
        sorted_order = sorted_order_largest
    sns.boxplot(
        x="location",
        y="performance",
        data=df.query("label_method == @label_method"),
        palette="viridis",
        showmeans=True,
        hue="model_name",
        hue_order=sorted_order[label_method],  # Use the sorted order
    )
    #if idx != 0:
    #    plt.gca().legend().set_visible(False)
    plt.ylabel("Performance")
    plt.title(f"{label_method}")
    plt.xlabel("Location")
plt.tight_layout()



plt.figure(figsize=(7, 7), dpi=300)
sns.boxplot(
    df.query("label_method != 'mean_absolute_error'"),
    palette="viridis",
    x="label_method",
    y="performance",
    showmeans=True,
    hue="comb_label",
)
plt.ylabel("Balanced Accuracy performance")
plt.title("Performance comparison location and model")
plt.tight_layout()

plt.figure(figsize=(7, 7), dpi=300)
sns.boxplot(
    df.query("label_method == 'mean_absolute_error'"),
    palette="viridis",
    x="label_method",
    y="performance",
    showmeans=True,
    hue="comb_label",
)
plt.ylabel("Balanced Accuracy performance")
plt.title("Performance comparison location and model")
plt.tight_layout()

plt.figure(figsize=(7, 7), dpi=300)
sns.boxplot(
    df.query("label_method != 'mean_absolute_error' and model == 'CEBRA'"),
    palette="viridis",
    x="label_method",
    y="performance",
    showmeans=True,
    hue="emb_dim",
)
plt.ylabel("Balanced Accuracy performance")
plt.title("Performance comparison location and model")
plt.tight_layout()


plt.figure(figsize=(7, 7), dpi=300)
sns.boxplot(
    df.query("label_method == 'mean_absolute_error' and model == 'CEBRA'"),
    palette="viridis",
    x="label_method",
    y="performance",
    showmeans=True,
    hue="emb_dim",
)
plt.ylabel("Mean Absolute Error performance")
plt.title("Performance comparison location and model")
plt.tight_layout()
