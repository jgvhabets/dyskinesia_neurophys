# read X_cross_val_data.pickle
import pickle
from sklearn import linear_model, metrics
import numpy as np
import os
from scipy import stats
from matplotlib import pyplot as plt
import cebra

dict_thresholds = {
    "008": -0.35,
    "009": -0.8,
    "010": -0.5,
    "012": -1.2,
    "013": -0.7,
    "014": -1.2,
    "016": -1,
    "017": -1,
    "019": -0.7,
    "020": -0.77,
    "021": -0.8,
    "022": -0.7,
    "023": -0.8,
    "101": -1.36,
    "102": -1,
    "103": -1,
    "105": -0.8,
    "107": -1.2,
    "108": -1.3,
    "109": -1.2,
    "110": -1,
}


def compute_balance(cnts):
    # 0 if the classes are perfectly balanced
    # get's higher if unbalanced
    classes, counts = np.unique(cnts, return_counts=True)
    return np.var([c / cnts.shape[0] for c in counts])


def get_cv_folds(
    sub,
    label_method: str = "binary",
    ADD_MOVEMENT_LABELS: str = "EphysMov",
    threshold_mov: float = -0.5,  # -0.5
):
    idx_train = np.where(sub_ids != sub)[0]
    idx_test = np.where(sub_ids == sub)[0]

    X_train = X_all[idx_train]
    X_test = X_all[idx_test]

    if label_method == "binary":
        y_train = y_all_binary[idx_train]
        y_test = y_all_binary[idx_test]
    elif label_method == "scale":
        y_train = y_all_scale[idx_train]
        y_test = y_all_scale[idx_test]
    else:
        y_train = y_all_categ[idx_train]
        y_test = y_all_categ[idx_test]

    if ADD_MOVEMENT_LABELS == "EphysMov":
        y_train = np.concatenate([y_train[:, None], acc_[idx_train][:, None]], axis=1)
        y_test = np.concatenate([y_test[:, None], acc_[idx_test][:, None]], axis=1)
        y_train = np.array([2 * pair[0] + pair[1] for pair in y_train])
        y_test = np.array([2 * pair[0] + pair[1] for pair in y_test])

    elif ADD_MOVEMENT_LABELS == "Mov":
        y_train = acc_[idx_train]
        y_test = acc_[idx_test]

    return X_train, X_test, y_train, y_test


def compute_embedding(X_train, y_train, X_test):
    # 1. Define a CEBRA model
    cebra_model = cebra.CEBRA(
        model_architecture="offset10-model",
        # model_architecture = "offset1-model",
        batch_size=300,
        learning_rate=0.001,
        max_iterations=1000,
        temperature_mode="constant",
        time_offsets=10,
        output_dimension=4,  # 4
        # conditional="time_delta",
        verbose=True,
        device="cuda",
    )

    cebra_model.fit(X_train, y_train)

    X_emb_train = cebra_model.transform(X_train)

    X_emb_test = cebra_model.transform(X_test)

    return X_emb_train, X_emb_test


def plot_data_single_subject(sub):
    idx_train = np.where(sub_ids == sub)[0]
    # plot the features of the first subject and have the x axis to be the feature name and x axis the ft_times
    plt.imshow(stats.zscore(X_all[idx_train].T, axis=0), aspect="auto")
    plt.yticks(np.arange(len(ft_names)), ft_names, ha="right", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_value_each_sub(val_plot, include_time: bool = True):
    plt.figure()
    for sub in np.unique(sub_ids):
        idx = np.where(sub_ids == sub)[0]
        if include_time:
            plt.plot(ft_times_all[idx], val_plot[idx], label=f"{sub}")
        else:
            plt.plot(val_plot[idx], label=f"{sub}")  # ft_times_all[idx]
    plt.legend()
    plt.ylabel("Features")
    plt.xlabel("Time")
    plt.title("Scale across subjects")
    plt.tight_layout()
    plt.show()


USE_CEBRA = True
REDUCE_FEATURES = True
ADD_ACC_FEATURES = False

LIMIT_STN_ONLY_FROM_SAME_SUBJECTS = False

if __name__ == "__main__":
    d_out = {}

    for loc in [
        "STN",
        "ECOGSTN",
        "ECOG",
    ]:
        if loc == "ECOG":
            with open(
                os.path.join("data", "X_cross_val_data_ECOG_v8.pickle"), "rb"
            ) as f:
                X_cross_val_data = pickle.load(f)
                ECOG_idx = [
                    f_idx
                    for f_idx, f in enumerate(X_cross_val_data["ft_names"])
                    if ("ecog" in f or "ECOG" in f) and "coh" not in f
                ]
                X_cross_val_data["X_all"] = X_cross_val_data["X_all"][:, ECOG_idx]
                X_cross_val_data["ft_names"] = [
                    X_cross_val_data["ft_names"][f_idx] for f_idx in ECOG_idx
                ]
            with open(os.path.join("data", "ACC_data_ECOG 1.pickle"), "rb") as f:
                acc_ = np.array(pickle.load(f)["ACC_RMS"])

        elif loc == "STN":
            if LIMIT_STN_ONLY_FROM_SAME_SUBJECTS:
                with open(
                    os.path.join("data", "X_cross_val_data_ECOG.pickle"), "rb"
                ) as f:
                    X_cross_val_data = pickle.load(f)

                with open(os.path.join("data", "ACC_data_ECOG 1.pickle"), "rb") as f:
                    acc_ = np.array(pickle.load(f)["ACC_RMS"])
            else:
                with open(
                    os.path.join("data", "X_cross_val_data_STN.pickle"), "rb"
                ) as f:
                    X_cross_val_data = pickle.load(f)
                with open(os.path.join("data", "ACC_data_STN 1.pickle"), "rb") as f:
                    acc_ = np.array(pickle.load(f)["ACC_RMS"])
        elif loc == "ECOGSTN":
            with open(os.path.join("data", "X_cross_val_data_ECOG.pickle"), "rb") as f:
                X_cross_val_data = pickle.load(f)
            with open(os.path.join("data", "ACC_data_ECOG 1.pickle"), "rb") as f:
                acc_ = np.array(pickle.load(f)["ACC_RMS"])

        X_all = X_cross_val_data["X_all"]
        y_all_binary = X_cross_val_data["y_all_binary"]
        y_all_scale = X_cross_val_data["y_all_scale"]
        y_all_categ = X_cross_val_data["y_all_categ"].astype(int)
        sub_ids = X_cross_val_data["sub_ids"]
        ft_times_all = X_cross_val_data["ft_times_all"]
        ft_names = X_cross_val_data["ft_names"]

        acc_thr = []
        for sub in X_cross_val_data["sub_ids"]:
            idx_ = np.where(X_cross_val_data["sub_ids"] == sub)[0]
            acc_thr.append(acc_[idx_] > dict_thresholds[sub])

        acc_ = np.concatenate(acc_thr)

        if REDUCE_FEATURES:
            new_features_names = []
            new_features_vals = []

            for f_band in [
                "theta",
                "alpha",
                "lo_beta",
                "hi_beta",
                "gammaPeak",
                "gammaBroad",
            ]:
                if loc == "ECOG":
                    # there was only one hemisphere recorded for ECoG
                    idx = [
                        f_idx
                        for f_idx, f in enumerate(ft_names)
                        if "mean_psd" in f and f_band in f
                    ][0]
                    new_features_vals.append(X_all[:, idx])
                    new_features_names.append(f"ecog_{f_band}_mean_psd")
                else:
                    idx_left = np.where(
                        np.array(ft_names) == f"lfp_left_{f_band}_mean_psd"
                    )[0][0]
                    idx_right = np.where(
                        np.array(ft_names) == f"lfp_right_{f_band}_mean_psd"
                    )[0][0]
                    new_features_vals.append(
                        X_all[:, [idx_left, idx_right]].mean(axis=1)
                    )
                    new_features_names.append(f"lfp_both_{f_band}_mean_psd")

            if loc == "ECOGSTN":
                for idx_, feature_name in enumerate(ft_names):
                    if (
                        "ecog" in feature_name and "mean_psd" in feature_name
                    ) or "sq_coh" in feature_name:
                        new_features_names.append(feature_name)
                        new_features_vals.append(X_all[:, idx_])

            X_all = np.array(new_features_vals).T
            ft_names = new_features_names
        if ADD_ACC_FEATURES:
            X_all = np.concatenate([X_all, acc_[:, None]], axis=1)

        # for ADD_MOVEMENT_LABELS in ["Mov", "Ephys", "EphysMov"]:
        #    if ADD_MOVEMENT_LABELS != "Ephys":
        #        mov_labels = ["binary"]
        #    else:
        #        # here the others could be added in case ADD_MOVEMENT_LABELS would be only "Ephys"
        #        mov_labels = ["binary"]  # ["categ", "binary", "scale"]

        # ADD_MOVEMENT_LABELS = "Ephys"
        for ADD_MOVEMENT_LABELS in [
            "EphysMov",
            "Ephys",
            "Mov",
        ]:  # ["Ephys"]:  # ["EphysMov", "Ephys", "Mov"]:
            # for ADD_MOVEMENT_LABELS in ["categ", "binary", "scale"]:
            ADD_MOVEMENT_LABELS = "binary"
            for label_method in [
                "categ",
                "binary",
                "scale",
            ]:  # ["categ", "binary", "scale"]:  # ["binary"]
                print(f"Label method: {label_method}")

                for USE_CEBRA in [
                    True,
                ]:  # False
                    print(f"USE_CEBRA: {USE_CEBRA}")

                    prediction = []
                    performances = []
                    performances_dys = []
                    performances_mov = []
                    cm_l = []
                    y_test_pred_l = []
                    y_test_true_l = []
                    y_test_pred_proba_l = []
                    y_train_pred_proba_l = []

                    X_test_emb_l = []
                    X_train_emb_l = []

                    y_train_pred_l = []
                    y_train_true_l = []

                    for sub in np.unique(sub_ids):
                        X_train, X_test, y_train, y_test = get_cv_folds(
                            sub,
                            label_method=label_method,
                            ADD_MOVEMENT_LABELS=ADD_MOVEMENT_LABELS,
                        )  # categ

                        if USE_CEBRA:
                            X_emb_train, X_emb_test = compute_embedding(
                                X_train, y_train, X_test
                            )
                            X_train = X_emb_train
                            X_test = X_emb_test
                            X_test_emb_l.append(X_test)

                            # cebra.plot_embedding(X_train, y_train, markersize=10, cmap="viridis")
                            # cebra.plot_embedding(X_test, y_test, markersize=10, cmap="viridis")

                        if label_method == "scale":
                            model = linear_model.LinearRegression()
                        else:
                            model = linear_model.LogisticRegression(
                                class_weight="balanced"
                            )

                        model.fit(X_train, y_train)
                        y_test_pred = model.predict(X_test)
                        y_train_pred = model.predict(X_train)

                        if label_method != "scale":
                            y_train_pred_proba = model.predict_proba(X_train)
                            y_test_pred_proba = model.predict_proba(X_test)

                            y_test_pred_proba_l.append(y_test_pred_proba)
                            y_train_pred_proba_l.append(y_train_pred_proba)

                        y_test_pred_l.append(y_test_pred)
                        y_test_true_l.append(y_test)
                        X_train_emb_l.append(X_train)
                        y_train_true_l.append(y_train)
                        prediction.append(y_test_pred)
                        if label_method == "scale" or label_method == "categ":
                            fnc_metric = metrics.mean_absolute_error
                            per = fnc_metric(y_test, y_test_pred)
                        else:
                            fnc_metric = metrics.balanced_accuracy_score
                            per = fnc_metric(y_test, y_test_pred)
                        performances.append(per)
                        if label_method != "scale":
                            cm = metrics.confusion_matrix(y_test, y_test_pred)
                            cm_l.append(cm)
                        if ADD_MOVEMENT_LABELS == "EphysMov":
                            # get binary movement traces back
                            y_true_dys = y_test > 1  #
                            y_pred_dys = y_test_pred > 1
                            per_dys = fnc_metric(y_true_dys, y_pred_dys)
                            performances_dys.append(per_dys)

                            # movemenet is True if y_test is either 1 or 3
                            y_true_mov = np.isin(y_test, [1, 3])
                            y_pred_mov = np.isin(y_test_pred, [1, 3])
                            per_mov = fnc_metric(y_true_mov, y_pred_mov)
                            performances_mov.append(per_mov)

                    key_ = f"{loc}_CEBRA_{USE_CEBRA}_{label_method}_AddMovementLabels_{ADD_MOVEMENT_LABELS}"
                    d_out[key_] = {
                        "prediction": prediction,
                        "performances": performances,
                        "cm": cm_l,
                        "y_test_pred": y_test_pred_l,
                        "y_test_true": y_test_true_l,
                        "X_test_emb": X_test_emb_l,
                        "X_train_emb": X_train_emb_l,
                        "y_train_pred_l": y_train_pred_l,
                        "y_train_true_l": y_train_true_l,
                    }

                    if label_method != "scale":
                        d_out[key_]["y_train_pred_proba"] = y_train_pred_proba_l
                        d_out[key_]["y_test_pred_proba"] = y_test_pred_proba_l
                    if ADD_MOVEMENT_LABELS == "EphysMov":
                        d_out[key_]["performances_dys"] = performances_dys
                        d_out[key_]["performances_mov"] = performances_mov
    # save d_out to pickle
    with open(
        "d_out_offset_10_dim_4_including_proba_plus_ECoG_only_v8_0305_v2_withMovmement.pickle",
        "wb",
    ) as f:
        pickle.dump(d_out, f)

# import pandas as pd
# import seaborn as sb

# balances = [compute_balance(f) for f in d_out[list(d_out.keys())[0]]["y_test_true"]]

# balances =
# df_corr = pd.DataFrame()
# df_corr["class_balances"] = 1 - np.array(balances)
# diffs_ = np.array(d_out[list(d_out.keys())[0]]["performances_dys"]) - np.array(performances)

# df_corr["BA Performance difference\n movement - no movement included"] = diffs_
# plt.figure(figsize=(6, 5), )
# sb.regplot(x="class_balances", y="BA Performance difference\n movement - no movement included", data=df_corr)

# corr_ = np.round(np.corrcoef(df_corr["class_balances"],df_corr["BA Performance difference\n movement - no movement included"]), 2)[0, 1]
# plt.title(
#     f"Correlation of Peformance difference \nwith movement and class balances: {corr_}")
# plt.tight_layout()
