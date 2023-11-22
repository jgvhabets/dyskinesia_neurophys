# read X_cross_val_data.pickle
import pickle
from sklearn import linear_model, metrics
import numpy as np
import os
from scipy import stats
from matplotlib import pyplot as plt
import cebra


def get_cv_folds(sub, label_method: str = "binary"):
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
    
    return X_train, X_test, y_train, y_test

def compute_embedding(X_train, y_train, X_test):

    # 1. Define a CEBRA model
    cebra_model = cebra.CEBRA(
        model_architecture = "offset10-model",
        #model_architecture = "offset1-model",
        batch_size = 300,
        learning_rate = 0.001,
        max_iterations = 1000,
        temperature_mode="constant",
        time_offsets = 10,
        output_dimension = 4,
        #conditional="time_delta",
        verbose = True,
        device = "cuda",
    )

    cebra_model.fit(X_train, y_train)

    X_emb_train = cebra_model.transform(X_train)

    X_emb_test = cebra_model.transform(X_test)

    return X_emb_train, X_emb_test

def plot_data_single_subject(sub):
    idx_train = np.where(sub_ids == sub)[0]
    # plot the features of the first subject and have the x axis to be the feature name and x axis the ft_times
    plt.imshow(stats.zscore(X_all[idx_train].T, axis=0), aspect='auto')
    plt.yticks(np.arange(len(ft_names)), ft_names, ha='right', fontsize=14)
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

if __name__ == "__main__":
    
    d_out = {}

    for loc in ["STN", "ECOG"]:

        if loc == "ECOG":
            with open(os.path.join("lfpecog_decode_cebra", "X_cross_val_data_ECOG_AND_STN.pickle"), 'rb') as f:
                X_cross_val_data = pickle.load(f)
        else:
            with open(os.path.join("lfpecog_decode_cebra", "X_cross_val_data_STN.pickle"), 'rb') as f:
                X_cross_val_data = pickle.load(f)

        X_all = X_cross_val_data['X_all']
        y_all_binary = X_cross_val_data['y_all_binary']
        y_all_scale = X_cross_val_data['y_all_scale']
        y_all_categ = X_cross_val_data["y_all_categ"].astype(int)
        sub_ids = X_cross_val_data['sub_ids']
        ft_times_all = X_cross_val_data['ft_times_all']
        ft_names = X_cross_val_data['ft_names']

        for label_method in ["categ", "binary", "scale"]:
            print(f"Label method: {label_method}")
            
            for USE_CEBRA in [False,True]:
                print(f"USE_CEBRA: {USE_CEBRA}")

                prediction = []
                performances = []
                cm_l = []
                y_test_pred_l = []
                y_test_true_l = []

                X_test_emb_l = []
                X_train_emb_l = []

                y_train_pred_l = []
                y_train_true_l = []

                for sub in np.unique(sub_ids):
                    X_train, X_test, y_train, y_test = get_cv_folds(sub, label_method=label_method)  # categ
                
                    if USE_CEBRA:
                        X_emb_train, X_emb_test = compute_embedding(X_train, y_train, X_test)
                        X_train = X_emb_train
                        X_test = X_emb_test
                        X_test_emb_l.append(X_test)
                    
                        #cebra.plot_embedding(X_train, y_train, markersize=10, cmap="viridis")
                        #cebra.plot_embedding(X_test, y_test, markersize=10, cmap="viridis")

                    if label_method == "scale":
                        model = linear_model.LinearRegression()
                    else:
                        model = linear_model.LogisticRegression(class_weight="balanced")

                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                    y_train_pred = model.predict(X_train)
                    y_test_pred_l.append(y_test_pred)
                    y_test_true_l.append(y_test)
                    X_train_emb_l.append(X_train)
                    y_train_true_l.append(y_train)
                    prediction.append(y_test_pred)
                    if label_method == "scale":
                        per = metrics.mean_absolute_error(y_test, y_test_pred)
                    else:
                        per = metrics.balanced_accuracy_score(y_test, y_test_pred)
                    performances.append(per)
                    if label_method != "scale":
                        cm = metrics.confusion_matrix(y_test, y_test_pred)
                        cm_l.append(cm)
                
                d_out[f"{loc}_CEBRA_{USE_CEBRA}_{label_method}"] = {
                    "prediction": prediction,
                    "performances": performances,
                    "cm": cm_l,
                    "y_test_pred": y_test_pred_l,
                    "y_test_true": y_test_true_l,
                    "X_test_emb": X_test_emb_l,
                    "X_train_emb" : X_train_emb_l,
                    "y_train_pred_l" : y_train_pred_l,
                    "y_train_true_l" : y_train_true_l
                }
        # save d_out to pickle
        with open("d_out_offset_10_dim_4.pickle", 'wb') as f:
            pickle.dump(d_out, f)



    PLT_EXAMPLAR_LOSSES = False
    if PLT_EXAMPLAR_LOSSES:
        with open(os.path.join("lfpecog_decode_cebra", "X_cross_val_data_STN.pickle"), 'rb') as f:
            X_cross_val_data = pickle.load(f)
        
        X_all = X_cross_val_data['X_all']
        y_all_binary = X_cross_val_data['y_all_binary']
        y_all_scale = X_cross_val_data['y_all_scale']
        y_all_categ = X_cross_val_data["y_all_categ"].astype(int)
        
        loss_ = []
        temperature_ = []
        for label_method in ["categ", "binary", "scale"]:
            if label_method == "binary":
                y_all = y_all_binary
            elif label_method == "scale":
                y_all = y_all_scale
            else:
                y_all = y_all_categ

            # 1. Define a CEBRA model
            cebra_model = cebra.CEBRA(
                model_architecture = "offset10-model",
                #model_architecture = "offset1-model",
                batch_size = 500,
                learning_rate = 0.001,
                max_iterations = 3000,
                temperature_mode="auto",
                time_offsets = 5,
                output_dimension = 3,
                #conditional="time_delta",
                verbose = True,
                device = "cuda",
            )

            cebra_model.fit(X_all, y_all)
            loss_.append(cebra_model.state_dict_["loss"])
            temperature_.append(cebra_model.state_dict_["log"]["temperature"])
        plt.figure(figsize=(4,3), dpi=300)
        plt.plot(loss_[0], label="categ")
        plt.plot(loss_[1], label="binary")
        plt.plot(loss_[2], label="scale")
        plt.legend()
        plt.ylabel("InfoNCE Loss")
        plt.xlabel("Iterations")
        plt.tight_layout()

        plt.figure(figsize=(4,3), dpi=300)
        plt.plot(temperature_[0], label="categ")
        plt.plot(temperature_[1], label="binary")
        plt.plot(temperature_[2], label="scale")
        plt.legend()
        plt.ylabel("Temperature")
        plt.xlabel("Iterations")
        plt.tight_layout()