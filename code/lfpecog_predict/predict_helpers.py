"""
Predict Helpers
"""

# import functions
import numpy as np
from scipy.signal import resample

# import cross-validation functions
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# import performance metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay,
    auc, roc_curve, RocCurveDisplay
)


def perform_prediction(
    X, y, groups, cv_method, clf_method,
    random_state: int = 42,
    perform_random_perm: bool = False,
    n_perms: int = 1,
    perm_return_ROC: bool = False,
    perm_return_labels: bool = False,
    return_dict_per_sub: bool = False,
    verbose: bool = False,
):
    # dict to store results per random-permutation
    perm_y_true, perm_y_pred, perm_conf_pred = {}, {}, {}
    perm_tpr, perm_fpr = [], []
    # define indices for individual label shuffling
    sub_indices = [0,]
    for i in np.arange(1, len(groups)):
        if groups[i] != groups[i - 1]: sub_indices.append(i)
    sub_indices.append(len(groups) - 1)

    # loop over permutation numbers
    if not perform_random_perm: n_perms = 1

    # create dict to store subject-specific results
    sub_results = {}

    for i_perm in np.arange(n_perms):
        # set random state    
        np.random.seed(random_state + i_perm)
        if perform_random_perm: y_shf = y.copy()  # create copy to shuffle later
        # get cross-validation splits
        cv_split = get_crossVal_split(
            cv_method=cv_method, X=X, y=y, groups=groups
        )
        # get classifier
        clf = get_classifier(clf_sel=clf_method)

        # dicts to store outcomes per CV fold
        y_pred_dict, y_proba_dict, y_true_dict, conf_scores = {}, {}, {}, {}
        
        # iterate over cross-val loops
        for F, (train_index, test_index) in enumerate(
            cv_split
        ):
            # define training and test data per fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if perform_random_perm:
                # np.random.shuffle(y_train)  # shuffle all labels at once
                for n_sub in np.arange(len(sub_indices) - 1):
                    # shuffle labels per individual
                    np.random.shuffle(
                        y_shf[sub_indices[n_sub]:sub_indices[n_sub + 1]]
                    )
                    y_train = y_shf[train_index]

                

            
            if cv_method == LeaveOneGroupOut:
                test_sub = groups[test_index[0]]
                sub_results[test_sub] = {}
                if verbose:
                    print(f'FOLD # {F}')
                    print(f'\tfold tests sub-{test_sub}')
                    print(f'\t# of samples: train {len(X_train)}, test {len(X_test)}')
            
            # fit model
            clf.fit(X=X_train, y=y_train)        
            # save predictions for posthoc analysis and conf matrix
            y_proba_dict[F] = clf.predict_proba(X=X_test)
            y_pred_dict[F] = clf.predict(X=X_test)
            conf_scores[F] = clf.decision_function(X=X_test)
            y_true_dict[F] = y_test

            # store per subject if defined
            if return_dict_per_sub and cv_method == LeaveOneGroupOut:
                sub_results[test_sub]['proba'] = y_proba_dict[F]
                sub_results[test_sub]['pred'] = y_pred_dict[F]
                sub_results[test_sub]['true'] = y_true_dict[F]

            # show metrics
            if verbose:
                print(
                    f'\n ######## FOLD {F} ########',
                    classification_report(y_test, y_pred_dict[F])
                )

        # merge CROSSVAL outcomes
        y_true_all, y_pred_all, y_pred_conf_all = [], [], []
        for l in y_true_dict.values(): y_true_all.extend(list(l.ravel()))
        for l in y_pred_dict.values(): y_pred_all.extend(list(l.ravel()))
        for l in conf_scores.values(): y_pred_conf_all.extend(list(l.ravel()))

        # directly return the results if there is no permutation ongoing
        if not perform_random_perm:
            if return_dict_per_sub: return sub_results
            else: return y_true_all, y_pred_all, y_pred_conf_all
        
        # gather permutation results in dict
        else:
            perm_y_true[i_perm] = y_true_all
            perm_y_pred[i_perm] = y_pred_all
            perm_conf_pred[i_perm] = y_pred_conf_all

            # calculate False and True Positive Rates for Rec Oper Curve
            print(y_true_all)
            print(X_test.shape)
            fpr, tpr, _ = roc_curve(y_true_all, y_pred_conf_all)
            # resample both for averaging later
            # tpr = resample(tpr, num=100)
            # fpr = resample(fpr, num=100)
            perm_tpr.append(tpr)
            perm_fpr.append(fpr)

    if perm_return_labels: return perm_y_true, perm_y_pred, perm_conf_pred
    elif perm_return_ROC: return perm_tpr, perm_fpr

def get_crossVal_split(
    cv_method: str, X, y,
    n_folds=5, groups=None,
):

    if cv_method == StratifiedKFold:
        cv = cv_method(n_splits=n_folds,)
        cv.get_n_splits(X, y)
        cv_split = cv.split(X, y)

    elif cv_method == LeaveOneGroupOut:
        cv = cv_method()
        cv.get_n_splits(groups=groups)
        cv_split = cv.split(X, y, groups=groups)
    
    return cv_split


def get_classifier(
    clf_sel: str, random_state = 42,
    sv_kernel='linear', lr_solver='lbfgs',
):
    if clf_sel.lower() == 'lda':
        clf = LDA()

    elif clf_sel.lower() == 'logreg':
        clf = LogisticRegression(
            random_state=random_state,
            solver=lr_solver,
        )

    elif clf_sel.lower() == 'svm' or clf_sel.lower() == 'svc':
        clf = SVC(
            C=1.0,
            kernel=sv_kernel,
            class_weight='balanced',
            gamma='scale',  # 'auto' correct for n_features, scale correct for n_features and X.var
            probability=True,
            tol=1e-3,
            random_state=random_state,
        )
        
    elif clf_sel.lower() == 'rf' or clf.lower() == 'randomforest':
        clf = RandomForestClassifier(
            n_estimators=1000,  # 500
            max_depth=None,
            min_samples_split=5,
            max_features='sqrt',
            random_state=random_state,
            class_weight='balanced',
        )

    return clf
