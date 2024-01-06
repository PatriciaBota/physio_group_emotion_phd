# Import libraries
# from __future__ import absolute_import, division, print_function
import biosppy as bs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, AffinityPropagation, \
    DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import seaborn as sb
import pickle
import numba
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import minmax_scale
from scipy.stats.mstats import gmean
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from sklearn.model_selection import train_test_split
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 7]
sb.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import glob
import h5py
import pyhrv
import os
sep = os.sep
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 7]
sb.set()
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import glob
import h5py
import pyhrv
import os
sep = os.sep
from scipy import stats
import time


def JOANA_dataset(windows_time, SENSOR, LABELS, INC5):
    print("JOANA DATASET")
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    if LABELS == "Valence":
        labels_a = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'rb'))
    elif LABELS == "Arousal":
        labels_a = pickle.load(
            open("input_Joana" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal',
                 'rb'))
    features_names = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels',
                     'rb'))
    for i in range(len(users_df)):
        users_df[i] = minmax_scale(users_df[i])
        for v, j in enumerate(labels_a[i]):
            if INC5:
                if j <= 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
            else:
                if j < 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
    return users_df, labels_a, features_names


def WESAD_dataset(windows_time, SENSOR, LABELS, INC5):
    print("WESAD DATASET")
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_WESAD" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    labels = pickle.load(open("input_WESAD" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels', 'rb'))
    features_names = pickle.load(open("input_WESAD" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels',
                     'rb'))

    # for i in range(len(users_df)):
    #     print(users_df[i].shape)
    #     print(len(labels[i]))

    allowed_labels = [1,2,3,4]
    n_labels, n_data = [], []
    for i in range(len(users_df)):
        # users_df[i] = minmax_scale(users_df[i])
        med = False
        if i <= 7:
            data = pd.read_csv("/Users/patriciabota/DATA/WESAD/S0" + str(i + 2) + "/S" + str(i + 2) + "_quest.csv", sep=";")
        elif i > 7 and i <= 9:
            data = pd.read_csv("/Users/patriciabota/DATA/WESAD/S" + str(i + 2) + "/S" + str(i + 2) + "_quest.csv", sep=";")
        else:
            data = pd.read_csv("/Users/patriciabota/DATA/WESAD/S" + str(i + 3) + "/S" + str(i + 3) + "_quest.csv", sep=";")

        # data["S2"]; data['Unnamed: 2'][10]-[20]
        n_labels.append([])
        n_data.append([])

        if i + 2 >= 12:
            idx = i + 3
        else:
            idx = i + 2
        # print("idx: ", idx)

        if LABELS == "Valence":
            head = "S" + str(idx)
        elif LABELS == "Arousal":
            head = 'Unnamed: 2'

        for j in range(len(labels[i])):
            if labels[i][j] in allowed_labels:
                n_data[i] += [users_df[i][j]]
                if labels[i][j] == 1:
                    n_labels[i] += [int(data[head][16])]  # valence; arousal- 'Unnamed: 2'
                elif labels[i][j] == 2:
                    n_labels[i] += [int(data[head][17])]  # valence
                elif labels[i][j] == 3:
                    n_labels[i] += [int(data[head][18])]  # valence
                elif labels[i][j] == 4 and not med:
                    n_labels[i] += [int(data[head][19])]  # valence
                    med = True
                elif labels[i][j] == 4 and med:
                    n_labels[i] += [int(data[head][20])]  # valence
            else:
                continue

    for i in range(len(users_df)):
        # print(np.array(n_data[i]).shape)
        # print(len(n_labels[i]))
        for v, j in enumerate(n_labels[i]):
            if INC5:
                if j <= 5:
                    n_labels[i][v] = int(0)
                else:
                    n_labels[i][v] = int(1)
            else:
                if j < 5:
                    n_labels[i][v] = int(0)
                else:
                    n_labels[i][v] = int(1)

    for i in range(len(n_data)):
        n_data[i] = minmax_scale(np.nan_to_num(n_data[i]))

    return np.array(n_data), np.array(n_labels), np.array(features_names)


def DEAP_dataset(windows_time, SENSOR, LABELS, INC5):
    print("DEAP DATASET")
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    if LABELS == "Valence":
        labels_a = pickle.load(open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels', 'rb'))
    elif LABELS == "Arousal":
        labels_a = pickle.load(
            open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal',
                 'rb'))
    features_names = pickle.load(open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels',
                     'rb'))

    for i in range(len(users_df)):
        users_df[i] = minmax_scale(users_df[i])
        for v, j in enumerate(labels_a[i]):
            if INC5:
                if j <= 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
            else:
                if j < 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
    for i in range(len(users_df)):
        users_df[i] = minmax_scale(np.nan_to_num(users_df[i]))

    return users_df, labels_a, features_names



def HCI_dataset(windows_time, SENSOR, LABELS, INC5):
    print("HCI DATASET")
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data_2', 'rb'))
    if LABELS == "Valence":
        labels_a = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence_2', 'rb'))
    elif LABELS == "Arousal":
        labels_a = pickle.load(
            open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal_2',
                 'rb'))
    features_names = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels_2',
                     'rb'))
    for i in range(len(users_df)):
        # users_df[i] = minmax_scale(users_df[i])
        for v, j in enumerate(labels_a[i]):
            if INC5:
                if j <= 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
            else:
                if j < 5:
                    labels_a[i][v] = int(0)
                else:
                    labels_a[i][v] = int(1)
                # else:
                #     labels_a[i][v] = int(2)

    # for i in range(len(users_df)):
    #     users_df[i] = minmax_scale(np.nan_to_num(users_df[i]))
    counter = 0
    n_users_df, n_labels_a = [], []
    for i in range(len(users_df)):
        n_d, n_l = [], []
        for v in range(len(labels_a[i])):
            if int(labels_a[i][v]) == int(2):
                counter += 1
                # continue
            else:
                n_d += [users_df[i][v]]
                n_l += [labels_a[i][v]]
        if not len(n_l):
            continue
        n_users_df += [minmax_scale(np.nan_to_num(n_d))]
        n_labels_a += [n_l]
    users_df = np.array(n_users_df)
    labels_a = np.array(n_labels_a)

    print("Removed 5s: ", counter)
    n_users_df, n_l = [], []
    for i in range(len(users_df)):
        if users_df[i].shape[1] != users_df[0].shape[1]:
            continue
        # print(users_df[i].shape)
        n_users_df += [users_df[i]]
        n_l += [labels_a[i]]
    users_df = np.array(n_users_df)
    labels_a = np.array(n_l)

    return users_df, labels_a, features_names



def eight_dataset(windows_time, SENSOR, LABELS, INC5):
    print("EIGHT DATASET")
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    if LABELS == "Valence":
        labels_a = pickle.load(open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'rb'))
    elif LABELS == "Arousal":
        labels_a = pickle.load(
            open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal',
                 'rb'))
    features_names = pickle.load(open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels',
                     'rb'))

    # for i in range(len(users_df)):
    #     users_df[i] = minmax_scale(users_df[i])
    #     for v, j in enumerate(labels_a[i]):
    #         if INC5:
    #             if j <= 5:
    #                 labels_a[i][v] = int(0)
    #             else:
    #                 labels_a[i][v] = int(1)
    #         else:
    #             if j < 5:
    #                 labels_a[i][v] = int(0)
    #             elif j > 5:
    #                 labels_a[i][v] = int(1)
    #             else:
    #                 labels_a[i][v] = int(2)
    # counter = 0
    # for i in range(len(users_df)):
    #     n_d, n_l = [], []
    #     for v in range(len(labels_a[i])):
    #         if labels_a[i][v] == int(2):
    #             counter += 1
    #             continue
    #         n_d += [users_df[i][v]]
    #         n_l += [labels_a[i][v]]
    #     users_df[i] = minmax_scale(np.nan_to_num(n_d))
    #     labels_a[i] = n_l


    for i in range(len(users_df)):
        users_df[i] = minmax_scale(np.nan_to_num(users_df[i]))

    return users_df, labels_a, features_names


def FSE(X_train, y_train, features_descrition, classifier, CV=10):
    """ Performs a sequential forward feature selection.
    Parameters
    ----------
    X_train : array
        Training set feature-vector.
    y_train : array
        Training set class-labels groundtruth.
    features_descrition : array
        Features labels.
    classifier : object
        Classifier.
    Returns
    -------
    FS_idx : array
        Selected set of best features indexes.
    FS_lab : array
        Label of the selected best set of features.
    FS_X_train : array
        Transformed feature-vector with the best feature set.
    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    """
    total_acc, FS_lab, acc_list, FS_idx = [], [], [], []
    X_train = np.array(X_train)

    print("*** Feature selection started ***")
    for feat_idx, feat_name in enumerate(features_descrition):
        try:
            classifier.max_features = 1
        except:
            continue
        acc_list.append(np.mean(cross_val_score(classifier, X_train[:, feat_idx].reshape(-1,1), y_train, cv=CV))*100)

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(features_descrition[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    FS_X_train = X_train[:, curr_acc_idx]
    total_acc.append(last_acc)
    FS_idx.append(curr_acc_idx)
    while 1:
        acc_list = []
        for feat_idx, feat_name in enumerate(features_descrition):
            if feat_name not in FS_lab:
                curr_train = np.column_stack((FS_X_train, X_train[:, feat_idx]))
                acc_list.append(np.mean(cross_val_score(classifier, curr_train, y_train, cv=CV))*100)
            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc <= acc_list[curr_acc_idx]:
            FS_lab.append(features_descrition[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)
            FS_idx.append(curr_acc_idx)
            FS_X_train = np.column_stack((FS_X_train, X_train[:, curr_acc_idx]))
        else:
            print("FINAL Features: " + str(FS_lab))
            print("Number of selected features", len(FS_lab))
            print("Features idx: ", FS_idx)
            print("Acc: ", str(total_acc))
            print("From ", str(X_train.shape[1]), " features to ", str(len(FS_lab)))
            break
    print("*** Feature selection finished ***")

    return np.array(FS_idx), np.array(FS_lab), FS_X_train



def hyperparam_tunning(X_train, X_test, y_train, y_test):
    """ This function performs the classification of the given features using several classifiers. From the obtained results
    the classifier which best fits the data and gives the best result is chosen and the respective confusion matrix is
    showed.
    Parameters
    ----------
    features: array-like
      set of features
    labels: array-like
      features class labelsX_train: array-like
      train set features
    y_train: array-like
      train set labels
    Returns
    -------
    best_clf: best classifier
    best_acc: best accuracy score
    """

    # Classifiers
    print("USING GRID SEARCH")
    names = ["Random Forest", "Decision Tree", "Nearest Neighbors", "SVM", "AdaBoost", "Naive Bayes", "QDA"]

    classifiers = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        LinearSVC(), #svm.SVC(cache_size=7000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    best_acc = 0
    best_clf = None
    for n, c in zip(names, classifiers):
        n_iter_search = 4
        print(n)
        if n == "Nearest Neighbors":
            # specify parameters and distributions to sample from
            param_dist = {"n_neighbors": sp_randint(1, 10),
                          "leaf_size":  sp_randint(1, 10),
                          'p': sp_randint(1, 10)
                         }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        elif n == "Random Forest":
            # specify parameters and distributions to sample from
            if np.array(X_train).shape[1] == 1 or np.array(X_train).shape[1] <= 3:
                max_f = [1]
                min_samples_split = [2]
                min_samples_leaf = [1]
                max_d = [1]
            else:
                if np.array(X_train).shape[1] <= 20:
                    max_f = sp_randint(1, X_train.shape[1])
                    max_d = sp_randint(1, X_train.shape[1])
                    min_samples_split = sp_randint(2, X_train.shape[1])
                    min_samples_leaf = sp_randint(1, X_train.shape[1])
                else:
                    max_f = sp_randint(1, 20)
                    max_d = sp_randint(1, 20)
                    min_samples_split = sp_randint(2, 20)
                    min_samples_leaf = sp_randint(1, 20)
            param_dist = {"max_depth":max_d,
                          "max_features": max_f,
                          "min_samples_split": min_samples_split,
                          "min_samples_leaf": min_samples_leaf,
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"],
                          "n_estimators": sp_randint(5, 20)}
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        # elif n == 'SVM':
        #     Cs = [0.001, 0.01, 0.1, 1, 10]
        #     gammas = [0.001, 0.01, 0.1, 1]
        #     #C_range = 10. ** np.arange(-3, 8)
        #     #gamma_range = 10. ** np.arange(-5, 4)
        #     param_dist = {'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr'], 'C': Cs,
        #                   'gamma': gammas, 'kernel': ['linear', 'poly', 'sigmoid', 'rbf']}  # 'poly', 'rbf', 'sigmoid'
        #     # run randomized search
        #     grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
        #     grid.fit(X_train, y_train)
        #     y_test_predict = grid.predict(X_test)
        #     acc = accuracy_score(y_test, y_test_predict)*100
        #     best_est = grid.best_estimator_
        elif n == 'Decision Tree':
            param_dist = {"criterion": ["gini", "entropy"],
                          'splitter': ['best', 'random'],
                          "min_samples_split": sp_randint(2, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "max_depth": sp_randint(1, 20)
                          }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        else:
            # Train the classifier
            X_data = np.vstack((X_train, X_test))
            y_data = np.hstack((y_train, y_test))
            acc = np.mean(cross_val_score(c, X_data, y_data, cv=2)*100)
            best_est = c
        print("Accuracy (%): " + str(acc) + '%')
        print('-----------------------------------------')
        if acc > best_acc:
            best_acc = acc
            best_clf = best_est

    print('******** Best Classifier: ' + str(best_clf) + ' ********')
    print("Accuracy: ", best_acc)

    return best_clf, best_acc
def remove_redundant(data, type = 'flat'):
    from scipy.stats import pearsonr
    data = np.vstack(data)

    if type == 'flat':
        remove_idx = [idx for idx in range(len(data.T)) if len(np.unique(data[:,idx])) <= 1]
    elif type == 'corr':
        remove_idx = [idx for idx in range(1, len(data.T))
                      if pearsonr(data.T[idx-1],data.T[idx])[0] >= 0.9]

    return remove_idx

dataset = "EIGHT"
LABELS = "Valence"
# LABELS = "Arousal"
TYPE = "HP"
INC5 = 0
WIND = 40

print("LABELS: ", LABELS)
if dataset == "JOANA":
    modalities_list = ["EDAhand", "EDAfingers", "ECG", "BVP", "Resp"]  # JOANA
elif dataset == "WESAD":
    modalities_list = ["EDA", "ECG", "BVP", "Resp"]  # WESAD
elif dataset == "DEAP":
    modalities_list = ["EDA", "BVP", "Resp"]  # DEAP
elif dataset == "HCI":
    modalities_list = ["EDA", "ECG", "Resp"]  # HCI
elif dataset == "EIGHT":
    modalities_list = ["EDA", "BVP", "Resp"]  # HCI

if dataset == "JOANA":
    if LABELS == "Arousal":
        c = svm.SVC(C=1, cache_size=7000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
        FS_IDX = [0]
    else:
        c = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=3, max_features=2, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        FS_IDX = [377, 1174, 840, 56]
elif dataset == "WESAD":
    if LABELS == "Arousal":
        c = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        FS_IDX = [633, 521, 802]
    else:
        c = svm.SVC(C=0.01, cache_size=7000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)
        FS_IDX = [633, 522, 524, 494]
elif dataset == "DEAP":
    if LABELS == "Arousal":
        c = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                  store_covariance=False, tol=0.0001)
        FS_IDX = [677, 429, 163, 187, 583]
    else:
        c = GaussianNB(priors=None, var_smoothing=1e-09)
        FS_IDX = [618, 212, 410, 439, 225, 531]
elif dataset == "HCI":
    if not INC5:
        if LABELS == "Arousal":
            c = GaussianNB(priors=None, var_smoothing=1e-09)
            FS_IDX =  [687, 173, 682, 677, 428, 403, 581, 81, 31, 453, 71]
        else:
            c = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
            FS_IDX = [396, 208, 703, 14]
    else:
        if LABELS == "Arousal":
            c = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
            FS_IDX = [395, 22, 597, 102, 411, 612, 133, 77, 615, 671, 596, 396, 66, 215, 249]
        else:
            c = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
            FS_IDX = [93, 407, 133]

elif dataset == "EIGHT":
    if LABELS == "Arousal":
        c = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=17, min_samples_split=15,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
        FS_IDX = [578, 101]
    else:
        c = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=3, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=11,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        FS_IDX = [98, 389, 453, 542, 30]


data_p_user, features_names = [], []
for m, modality in enumerate(modalities_list):  # iterate over different modalities
    print("Running for modality: " + modality)
    if dataset == "WESAD":
        _data_p_user, labers_p_user, _features_names = WESAD_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "DEAP":
        _data_p_user, labers_p_user, _features_names = DEAP_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "JOANA":
        _data_p_user, labers_p_user, _features_names = JOANA_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "HCI":
        _data_p_user, labers_p_user, _features_names = HCI_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "EIGHT":
        _data_p_user, labers_p_user, _features_names = eight_dataset(WIND, modality, LABELS, INC5)
    remove_idx = remove_redundant(_data_p_user)
    remove_corr = remove_redundant(_data_p_user, 'corr')
    remove_idx = sorted(np.hstack([remove_idx, remove_corr]))
    remove_columns = [_features_names[idx] for idx in remove_idx]
    n_data_p_user = []
    for user in range(len(_data_p_user)):
        d = {str(lab): _data_p_user[user][:, idx] for idx, lab in enumerate(_features_names)}
        df = pd.DataFrame(data=d, columns=_features_names)
        df = df.drop(columns=remove_columns)
        n_data_p_user += [df.values]
    _data_p_user = np.array(n_data_p_user)
    _features_names = np.array(df.columns)

    if not len(data_p_user):
        data_p_user = _data_p_user
        features_names = _features_names
        # f_FS_idx = FS_idx[m]
        data_p_user = _data_p_user
        # f_data_p_user = []
        # for i in range(len(_data_p_user)):
        #     f_data_p_user += [_data_p_user[i][:, FS_idx[m]]]
        # f_data_p_user = np.array(f_data_p_user)
    else:
        f_data_p_user = []
        for i in range(len(_data_p_user)):
            f_data_p_user.append([])
            f_data_p_user[i] = np.hstack((data_p_user[i], _data_p_user[i]))
        data_p_user = np.array(f_data_p_user)

        # data_p_user = np.hstack((data_p_user, _data_p_user))
        features_names = np.hstack((features_names, _features_names))
        # f_FS_idx = np.hstack((f_FS_idx, FS_idx[m]))

f_feat_names = features_names[FS_IDX]
print("FT: ", f_feat_names)
a

acc_list = []
final_results = {}
list_best_clf, list_best_acc, list_best_feats = [], [], []
NUM_IT = 1
it_acc, it_f1 = [], []
_t = []
for it in range(NUM_IT):
    acc, f1_sc = [], []
    kf = KFold(n_splits=len(data_p_user), random_state=it, shuffle=False)
    for train_index, test_index in kf.split(data_p_user):
        t0 = time.time()
        samples_pred = []

        X_train = np.vstack((data_p_user[train_index]))[:, FS_IDX]
        X_test = np.vstack((data_p_user[test_index]))[:, FS_IDX]
        y_train = np.hstack((np.array(labers_p_user)[train_index]))
        y_test = np.hstack((np.array(labers_p_user)[test_index]))

        # get prediction for that modality
        c.fit(X_train, y_train)
        y_pred = c.predict(X_test)
        acc += [accuracy_score(y_test, y_pred) * 100]
        f1_sc += [f1_score(y_test, y_pred, average='binary') * 100]
        _t.append(time.time() - t0)
    it_acc += [np.mean(acc)]
    it_f1 += [np.mean(f1_sc)]


print("\n")
print(str(dataset) + ": FINAL RESULTS - " + str(TYPE) + " - " + str(LABELS) + " - not cut" + "INC 5: " + str(INC5) + " WIND: " + str(WIND))
print("Acc: " + str(np.round(np.mean(acc), 2)) + " +- " + str(np.round(np.std(acc), 2)))
print("F1-score: " + str(np.round(np.mean(f1_sc), 2)) + " +- " + str(np.round(np.std(f1_sc), 2)))
print("TIME: " + str(np.round(np.mean(_t), 2)) + " +- " + str(np.round(np.std(_t), 2)))