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
from scipy import stats
from scipy.stats import pearsonr


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
    # names = ["Random Forest", "Decision Tree", "Nearest Neighbors", "SVM", "AdaBoost", "Naive Bayes", "QDA"]
    names = ["Random Forest", "Decision Tree", "SVM", "AdaBoost", "QDA"]
    # names = ["SVM"]

    classifiers = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        # KNeighborsClassifier(),
        # LinearSVC(), #
        svm.SVC(cache_size=7000),
        AdaBoostClassifier(),
        # # GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    best_acc = 0
    best_clf = None
    for n, c in zip(names, classifiers):
        n_iter_search = 3
        print(n)
        if n == "Nearest Neighbors":
            # specify parameters and distributions to sample from
            param_dist = {"n_neighbors": sp_randint(1, 10),
                          "leaf_size":  sp_randint(1, 10),
                          'p': sp_randint(1, 10)
                         }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            acc_list = []
            est = []
            for fold in range(len(X_train)):
                grid.fit(X_train[fold], y_train[fold])
                y_test_predict = grid.predict(X_test[fold])
                acc_list.append(accuracy_score(y_test[fold].reshape(-1,1), y_test_predict)*100)
                est.append(grid.best_estimator_)
            grid = est[int(np.argmax(acc_list))]
        elif n == "Random Forest":
            # specify parameters and distributions to sample from
            if np.array(X_train[0]).shape[1] == 1 or np.array(X_train[0]).shape[1] <= 3:
                max_f = [1]
                min_samples_split = [2]
                min_samples_leaf = [1]
                max_d = [1]
            else:
                if np.array(X_train[0]).shape[1] <= 20:
                    max_f = sp_randint(1, X_train[0].shape[1])
                    max_d = sp_randint(1, X_train[0].shape[1])
                    min_samples_split = sp_randint(2, X_train[0].shape[1])
                    min_samples_leaf = sp_randint(1, X_train[0].shape[1])
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
            acc_list = []
            est = []
            for fold in range(len(X_train)):
                grid.fit(X_train[fold], y_train[fold])
                y_test_predict = grid.predict(X_test[fold])
                acc_list.append(accuracy_score(y_test[fold].reshape(-1,1), y_test_predict)*100)
                est.append(grid.best_estimator_)
            grid = est[int(np.argmax(acc_list))]
        elif n == 'SVM':
            Cs = [0.001, 0.01, 0.1, 1, 10]
            gammas = [0.001, 0.01, 0.1, 1]
            #C_range = 10. ** np.arange(-3, 8)
            #gamma_range = 10. ** np.arange(-5, 4)
            param_dist = {'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr'], 'C': Cs,
                          'gamma': gammas, 'kernel': ['linear', 'poly', 'sigmoid', 'rbf']}  # 'poly', 'rbf', 'sigmoid'
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            acc_list = []
            est = []
            for fold in range(len(X_train)):
                grid.fit(X_train[fold], y_train[fold])
                y_test_predict = grid.predict(X_test[fold])
                acc_list.append(accuracy_score(y_test[fold].reshape(-1,1), y_test_predict)*100)
                est.append(grid.best_estimator_)
            grid = est[int(np.argmax(acc_list))]
        elif n == 'Decision Tree':
            param_dist = {"criterion": ["gini", "entropy"],
                          'splitter': ['best', 'random'],
                          "min_samples_split": sp_randint(2, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "max_depth": sp_randint(1, 20)
                          }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=2, scoring='accuracy', n_iter=n_iter_search)
            acc_list = []
            est = []
            for fold in range(len(X_train)):
                grid.fit(X_train[fold], y_train[fold])
                y_test_predict = grid.predict(X_test[fold])
                acc_list.append(accuracy_score(y_test[fold].reshape(-1,1), y_test_predict)*100)
                est.append(grid.best_estimator_)
            grid = est[int(np.argmax(acc_list))]
        else:
            # Train the classifier
            grid = c
            acc_list = []
            for fold in range(len(X_train)):
                grid.fit(X_train[fold], y_train[fold])
                y_test_predict = grid.predict(X_test[fold])
                acc_list.append(accuracy_score(y_test[fold].reshape(-1,1), y_test_predict)*100)
        print("Accuracy (%): " + str(np.mean(acc_list)) + ' +- ' + str(np.std(acc_list))+ '%')
        print('-----------------------------------------')
        if np.mean(acc_list) > best_acc:
            best_acc = np.mean(acc_list)
            best_clf = grid

    print('******** Best Classifier: ' + str(best_clf) + ' ********')
    print("Accuracy: ", best_acc)

    return best_clf, best_acc


def CV_FSE(X_train, X_test, y_train, y_test, features_descrition, classifier):
    """ Performs a forward feature selection.
    Parameters
    ----------
    X_train: array-like
      train set features
    X_test: array-like
      test set features
    y_train: array-like
      train set labels
    y_test: array-like
      test set labels
    y_test: array-like
      test set labels
    features_descrition: array of strings
      array with extracted features names
    classifier: object
      classifier object
    Returns
    -------
    FS_X_train: train set best set of features
    FS_X_test: test set best set of features
    FS_lab: name of the best set of features
    """
    total_acc, FS_lab, acc_list, FS_idx = [], [], [], []
    FS_X_train = []
    FS_X_test = []
    for kf in range(len(X_train)):
        FS_X_train.append([])
        FS_X_test.append([])
    try:
        classifier.max_features = 1
    except:
        print("classifier max feat")

    print("*** Feature selection started ***")
    for feat_idx, feat_name in enumerate(features_descrition):
        acc_fold = []
        for fold in range(len(X_train)):
            classifier.fit(np.array(X_train[fold][:, feat_idx]).reshape(len(X_train[fold]),-1), y_train[fold].reshape(-1,1))
            y_test_predict = classifier.predict(np.array(X_test[fold][:, feat_idx]).reshape(len(X_test[fold]), -1))
            acc_fold.append(accuracy_score(y_test[fold], y_test_predict)*100)
        acc_list.append(np.mean(acc_fold))

    curr_acc_idx = np.argmax(acc_list)
    FS_lab.append(features_descrition[curr_acc_idx])
    last_acc = acc_list[curr_acc_idx]
    FS_idx.append(curr_acc_idx)
    for kf in range(len(X_train)):
        FS_X_train[kf] = X_train[kf][:, curr_acc_idx]
        FS_X_test[kf] = X_test[kf][:, curr_acc_idx]
    total_acc.append(last_acc)
    while 1:
        acc_list = []
        for feat_idx, feat_name in enumerate(features_descrition):
            if feat_name not in FS_lab:
                acc_fold = []
                for fold in range(len(X_train)):
                    curr_train = np.column_stack((FS_X_train[fold], X_train[fold][:, feat_idx]))
                    curr_test = np.column_stack((FS_X_test[fold], X_test[fold][:, feat_idx]))
                    classifier.fit(curr_train, y_train[fold].reshape(-1,1))
                    y_test_predict = classifier.predict(curr_test)
                    acc_fold.append(accuracy_score(y_test[fold], y_test_predict)*100)
                acc_list.append(np.mean(acc_fold))
            else:
                acc_list.append(0)
        curr_acc_idx = np.argmax(acc_list)
        if last_acc < acc_list[curr_acc_idx]:
            FS_lab.append(features_descrition[curr_acc_idx])
            last_acc = acc_list[curr_acc_idx]
            total_acc.append(last_acc)
            FS_idx.append(curr_acc_idx)
            for kf in range(len(X_train)):
                FS_X_train[kf] = np.column_stack((FS_X_train[kf], X_train[kf][:, curr_acc_idx]))
                FS_X_test[kf] = np.column_stack((FS_X_test[kf], X_test[kf][:, curr_acc_idx]))
        else:
            print('FS_idx: ', FS_idx)
            print("FINAL Features: " + str(FS_lab))
            print("Number of features", len(FS_lab))
            print("Acc: ", str(total_acc))
            print("From ", str(X_train[0].shape[1]), "to ", str(len(FS_lab)))
            break
    print("*** Feature selection finished ***")

    return np.array(FS_idx), np.array(FS_lab), np.array(FS_X_train), np.array(FS_X_test)


def remove_redundant(data, type = 'flat'):
    data = np.vstack(data)

    if type == 'flat':
        remove_idx = [idx for idx in range(len(data.T)) if len(np.unique(data[:,idx])) <= 1]
    elif type == 'corr':
        remove_idx = [idx for idx in range(1, len(data.T))
                      if pearsonr(data.T[idx-1],data.T[idx])[0] >= 0.9]

    return remove_idx


def remove_correlatedFeatures(df, threshold=0.85):
    import pandas_profiling
    """ Removes highly correlated features.
    Parameters
    ----------
    df : dataframe
        Feature vector.
    threshold : float
        Threshold for correlation.
    Returns
    -------
    df : dataframe
        Feature dataframe without high correlated features.
    """
    df = df.replace([np.inf, -np.inf, np.nan, None], 0.0)
    profile = pandas_profiling.ProfileReport(df)
    reject = profile.get_rejected_variables(threshold=threshold)
    # for rej in reject:
    #     print('Removing ' + str(rej))
    #     df = df.drop(rej, axis=1)

    return reject

dataset = "WESAD"
TYPE = "HP"
LABELS = "Valence"
# LABELS = "Arousal"
print("LABELS: ", LABELS)
INC5 = 0
WIND = 40

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

final_results = {}
for modality in modalities_list:  # iterate over different modalities
    print("Running for modality: " + modality)
    if dataset == "WESAD":
        data_p_user, labers_p_user, features_names = WESAD_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "DEAP":
        data_p_user, labers_p_user, features_names = DEAP_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "JOANA":
        data_p_user, labers_p_user, features_names = JOANA_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "HCI":
        data_p_user, labers_p_user, features_names = HCI_dataset(WIND, modality, LABELS, INC5)
    elif dataset == "EIGHT":
        data_p_user, labers_p_user, features_names = eight_dataset(WIND, modality, LABELS, INC5)

    remove_idx = remove_redundant(data_p_user)
    remove_corr = remove_redundant(data_p_user, 'corr')
    remove_idx = sorted(np.hstack([remove_idx, remove_corr]))
    remove_columns = [features_names[idx] for idx in remove_idx]
    n_data_p_user = []
    for user in range(len(data_p_user)):
        d = {str(lab): data_p_user[user][:, idx] for idx, lab in enumerate(features_names)}
        df = pd.DataFrame(data=d, columns=features_names)
        df = df.drop(columns=remove_columns)
        n_data_p_user += [df.values]
    data_p_user = np.array(n_data_p_user)
    features_names = np.array(df.columns)

    # d = {str(lab): np.vstack((data_p_user))[:, idx] for idx, lab in enumerate(features_names)}
    # df = pd.DataFrame(data=d, columns=features_names)
    # remove_idx = remove_correlatedFeatures(df)

    # remove_columns = [features_names[idx] for idx in remove_idx]
    # n_data_p_user = []
    # for user in range(len(data_p_user)):
    #     d = {str(lab): data_p_user[user][:, idx] for idx, lab in enumerate(features_names)}
    #     df = pd.DataFrame(data=d, columns=features_names)
    #     df = df.drop(columns=remove_columns)
    #     n_data_p_user += [df.values]
    # features_names = df.columns
    # data_p_user = np.array(n_data_p_user)

    # pickle.dump(data_p_user, open("input_WESAD" + sep + str(modality) + '_wo_cut2c_' + str(5) + 's_25ol_users_data_c', 'wb'))
    # pickle.dump(features_names, open("input_WESAD" + sep + str(modality) + '_wo_cut2c_' + str(5) + 's_25ol_features_labels_c',
    #                  'wb'))

    list_best_clf, list_best_acc, list_best_feats, list_best_clf_name = [], [], [], []
    X_train, X_test, y_train, y_test =  [], [], [], []
    kf = KFold(n_splits=4, random_state=1, shuffle=False)
    for train_index, test_index in kf.split(data_p_user):
        X_train += [np.vstack((data_p_user[train_index]))]
        X_test += [np.vstack((data_p_user[test_index]))]
        y_train += [np.hstack((np.array(labers_p_user)[train_index]))]
        y_test += [np.hstack((np.array(labers_p_user)[test_index]))]

    if TYPE == "G":
        classifier = bs.classification.supervised_learning.supervised_classification(X_train, y_train, CV=2)
    else:
        classifier, _ = hyperparam_tunning(X_train, X_test, y_train, y_test)

    FS_idx, FS_lab, FS_X_train, FS_X_test = CV_FSE(X_train, X_test, y_train, y_test, features_names, classifier)

    if len(FS_idx) == 1:
        _FS_X_train = [0] * len(FS_X_train)
        _FS_X_test = [0] * len(FS_X_test)
        for i in range(len(FS_X_train)):
            _FS_X_train[i] = FS_X_train[i].reshape(-1, 1)
            _FS_X_test[i] = FS_X_test[i].reshape(-1, 1)
        FS_X_train = np.array(_FS_X_train)
        FS_X_test = np.array(_FS_X_test)

    best_clf, best_acc = hyperparam_tunning(FS_X_train, FS_X_test, y_train, y_test)
    final_results[modality] = [best_clf, best_acc, FS_idx]
print("\n")
print(str(dataset) + ": FINAL RESULTS - " + str(TYPE) + " - " + str(LABELS) + " - not cut" + "INC 5: " + str(INC5) +
      "WIND: " + str(WIND))
print(final_results)
print("\n")

for i, mod in enumerate(final_results.keys()):
    # itera para cada modalidade
    print("BESTACC- MOD: " + mod + "; CLASS: " + str(final_results[mod][0]) + " FS IDX: " + str(final_results[mod][2]) + "; ACC: " + str(final_results[mod][1]))
    print("************* \n")
# pickle.dump(final_results, open("/Users/patriciabota/CODE/Emotion_Recognition/Results/select_best_feat/" + dataset + "/" + str(TYPE) + "_NOTCUT_" + LABELS + "INC 5: " + str(INC5),"wb"))

