# Import libraries
from __future__ import absolute_import, division, print_function
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
from scipy import stats
from sklearn.metrics import f1_score
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

def remove_redundant(data, type = 'flat'):
    from scipy.stats import pearsonr
    data = np.vstack(data)

    if type == 'flat':
        remove_idx = [idx for idx in range(len(data.T)) if len(np.unique(data[:,idx])) <= 1]
    elif type == 'corr':
        remove_idx = [idx for idx in range(1, len(data.T))
                      if pearsonr(data.T[idx-1],data.T[idx])[0] >= 0.9]

    return remove_idx


TUNNING = 1
dataset = "EIGHT"
# LABELS = "Arousal"
LABELS = "Valence"
print("LABELS: ", LABELS)
TYPE = "HP"
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

F1 = 0
if dataset == "JOANA":
    if LABELS == "Arousal":
        EDAH_C = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=16,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=6, min_samples_split=14,
                               min_weight_fraction_leaf=0.0, presort=False,
                               random_state=None, splitter='random')
        EDAF_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                           n_estimators=50, random_state=None)
        ECG_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                           n_estimators=50, random_state=None)
        BVP_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                      store_covariance=False, tol=0.0001)
        RESP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                           n_estimators=50, random_state=None)
        EDAH_F = [391, 400, 124, 315]
        EDAF_F = [24, 373, 301]
        ECG_F = [83, 125, 304,  17 ,147 ,159]
        BVP_F = [244 ,253]
        RESP_F = [316, 321]
        _weights = [0.19568926, 0.1838134,  0.22416508, 0.1917197,  0.20461256]
    else:
        EDAH_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDAF_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        ECG_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=14,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=19,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=18,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

        EDAH_F = [0, 233, 214]
        EDAF_F = [351, 231, 133 , 63, 279 , 34 ,233, 372]
        ECG_F = [283]
        BVP_F =  [39, 191, 248 ,207, 253]
        RESP_F = [0, 294 ,196]
        _weights =  [0.19922328, 0.20290599 ,0.19922328, 0.19942416, 0.19922328]
    class_idx = [EDAH_C, EDAF_C, ECG_C, BVP_C, RESP_C]
    FS_idx =[EDAH_F, EDAF_F, ECG_F, BVP_F, RESP_F]
elif dataset == "WESAD":
    if LABELS == "Arousal":
        EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        ECG_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C =  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=18,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDA_F = [0]
        ECG_F =  [0 ,203, 264]
        BVP_F =  [0]
        RESP_F = [7]
        _weights = [0.25002186, 0.24993442, 0.25002186, 0.25002186]
    else:
        EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=11,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        ECG_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=6,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        BVP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        RESP_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

        EDA_F =  [0]
        ECG_F = [244, 101]
        BVP_F = [1, 25]
        RESP_F = [0]
        _weights = [0.24897445, 0.24897445, 0.25307665, 0.24897445]
    class_idx = [EDA_C, ECG_C, BVP_C, RESP_C]
    FS_idx = [EDA_F, ECG_F, BVP_F, RESP_F]
elif dataset == "DEAP":
    if LABELS == "Arousal":
        EDA_C = svm.SVC(C=1, cache_size=7000, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=5,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        EDA_F = [0]
        BVP_F =[0]
        RESP_F =[0]
        _weights = [0.33354094, 0.33337108, 0.33308799]
    else:
        EDA_C = svm.SVC(C=0.01, cache_size=7000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=14,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=6,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDA_F = [0]
        BVP_F = [0]
        RESP_F = [0]
        _weights = [0.33333333, 0.33333333, 0.33333333]
    class_idx = [EDA_C, BVP_C, RESP_C]
    FS_idx = [EDA_F, BVP_F, RESP_F]
elif dataset == "HCI":
    if F1:
        if not INC5:
            if LABELS == "Arousal":
                EDA_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                       n_estimators=50, random_state=None)
                ECG_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                       n_estimators=50, random_state=None)
                RESP_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                  store_covariance=False, tol=0.0001)
                EDA_F = [244 ,242, 153  ,52, 213 ,188  ,91, 364]
                ECG_F = [70, 50]
                RESP_F = [161 ,115, 101, 251 , 33 ,160]
            else:
                EDA_C = svm.SVC(C=0.001, cache_size=7000, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
        max_iter=-1, probability=False, random_state=None, shrinking=False,
        tol=0.001, verbose=False)
                ECG_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                           max_depth=3, max_features=4, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=4, min_samples_split=4,
                           min_weight_fraction_leaf=0.0, n_estimators=6,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
                RESP_C = svm.SVC(C=0.01, cache_size=7000, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.001, kernel='sigmoid',
        max_iter=-1, probability=False, random_state=None, shrinking=False,
        tol=0.001, verbose=False)
                EDA_F = [0]
                ECG_F =  [88, 65 ,33 ,29 ,24, 23]
                RESP_F = [0]
        else:
            if LABELS == "Arousal":
                EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                           max_depth=1, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=12,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
                ECG_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                  store_covariance=False, tol=0.0001)
                RESP_C = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=12,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=13,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')
                EDA_F = [0]
                ECG_F = [26]
                RESP_F = [163]
            else:
                EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                           max_depth=1, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=7,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
                ECG_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=1, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=15,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
                RESP_C = svm.SVC(C=1, cache_size=7000, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma=0.01, kernel='sigmoid',
        max_iter=-1, probability=False, random_state=None, shrinking=False,
        tol=0.001, verbose=False)
                EDA_F = [0]
                ECG_F = [0]
                RESP_F =  [158 ,165,  41 ,168 ,197]
    else:
        if LABELS == "Arousal":
            EDA_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
            ECG_C =  RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=14,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
            RESP_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)

            EDA_F = [244 ,242, 153,  52 ,213 ,188,  91, 364]
            ECG_F = [89]
            RESP_F =[161 ,115 ,101, 251 , 33 ,160]
            _weights = [0.29782325, 0.3385495 , 0.36362724]
        else:
            EDA_C = svm.SVC(C=1, cache_size=7000, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)
            ECG_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
            RESP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)

            EDA_F = [0]
            ECG_F = [65, 24 ,29]
            RESP_F = [163 , 61, 130 , 42]
            _weights =  [0.33950617, 0.32716049, 0.33333333]
    class_idx = [EDA_C, ECG_C, RESP_C]
    FS_idx = [EDA_F, ECG_F, RESP_F]
elif dataset == "EIGHT":
    if LABELS == "Arousal":
        _weights =  [0.29782325, 0.3385495,  0.36362724]
        EDA_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        BVP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        RESP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        EDA_F = [125, 292]
        BVP_F = [71 ,64, 24, 19 ,21 ,60]
        RESP_F =  [136 , 54 , 76]
    else:
        _weights =[0.33950617, 0.32716049 ,0.33333333]
        EDA_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        BVP_C =  AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        RESP_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        EDA_F = [ 98 ,  5, 181,  17, 338, 217]
        BVP_F = [ 0, 22 ,11 ,37, 60]
        RESP_F =[4]
    class_idx = [EDA_C, BVP_C, RESP_C]
    FS_idx = [EDA_F, BVP_F, RESP_F]


acc_list = []
final_results = {}

if dataset == "WESAD":
    data_p_user, labers_p_user, features_names = WESAD_dataset(WIND, modalities_list[0], LABELS, INC5)
elif dataset == "DEAP":
    data_p_user, labers_p_user, features_names = DEAP_dataset(WIND, modalities_list[0], LABELS, INC5)
elif dataset == "JOANA":
    data_p_user, labers_p_user, features_names = JOANA_dataset(WIND, modalities_list[0], LABELS, INC5)
elif dataset == "HCI":
    data_p_user, labers_p_user, features_names = HCI_dataset(WIND, modalities_list[0], LABELS, INC5)
elif dataset == "EIGHT":
    data_p_user, labers_p_user, features_names = eight_dataset(WIND, modalities_list[0], LABELS, INC5)

NUM_IT = 1
it_acc, it_f1 = [], []
for it in range(NUM_IT):
    acc, f1_sc = [], []
    kf = KFold(n_splits=len(data_p_user), random_state=it, shuffle=False)
    for train_index, test_index in kf.split(data_p_user):
        samples_pred, _t = [], []
        t0 = time.time()
        for m, modality in enumerate(modalities_list):  # iterate over different modalities
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

            print("Running for modality: " + modality)
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

            X_train = np.vstack((data_p_user[train_index]))[:, FS_idx[m]]
            X_test = np.vstack((data_p_user[test_index]))[:, FS_idx[m]]

            y_train = np.hstack((np.array(labers_p_user)[train_index]))
            y_test = np.hstack((np.array(labers_p_user)[test_index]))

            # get prediction for that modality
            c = class_idx[m]

            c.fit(X_train, y_train)
            y_pred = c.predict(X_test)

            if not len(samples_pred):
                samples_pred = y_pred.reshape(-1, 1)
            else:
                samples_pred = np.hstack((samples_pred, y_pred.reshape(-1, 1)))
        f_y_pred = []
        # for pred in samples_pred:
        #     try:
        #         f_y_pred += [stats.mode(pred)[0][0]]
        #     except:  # 2 modes
        #         f_y_pred += [stats.mode(pred)[0][0][0]]
        for sample in range(len(samples_pred)):
            f_y_pred += [bs.biometrics.combination(results=dict(zip(modalities_list, samples_pred[sample].reshape(1,-1))),
                               weights=dict(zip(modalities_list, _weights)))[
                    'decision']]

        acc += [accuracy_score(y_test, f_y_pred) * 100]
        f1_sc += [f1_score(y_test, f_y_pred) * 100]
        _t.append(time.time() - t0)

    it_acc += [np.mean(acc)]
    it_f1 += [np.mean(f1_sc)]

print("\n")
print(str(dataset) + ": FINAL RESULTS - " + str(TYPE) + " - " + str(LABELS) + " - not cut" + "INC 5: " + str(INC5) + " WIND: " + str(WIND))
print("Acc: " + str(np.round(np.mean(acc), 2)) + " +- " + str(np.round(np.std(acc), 2)))
print("F1-score: " + str(np.round(np.mean(f1_sc), 2)) + " +- " + str(np.round(np.std(f1_sc), 2)))
print("TIME: " + str(np.round(np.mean(_t), 2)) + " +- " + str(np.round(np.std(_t), 2)))
