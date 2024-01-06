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
        labels_a = pickle.load(open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'rb'))
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

    users_df = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    if LABELS == "Valence":
        labels_a = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'rb'))
    elif LABELS == "Arousal":
        labels_a = pickle.load(
            open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal',
                 'rb'))
    features_names = pickle.load(open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels',
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
dataset = "HCI"
# LABELS = "Arousal"
LABELS = "Valence"
print("LABELS: ", LABELS)
TYPE = "HP"
INC5 = 0
WIND = 5

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
        weights = [0.20483331, 0.18145601 ,0.19539971, 0.20073665, 0.21757432]
        EDAH_C = GaussianNB(priors=None, var_smoothing=1e-09)
        EDAH_F = [319 ,163, 291, 205]
        EDAF_C = GaussianNB(priors=None, var_smoothing=1e-09)
        EDAF_F = [129 ,311,  64, 265, 181, 284]
        ECG_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        ECG_F = [ 37, 102 , 34, 147,  68]
        BVP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        BVP_F = [148, 147]
        RESP_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        RESP_F = [171,  45 ,102  ,98  ,78 ,109 , 31 ,  1, 105 , 30, 134]
    else:
        weights = [0.1999954,  0.20001839 ,0.1999954 , 0.1999954,  0.1999954 ]
        EDAH_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=14,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDAF_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        ECG_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=8,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=14,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=7,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDAH_F =  [16]
        EDAF_F = [166]
        ECG_F =  [0]
        BVP_F =  [0]
        RESP_F =  [0]

    class_idx = [EDAH_C, EDAF_C, ECG_C, BVP_C, RESP_C]
    FS_idx =[EDAH_F, EDAF_F, ECG_F, BVP_F, RESP_F]
elif dataset == "WESAD":
    if LABELS == "Arousal":
        weights =  [0.25015317, 0.24954048, 0.25015317, 0.25015317]
        EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                           max_depth=1, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=18,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
        ECG_C = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=9, min_samples_split=6,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='random')
        BVP_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                           max_depth=1, max_features=1, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=15,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
        RESP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=1, max_features=4, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=3, min_samples_split=3,
                           min_weight_fraction_leaf=0.0, n_estimators=14,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
        EDA_F = [1]
        ECG_F = [157, 154]
        BVP_F = [89]
        RESP_F = [ 30 , 24,  59, 154, 168]
    else:
        weights =  [0.24973089, 0.25080732, 0.24973089, 0.24973089]
        EDA_C = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=15,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        ECG_C = KNeighborsClassifier(algorithm='auto', leaf_size=7, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=8, p=5,
                     weights='uniform')
        BVP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=13,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        RESP_C = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=1, max_features=1, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=12,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        EDA_F = [2]
        ECG_F = [157]
        BVP_F = [0]
        RESP_F = [0]
    class_idx = [EDA_C, ECG_C, BVP_C, RESP_C]
    FS_idx =[EDA_F, ECG_F, BVP_F, RESP_F]
elif dataset == "HCI":
    if LABELS == "Arousal":
        weights =   [0.33848442, 0.33499289 ,0.32652269]
        EDA_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        ECG_C = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                              store_covariance=False, tol=0.0001)
        RESP_C = GaussianNB(priors=None, var_smoothing=1e-09)
        EDA_F = [111,  27 , 65, 347, 302,  70, 341,  19, 107]
        ECG_F =  [63 ,54, 91, 22, 76,  4, 44]
        RESP_F = [ 94, 162,  57, 175,  93 , 82]
    elif LABELS == "Valence":
        weights =  [0.33359771, 0.33512293, 0.33127936]
        EDA_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        ECG_C = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=18,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=8, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
        RESP_C = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
        EDA_F = [216, 88]
        ECG_F = [116]
        RESP_F = [79, 101]
    class_idx = [EDA_C, ECG_C, RESP_C]
    FS_idx =[EDA_F, ECG_F, RESP_F]

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

NUM_IT = 20
it_acc, it_f1 = [], []
for it in range(NUM_IT):
    acc, f1_sc = [], []
    kf = KFold(n_splits=4, random_state=it, shuffle=False)
    for train_index, test_index in kf.split(data_p_user):
        samples_pred = []
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
            f_y_pred += [bs.biometrics.combination(results=dict(zip(modality, samples_pred)),
                               weights=dict(zip(modality, weights)))[
                    'decision']]

        acc += [accuracy_score(y_test, f_y_pred) * 100]
        f1_sc += [f1_score(y_test, f_y_pred, average='binary') * 100]

    it_acc += [np.mean(acc)]
    it_f1 += [np.mean(f1_sc)]


print("\n")
print(str(dataset) + ": FINAL RESULTS - " + str(TYPE) + " - " + str(LABELS) + " - not cut" + "INC 5: " + str(INC5))
print("Acc: " + str(np.round(np.mean(it_acc), 2)) + " +- " + str(np.round(np.std(it_acc), 2)))
print("F1-score: " + str(np.round(np.mean(it_f1), 2)) + " +- " + str(np.round(np.std(it_f1), 2)))

