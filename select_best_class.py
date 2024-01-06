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


def hyperparam_tunning(X_train, X_test, y_train, y_test, _FSE=False):
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
    names = ["Nearest Neighbors", "Random Forest", "Decision Tree", "SVM", "AdaBoost", "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        svm.SVC(),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    best_acc = 0
    best_clf = None
    for n, c in zip(names, classifiers):
        n_iter_search = 10
        print(n)
        if n == "Nearest Neighbors":
            # specify parameters and distributions to sample from
            param_dist = {"n_neighbors": sp_randint(1, 5),
                          "leaf_size":  sp_randint(1, 50),
                          'p': sp_randint(1, 10)
                         }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=4, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        elif n == "Random Forest":
            # specify parameters and distributions to sample from
            if _FSE:
                max_f = [1]
            else:
                if np.array(X_train).shape[1] <= 20:
                    max_f = sp_randint(1, X_train.shape[1])
                else:
                    max_f = sp_randint(1, 50)
            param_dist = {"max_depth": sp_randint(1, 50),
                          "max_features": max_f,
                          "min_samples_split": sp_randint(2, 50),
                          "min_samples_leaf": sp_randint(1, 50),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"],
                          "n_estimators": sp_randint(5, 50)}
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=4, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        elif n == 'SVM':
            Cs = [0.001, 0.01, 0.1, 1, 10]
            gammas = [0.001, 0.01, 0.1, 1]
            C_range = 10. ** np.arange(-3, 8)
            gamma_range = 10. ** np.arange(-5, 4)
            param_dist = {'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr'], 'C': C_range,
                          'gamma': gamma_range}
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=4, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        elif n == 'Decision Tree':
            param_dist = {"criterion": ["gini", "entropy"],
                          'splitter': ['best', 'random'],
                          "min_samples_split": sp_randint(2, 20),
                          "min_samples_leaf": sp_randint(1, 20),
                          "max_depth": sp_randint(1, 20)
                          }
            # run randomized search
            grid = RandomizedSearchCV(c, param_dist, cv=4, scoring='accuracy', n_iter=n_iter_search)
            grid.fit(X_train, y_train)
            y_test_predict = grid.predict(X_test)
            acc = accuracy_score(y_test, y_test_predict)*100
            best_est = grid.best_estimator_
        else:
            # Train the classifier
            X_data = np.vstack((X_train, X_test))
            y_data = np.hstack((y_train, y_test))
            acc = np.mean(cross_val_score(c, X_data, y_data, cv=4)*100)
            best_est = c
        print("Accuracy (%): " + str(acc) + '%')
        print('-----------------------------------------')
        if acc > best_acc:
            best_acc = acc
            best_clf = best_est

    print('******** Best Classifier: ' + str(best_clf) + ' ********')
    print("Accuracy: ", best_acc)

    return best_clf, best_acc


def JOANA_dataset(SENSOR):
    windows_time = 5
    print("Windows time: ", windows_time)

    users_df = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_cut2c_' + str(windows_time) + 's_25ol_users_data', 'rb'))
    labels_a = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'rb'))
    features_names = pickle.load(open("input_Joana" + sep + str(SENSOR) + '_cut2c_' + str(windows_time) + 's_25ol_features_labels',
                     'rb'))

    for i in range(len(users_df)):
        users_df[i] = minmax_scale(users_df[i])

    return users_df, labels_a, features_names


modalities_list = ["EDAh", "ECG", "BVP", "Resp"]
final_results = {}
for modality in modalities_list:  # iterate over different modalities
    print("Running for modality: " + modality)
    data_p_user, labers_p_user, features_names = JOANA_dataset(modality)
    kf = KFold(n_splits=4, random_state=1, shuffle=False)
    list_best_clf, list_best_acc = [], []
    for train_index, test_index in kf.split(data_p_user):
        X_train = np.vstack((data_p_user[train_index]))
        X_test = np.vstack((data_p_user[test_index]))
        y_train = np.hstack((np.array(labers_p_user)[train_index]))
        y_test = np.hstack((np.array(labers_p_user)[test_index]))

        best_clf, best_acc = hyperparam_tunning(X_train, X_test, y_train, y_test, _FSE=False)
        list_best_clf += [best_clf]
        list_best_acc += [best_acc]
    final_results[modality] = [list_best_clf, list_best_acc]
print("\n")
print("FINAL RESULTS VALENCE - EDA HAND")
print(final_results)
