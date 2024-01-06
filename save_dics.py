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


dict={'EDAh': [[svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.1, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)], [60.539752005835155, 86.25541125541125, 54.127257093723124, 75.1089588377724], [np.array([0, 1]), np.array([  0, 174, 175,  48]), np.array([204, 121, 381, 449, 447]), np.array([ 73, 393, 125, 394, 447])]], 'ECG': [[svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.001, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.001, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)], [60.539752005835155, 86.32756132756133, 54.127257093723124, 75.1089588377724], [np.array([0, 1]), np.array([ 97, 183]), np.array([0, 1]), np.array([0, 1])]], 'BVP': [[svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)], [60.539752005835155, 86.32756132756133, 46.173688736027515, 75.1089588377724], [np.array([0, 1]), np.array([0, 1]), np.array([  2, 135,  11, 168]), np.array([0, 1])]], 'Resp': [[svm.SVC(C=0.001, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False), svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False), svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.001, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=False,
    tol=0.001, verbose=False)], [60.539752005835155, 86.32756132756133, 54.127257093723124, 75.1089588377724], [np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]]}
pickle.dump(dict,open("/Users/patriciabota/CODE/Emotion_Recognition/Results/select_best_feat/HP_CUT_EDAH_V_SVM","wb"))
#f_d=pickle.load(open("/Users/patriciabota/CODE/Emotion_Recognition/Results/select_best_feat/G_CUT_EDAH_A","rb"))

for i, mod in enumerate(dict.keys()):
    # itera para cada modalidade

    bc_i = np.argmax(dict[mod][1])
    print("MOD: " + mod + "; CLASS: " + str(dict[mod][0][bc_i]) + " FS IDX: " + str(dict[mod][2][bc_i]) + "; ACC: " + str(dict[mod][1][bc_i]))
    print("************* \n")
