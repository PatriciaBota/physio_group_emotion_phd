import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import random
from scipy import interpolate
import itertools
from scipy import signal
import pandas as pd
import biosppy as bp
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import pdist,squareform


def rec_plot(s):
    """
    Compute recurrence plot (distance matrix).
    parameters:
        -s (np.array): raw signal.
    return:
        -rec (ndarray): recplot matrix.
    """
    sig_down = signal.resample(s,224)
    d = pdist(sig_down[:,None])
    rec = squareform(d)
    return rec

class customDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        X = np.array(X).astype(np.float32)
        #X = torch.tensor(X)  #torch.float32
        #y = torch.tensor(y)  #, dtype=torch.long
        
        #self.X = X
        #self.y = y
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) # float for binary CE; long for cross entropy

    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




def getDt(EDA_seg, y_a, y_s, usID):
    uIDX = np.where(y_s == usID)[0]
    X = np.vstack((EDA_seg[uIDX]))
    y = np.hstack((y_a[uIDX]))
    sjb_ID = [usID]*len(X)  
            
    return X, y, sjb_ID



def getNegPosY(X_train, y_train, labels, TH=0):
    negIDX = np.where(y_train < 5)[0]
    posIDX = np.where(y_train >= 6)[0]
    #ntIDX = np.where(y_train == TH)[0]
    #keepIDX = np.delete(np.arange(len(y_train)), ntIDX)
    keepIDX = np.sort(np.hstack((negIDX, posIDX)))


    y_train[posIDX] = labels[1]
    y_train[negIDX] = labels[0]
    
    # delete neutral
    #y_train = np.delete(y_train, ntIDX)
    #yus_train = np.delete(yus_train, ntIDX)
    X_train = X_train[keepIDX]

    return X_train, y_train



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)
    

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def padRR(R_pks, maxSz=None):
    if maxSz == None:  # else a pad was given
        maxSz = 0
        for rpk in R_pks:
            if maxSz < len(rpk[0]):
                maxSz = len(rpk[0])

    pad_rr = []
    for rpk in R_pks:
        pad_rr += [rpk[0].tolist() + [0] * (maxSz - len(rpk[0]))]

    return np.array(pad_rr)


def getTrainValTestSplit(inputX, inputy, y_s, fold_i, downsample=True):
    IDsInGrps = [["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"]
            , ["P15", "P11", "P12", "P10"]]

    INDvIDs = ["P09", "P13", "P19", "P20", "P23", "P25", "P26", "P30", "P21", "P33", "P34", "P35", "P36", "P37", "P38", "P39", "P40"]


    X_train, y_train, X_test, y_test, X_val, y_val = [], [], [], [], [], []
    GID = [0, 1, 2, 3, 4]

    for gid in GID:
        perm = list(itertools.permutations(np.arange(4), 2))
        random.seed(fold_i)
        random.shuffle(perm)
        trainIDX = [perm[0][0], perm[0][1]]
        valdIDX, testIDX = np.delete(np.arange(4), trainIDX)
        
        for i, usID in enumerate(IDsInGrps[gid]):  # iterate over group members
            uIDX = np.where(y_s == usID)[0]  
            if downsample:
                _dt = []
                
                for dt_s in inputX[uIDX]:
                    x = np.arange(0, len(dt_s))
                    f = interpolate.interp1d(x, dt_s)
                    xnew = np.linspace(0, x[-1], num=224)  # resample to fit resnet
                    _dt += [f(xnew)]
                _dt = minmax_scale(np.array(_dt), axis=1)
            else:
                _dt = minmax_scale(inputX[uIDX], axis=1)
             
            if (i in trainIDX) :
                if not len(X_train): 
                    X_train = _dt
                    y_train = inputy[uIDX]
                else:
                    X_train = np.vstack((X_train, _dt))
                    y_train = np.hstack((y_train, inputy[uIDX]))
            elif i in [valdIDX]:
                if not len(X_val): 
                    X_val = _dt
                    y_val = inputy[uIDX]
                else:
                    X_val = np.vstack((X_val, _dt))
                    y_val = np.hstack((y_val, inputy[uIDX]))
            else:
                if not len(X_test): 
                    X_test = _dt
                    y_test = inputy[uIDX]
                else:
                    X_test = np.vstack((X_test, _dt))
                    y_test = np.hstack((y_test, inputy[uIDX]))
    
    for i, usID in enumerate(INDvIDs):  # iterate over individual members
        uIDX = np.where(y_s == usID)[0]  
        if downsample:
            _dt = []
            for dt_s in inputX[uIDX]:
                x = np.arange(0, len(dt_s))
                f = interpolate.interp1d(x, dt_s)
                xnew = np.linspace(0, x[-1], num=224)  # resample to fit resnet
                _dt += [f(xnew)]
            _dt = minmax_scale(np.array(_dt), axis=1)
        else:
            _dt = minmax_scale(inputX[uIDX], axis=1)

        X_train = np.vstack((X_train, _dt))
        y_train = np.hstack((y_train, inputy[uIDX]))
    return X_train, y_train, X_val, y_val, X_test, y_test


## Feature extraction
def remove_correlatedFeatures(stfv, feat_header, threshold=0.95):
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
    d = {str(lab): stfv[:, idx] for idx, lab in enumerate(feat_header)}
    df = pd.DataFrame(data=d, columns=feat_header)

    df = df.replace([np.inf, -np.inf, np.nan, None], 0.0)
    constant_filter = VarianceThreshold(threshold=0)   # threshold = 0 for constant


# fit the data
    constant_filter.fit(df.values)
# We can check the variance of different features as

    redfeat_header = [column for column in feat_header
                        if column in df.columns[constant_filter.get_support()]]
      
    redstfv = constant_filter.transform(df.values).astype(np.float32)

    _d = {str(lab): redstfv[:, idx] for idx, lab in enumerate(redfeat_header)}
    df = pd.DataFrame(data=_d, columns=redfeat_header)

# Create correlation matrix
    corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop features 
    df.drop(to_drop, axis=1, inplace=True)

#df = utils.remove_correlatedFeatures(df)
    redstfv = df.values
    redfeat_header = np.array(df.keys())

    return redstfv, redfeat_header


def getFts(fv, feat_header, label, sample, SR):
    label += "_"
    
    # temporal features
    ftsDict = bp.features.temporal_features.signal_temp(sample, SR)
    feat_header += [label + fn for fn in ftsDict.keys()]
    fv += list(ftsDict[:])  
    
    # statistic features
    ftsDict = bp.features.statistic_features.signal_stats(sample)
    feat_header += [label + fn for fn in ftsDict.keys()]
    fv += list(ftsDict[:])  

    # spectral features
    ftsDict = bp.features.spectral_features.signal_spectral(sample, SR)
    feat_header += [label + fn for fn in ftsDict.keys()]
    fv += list(ftsDict[:]) 

    return feat_header, fv


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import minmax_scale
from scipy.stats.mstats import gmean
from sklearn.model_selection import train_test_split
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score



def hyperparam_tunning(n, X_train, y_train, priors=None, CV=4):
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

    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "ExtraTree", "AdaBoost", "GradientBoosting", "Gaussian NB",
             "Multinomial NB", "Complement NB", "Bernoulli NB", "Linear Discriminant Analysis","Quadratic Discriminant Analysis", "SVM", "Linear SVM", "Gaussian Process", "LogisticRegression"]
   
    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(class_weight="balanced"),
        ExtraTreesClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(priors=priors),
        MultinomialNB(),
        ComplementNB(),
        BernoulliNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        svm.SVC(cache_size=3000, max_iter=3000, class_weight="balanced"),
        svm.LinearSVC(),
        GaussianProcessClassifier(),
        LogisticRegression()
    ]

    print("USING GRID SEARCH")
    c = classifiers[names.index(n)]  # get classifier

    n_iter_search = CV
    print(n)
    print("cl", c)
    if n == "Random Forest":
        # specify parameters and distributions to sample from
        max_f = int(X_train.shape[1]//2) +1
        print("max_f", max_f)
        param_dist = {"max_depth": sp_randint(1, max_f),
                  "max_features": sp_randint(1, max_f),
                  "min_samples_split": sp_randint(1, 50),
                  "min_samples_leaf": sp_randint(1, 50),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": sp_randint(2, 50)}

        # run randomized search
        grid = RandomizedSearchCV(c, param_dist, cv=CV, scoring='f1_macro', n_iter=n_iter_search)
        grid.fit(X_train, y_train)
        y_test_predict = grid.predict(X_train)
        acc = f1_score(y_train, y_test_predict, average="macro")*100
        best_est = grid.best_estimator_
    elif n == 'SVM':
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1, "scale"]
        kernel = ["linear", "poly", "rbf", "sigmoid"]
        param_dist = {'shrinking': [True, False], 'decision_function_shape': ['ovo', 'ovr'], 'C': Cs,
                'gamma': gammas, 'kernel': kernel}
        # run randomized search
        grid = RandomizedSearchCV(c, param_dist, cv=CV, scoring='f1_macro')
        grid.fit(X_train, y_train)
        y_test_predict = grid.predict(X_train)
        acc = f1_score(y_train, y_test_predict, average="macro")*100
        best_est = grid.best_estimator_
    else:
        # Train the classifier
        #X_data = np.vstack((X_train, X_test))
        #y_data = np.hstack((y_train, y_test))
        #c.fit(X_train, y_train)
        acc = None #f1_score(X_train, y_train, average="macro")*100
        #acc = np.mean(cross_val_score(c, X_data, y_data, cv=CV, scoring="f1_macro")*100)
        best_est = c
    print("Accuracy (%): " + str(acc) + '%')
    print('-----------------------------------------')

    print('******** Best Classifier: Train' + str(best_est) + ' ********')
    print("Accuracy: ", acc)

    return best_est, acc
