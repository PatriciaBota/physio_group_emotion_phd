import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import matplotlib.pyplot as plt
import models
import utils
import biosppy as bp
from scipy import stats, signal
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report
import itertools
import random
from fastdtw import fastdtw
from scipy import interpolate
import sys
import math
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import AMIGOS_model
import adapt_AMIGOS_Resnet1d
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import gc 
#os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

WINDSC = 20
DATASET = "RECOLA"
TYPE = "FV"

if DATASET == "AMIGOS":
    VIDSELECT = "LVSV"
    NAME = "normPVid_woutSpks"
    VID = "LVSVO"
    
    _dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"
else:
    _dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"


with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

SR = data['SR']  # sampling rate

#inputX = data['RR_redstfv']
inputX = np.hstack((data['EDA_redstfv'], data['RR_redstfv']))

del data['EDA_RP']
del data['EDA_spect']
del data['EDA_cvx']
del data['EDR']
del data['EDL']
#del data['ECG']
del data['EDA_stfv']
del data['EDA_redstfv']
del data['EDA_fv_h']
del data['RR_redstfv']
del data['RR_stfv']
del data['RR_fv_h']

gc.collect()

if DATASET == "KEmoCon":
    data['y_a'] = np.array(data['y_a'])
    data['y_a'] = np.append(data['y_a'], [1, 5])
    data['y_a'] = minmax_scale(data['y_a'], [-1, 1])
    data['y_a'] = data['y_a'][:-2]


if DATASET == "KEmoCon":
    data['y_v'] = np.array(data['y_v'])
    data['y_v'] = np.append(data['y_v'], [1, 5])
    data['y_v'] = minmax_scale(data['y_v'], [-1, 1])
    data['y_v'] = data['y_v'][:-2]

inputY = data['y_v'].copy()

dY = [1 if d > 0 else 0 for d in inputY]

if DATASET == "AMIGOS":
    IDsInGrps = np.array([["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"], ["P15", "P11", "P12", "P10"]])
elif DATASET == "KEmoCon":
    IDsInGrps = np.array([["P1", "P2"], ["P3", "P4"], ["P5", "P6"], ["P7", "P8"], ["P9", "P10"], ["P11", "P12"], ["P13", "P14"], ["P15", "P16"], ["P17", "P18"], ["P19", "P20"], ["P21", "P22"], ["P22", "P23"], ["P23", "P24"], ["P25", "P26"], ["P27", "P28"], ["P29", "P30"], ["P31", "P32"]])
else:
    IDsInGrps = np.array([["P25", "P26"], ["P41", "P42"], ["P45", "P46"]])

print(np.round(np.unique(dY, return_counts=True)[1]/len(dY)*100, 2))

print("input sh", inputX.shape)

labels = [-1, 1]

assert len(inputX) == len(inputY)
assert len(inputX) == len(data['y_s'])

print(labels, np.round(np.unique(inputY, return_counts=True)[1]/len(inputY)*100, 2))

data["y_g"] = np.array(data["y_g"])

# divide by user
X, y, Y_VID, Y_S, TS, Y_G = [], [], [], [], [], []
_, idx = np.unique(data['y_s'], return_index=True)
for _, sID in enumerate(data['y_s'][np.sort(idx)]):
    sIDX = np.where(sID == data['y_s'])[0]  # samples idx of user 
    
    X += [inputX[sIDX]]
    #kmeans = KMeans(n_clusters=2, random_state=0).fit(inputY[sIDX].reshape(-1, 1))
    #TH = np.mean(kmeans.cluster_centers_)
    
    y += [[1 if d >= 0 else 0 for d in inputY[sIDX]]]
    #y += [inputY[sIDX]]
    Y_S += [data['y_s'][sIDX]]
    Y_VID += [data['y_vid'][sIDX]]
    TS += [data['timestamp'][sIDX]]
    Y_G += [data['y_g'][sIDX]]
    #y += [[d if d >= TH else d for d in inputY[sIDX]]]

    #_sidx = samplsIDX[sIDX]  # user samples
X = np.array(X)
Y = np.array(y)
Y_S = np.array(Y_S)
Y_VID = np.array(Y_VID)
TS = np.array(TS)
Y_G = np.array(Y_G)


def rmoveLV(y_vid_test):
    VIDEOS_LABELS = ["N1", "P1", "B1", "U1"]
    vidIDX = []
    for vidID in VIDEOS_LABELS:
        vidIDX += np.where(vidID == y_vid_test)[0].tolist()
    return vidIDX

cpIDsGP = IDsInGrps.copy()
_, idx = np.unique(np.hstack((Y_S)), return_index=True)
SUBJ_ORD = np.hstack((Y_S))[np.sort(idx)]
assert len(SUBJ_ORD) == len(Y)

RUN_TIME = time.time()
cv_y_pred, cv_y_test, cv_test_acc, cv_test_f1, cv_pred_tm, cv_noCorr, cv_eqlb, cv_wstd, cv_train_tm, cv_test_f1_macro =\
        [], [], [], [], [], [], [], [], [], []
gseed, fold_i = 0, -1
kf = KFold(n_splits=len(Y))  # LOSO
for trainUS, testUS in kf.split(np.arange(len(Y))):  # divide by user  
    fold_i += 1
    print("Fold: ", fold_i)
    train_us_byf = np.array([np.unique(Y_S[f])[0] for f in trainUS])
    if DATASET == "AMIGOS":
        assert len(train_us_byf) == 36

    if DATASET == "KEmoCon":
        assert len(np.unique(Y_G[testUS][0])) == 1
        if np.unique(Y_G[testUS][0])[0] == -1:
            continue
    elif DATASET == "AMIGOS":
        assert len(np.unique(Y_G[testUS])) == 1
        if np.unique(Y_G[testUS])[0] == -1:
            continue
    else:
        if np.unique(Y_S[testUS][0])[0] not in IDsInGrps:
            continue
    if DATASET == "KEmoCon" or DATASET == "RECOLA":
        groupID = np.where(np.unique(Y_S[testUS][0]) == IDsInGrps)[0]
        todlt = np.where(IDsInGrps[groupID][0] == np.unique(Y_S[testUS][0])[0])[0]  # delete test us from gp
    else:
        groupID = np.where(np.unique(Y_S[testUS]) == IDsInGrps)[0]
        todlt = np.where(IDsInGrps[groupID][0] == np.unique(Y_S[testUS])[0])[0]  # delete test us from gp
    train_s = np.delete(IDsInGrps[groupID][0], todlt)  # list with group users for training + val

    if DATASET == "KEmoCon":
        assert SUBJ_ORD[testUS] == np.unique(Y_S[testUS][0])[0]
    else:
        assert SUBJ_ORD[testUS] == np.unique(Y_S[testUS])[0]
    
    if DATASET == "AMIGOS":
        if not GROUPBIAS:
            us_to_rtrainIDX = []
            for rmU in train_s:
                us_to_rtrainIDX += [np.where(rmU == train_us_byf)[0][0]]  # remove remaining group users from training  
            trainUS = np.delete(trainUS, us_to_rtrainIDX)  # must have all users \ val + test

            np.random.seed(fold_i)
            val_IDX = np.random.choice(SUBJ_ORD[trainUS])

            valIDX = np.where(val_IDX == SUBJ_ORD)[0]  # fold idx of validation user
        else:
            np.random.seed(fold_i)
            val_IDX = np.random.choice(train_s)

            valIDX = np.where(val_IDX == SUBJ_ORD)[0]  # fold idx of validation user
    elif DATASET == "KEmoCon" or DATASET == "RECOLA":
        np.random.seed(fold_i)
        val_IDX = np.random.choice(SUBJ_ORD[trainUS])
        valIDX = np.where(val_IDX == SUBJ_ORD)[0]  # fold idx of validation user

        if train_s[0] not in np.unique(np.hstack((Y_S))):
            print("h", SUBJ_ORD[testUS])
            continue
        #if DATASET == "KEmoCon":
        #    valIDX = valIDX[0]
     
    train_us_byf = np.array([np.unique(Y_S[f])[0] for f in trainUS])
    v_trainIDX = np.where(val_IDX == train_us_byf)[0]
    trainUS = np.delete(trainUS, v_trainIDX)  # must have all users \ val + test
        
    if DATASET == "AMIGOS":
        if not GROUPBIAS:
            assert len(trainUS) == 32
            assert len(trainUS) + len(valIDX) + len(testUS) + 3 == 37
        else:
            assert len(trainUS) == 35
            assert len(trainUS) + len(valIDX) + len(testUS) == 37
    elif DATASET == "KEmoCon":
        assert len(trainUS) + len(valIDX) + len(testUS) == 26
    else:
        assert len(trainUS) + len(valIDX) + len(testUS) == 18

    assert SUBJ_ORD[testUS] not in SUBJ_ORD[trainUS]
    assert SUBJ_ORD[valIDX] not in SUBJ_ORD[trainUS]
    assert SUBJ_ORD[valIDX] not in SUBJ_ORD[testUS]

    print("########################")
    print("group: ", IDsInGrps[groupID])  
    print("test user", testUS, SUBJ_ORD[testUS])
    print("val user", valIDX, SUBJ_ORD[valIDX])
    print("train user", SUBJ_ORD[trainUS]) 
    
    if TYPE == "FV" or TYPE == "MORP":
        X_train = np.vstack((X[trainUS]))  # concatenate users
        X_test = np.vstack((X[testUS]))  
        X_val = np.vstack((X[valIDX]))
    elif TYPE == "IMG":
        X_train = np.hstack((X[trainUS]))  # concatenate users
        X_test = np.hstack((X[testUS]))  
        X_val = np.hstack((X[valIDX]))

    y_train = np.hstack((Y[trainUS]))
    y_test = np.hstack((Y[testUS]))
    y_val = np.hstack((Y[valIDX]))

    tm_train = np.hstack((TS[trainUS]))
    tm_test = np.hstack((TS[testUS]))
    tm_val = np.hstack((TS[valIDX]))

    y_s_train = np.hstack((Y_S[trainUS]))
    y_s_test = np.hstack((Y_S[testUS]))
    y_s_val = np.hstack((Y_S[valIDX]))

    y_vid_train = np.hstack((Y_VID[trainUS]))
    y_vid_test = np.hstack((Y_VID[testUS]))
    y_vid_val = np.hstack((Y_VID[valIDX]))
    
    y_g_test = np.hstack((Y_G[testUS]))
    y_g_val = np.hstack((Y_G[valIDX]))
    y_g_train = np.hstack((Y_G[trainUS]))
    
     
    # remove short videos from test - only maintain data from LV
    if DATASET == "AMIGOS": 
        if VIDSELECT == "LVSV":
            vidIDX = rmoveLV(y_vid_test)
            y_vid_test = y_vid_test[vidIDX]
            y_test = y_test[vidIDX]
            X_test = X_test[vidIDX]
            y_s_test = y_s_test[vidIDX]
            tm_test = tm_test[vidIDX]

            vidIDX = rmoveLV(y_vid_val)
            y_vid_val = y_vid_val[vidIDX]
            y_val = y_val[vidIDX]
            X_val = X_val[vidIDX]
            y_s_val = y_s_val[vidIDX]
            tm_val = tm_val[vidIDX]

            vidIDX = rmoveLV(y_vid_train)
            y_vid_train = y_vid_train[vidIDX]
            y_train = y_train[vidIDX]
            X_train = X_train[vidIDX]
            y_s_train = y_s_train[vidIDX]
    print("unique test", np.unique(y_vid_test))
    print("val size", X_val.shape)
    print("test size", X_test.shape)
    print("train size", X_train.shape)

    assert len(np.unique(tm_test)) == len(tm_test)
    assert len(np.unique(tm_val)) == len(tm_val)
    if DATASET == "AMIGOS":
         assert len(tm_val) == len(tm_test)

    if DATASET == "AMIGOS":
        assert len(np.unique(y_vid_test)) == 4  # 4 long-videos
        assert len(np.unique(y_vid_val)) == 4  # 4 long-videos
    assert len(np.unique(y_s_val)) == 1  # 1 val subj 
    assert len(np.unique(y_s_test)) == 1  # 1 test subj 

    print("SIZE", X_train.shape)
    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

    # SVM "Gaussian NB" "Random Forest"
    #prior = np.unique(y_train, return_counts=True)[1]/len(y_train)
    clf, _ = utils.hyperparam_tunning("Gaussian NB", X_train, y_train, CV=4)
 
    t0 = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cv_train_tm += [time.time()-t0]
    
    print("#### Test set V")
    acc = accuracy_score(y_test, y_pred)
    print("Test set", testUS)   
    print("_y_pred", y_pred)
    print("_y_test", y_test)
    print(classification_report(y_test, y_pred, labels=np.unique(y_test)))
    
    cv_test_acc += [acc*100]
    cv_test_f1 += [f1_score(y_test, y_pred, average='weighted')*100]
    cv_test_f1_macro += [f1_score(y_test, y_pred, average='macro')*100]
    cv_y_test += y_test.tolist()

    cv_y_pred += y_pred.tolist()

print("DATASET", DATASET) 
print("clf: ", clf)
print("Avg acc: ", np.round(np.mean(cv_test_acc), 2), " +- ", np.round(np.std(cv_test_acc), 2))
print("Avg weighted f1 score: ", np.round(np.mean(cv_test_f1), 2), " +- ", np.round(np.std(cv_test_f1), 2))
print("Avg macro f1 score: ", np.round(np.mean(cv_test_f1_macro), 2), " +- ", np.round(np.std(cv_test_f1_macro), 2))
print("Avg Pred time: ", np.round(np.mean(cv_train_tm), 5), " +- ", np.round(np.std(cv_train_tm), 5))

print("\n")
print("Eq labels: ", np.round(np.mean(cv_eqlb), 2), " +- ", np.round(np.std(cv_eqlb), 2))

print(labels, np.round(np.unique(inputY, return_counts=True)[1]/len(inputY)*100, 2))

print("ALL", np.round(np.mean(cv_test_acc), 2), " +- ", np.round(np.std(cv_test_acc), 2), " & ", np.round(np.mean(cv_test_f1), 2), " +- ", np.round(np.std(cv_test_f1), 2), " & ", np.round(np.mean(cv_test_f1_macro), 2), " +- ", np.round(np.std(cv_test_f1_macro), 2), " & ", np.round(np.mean(cv_train_tm), 5), " +- ", np.round(np.std(cv_train_tm), 5))
