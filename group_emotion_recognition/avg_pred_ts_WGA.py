import torch
from torch import nn
from torch.functional import _return_counts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor
import pickle
import numpy as np
import matplotlib.pyplot as plt
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
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from functools import partial
import resNet_lin
import torchvision.models as models
import gc
import psutil
import json
import pdb
import recurrence_plot as rp
import scipy
from PIL import Image
from matplotlib import cm


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

WINDSC = 20
DATASET = "AMIGOS"
DISDT = False
DIM = "Valence"  # "Valence"
ANN_TH = 0

if DATASET == "AMIGOS":
    VIDSELECT = "LVSV"
    NAME = "normPVid_woutSpks"
    VID = "LVSVO"
    
    _dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"
else:
    _dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"


with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
    data = pickle.load(handle)
#print(data.keys())

#print("RR",data["RR_red_fv_h"])
#print("EDA", data["EDA_red_fv_h"])

def resample(_signal):
    n_s = []
    for sidx in range(len(_signal)):
        x = np.arange(0, len(_signal[sidx]))
        f = interpolate.interp1d(x, _signal[sidx])
        xnew = np.linspace(0, x[-1], num=2500)  # resample to fit resnet
        n_s += [f(xnew)]
    return np.array(n_s)


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

v = [1 if d >= 0 else 0 for d in data["y_v"]]
print(np.unique(v, return_counts=True)[1]/len(v))
v = [1 if d > 0 else 0 for d in data["y_a"]]
print(np.unique(v, return_counts=True)[1]/len(v))
print(len(v))

SR = data['SR']  # sampling rate
TYPE = "FV"  # FV, IMG, MORP
inputX = np.array(data['RR_redstfv'])

if DIM == "Arousal":
    inputY = data["y_a"]
elif DIM == "Valence":
    inputY = data["y_v"]

#inputX = data['EDA_cvx'] #"EDA_redstfv" RR_redstfv
#inputX = np.hstack((data['EDA_redstfv'], data['RR_redstfv']))
#inputX = data['EDA_RP'] 



del data['EDA_RP'] 
del data['EDA_spect']
del data['EDA_cvx'] 
del data['EDR'] 
del data['EDL']
#del data['ECG']
#del data['EDA_stfv']
#del data['EDA_fv_h']
#del data['EDA_redstfv']
#del data['RR_stfv']
#del data['RR_fv_h']
#del data['RR_redstfv']

for k in data:
    print(k)
    data[k] = np.array(data[k])
gc.collect()

if DATASET == "AMIGOS":
    IDsInGrps = np.array([["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"], ["P15", "P11", "P12", "P10"]])
elif DATASET == "KEmoCon":
    IDsInGrps = np.array([["P1", "P2"], ["P3", "P4"], ["P5", "P6"], ["P7", "P8"], ["P9", "P10"], ["P11", "P12"], ["P13", "P14"], ["P15", "P16"], ["P17", "P18"], ["P19", "P20"], ["P21", "P22"], ["P22", "P23"], ["P23", "P24"], ["P25", "P26"], ["P27", "P28"], ["P29", "P30"], ["P31", "P32"]])
else:
    IDsInGrps = np.array([["P25", "P26"], ["P41", "P42"], ["P45", "P46"]])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


print("input sh", inputX.shape)
labels = [-1, 1]

assert len(inputX) == len(inputY)
assert len(inputX) == len(data['timestamp'])
assert len(inputX) == len(data['y_s'])

data["y_g"] = np.array(data["y_g"])
# divide by user
X, y, Y_VID, Y_S, TS, Y_G = [], [], [], [], [], []
_, idx = np.unique(data['y_s'], return_index=True)
for _, sID in enumerate(data['y_s'][np.sort(idx)]):
    sIDX = np.where(sID == data['y_s'])[0]  # samples idx of user 
    
    X += [inputX[sIDX]]
    #kmeans = KMeans(n_clusters=2, random_state=0).fit(inputY[sIDX].reshape(-1, 1))
    #TH = np.mean(kmeans.cluster_centers_)
    
    #y += [[1 if d >= 0 else 0 for d in inputY[sIDX]]]
    y += [inputY[sIDX]]
    Y_S += [data['y_s'][sIDX]]
    if DATASET == "AMIGOS":
        Y_VID += [data['y_vid'][sIDX]]
    TS += [data['timestamp'][sIDX]]
    Y_G += [data['y_g'][sIDX]]
    #y += [[d if d >= TH else d for d in inputY[sIDX]]]

    #_sidx = samplsIDX[sIDX]  # user samples8
X = np.array(X)
Y = np.array(y)
Y_S = np.array(Y_S)
if DATASET == "AMIGOS":
    Y_VID = np.array(Y_VID)
TS = np.array(TS)
Y_G = np.array(Y_G)

def rmoveLV(y_vid_test):
    VIDEOS_LABELS = ["N1", "P1", "B1", "U1"] # - 20 videos
    vidIDX = []
    for vidID in VIDEOS_LABELS:
        vidIDX += np.where(vidID == y_vid_test)[0].tolist()
    return np.sort(vidIDX)


RUN_TIME = time.time()
cv_y_pred, cv_y_test, cv_test_acc, cv_test_f1, cv_pred_tm, cv_noCorr, cv_eqlb, cv_wstd, cv_train_tm, cv_test_f1_macro = \
        [], [], [], [], [], [], [], [], [], []
gseed, fold_i = 0, -1
kf = KFold(n_splits=len(Y))  # LOSO
for trainUS, testUS in kf.split(np.arange(len(Y))):  # divide by user  
    fold_i += 1
    print("Fold: ", fold_i, np.unique(Y_S[testUS])[0] )
    if DATASET == "KEmoCon" or DATASET == "RECOLA":
        testUS = testUS[0]
        if DATASET == "RECOLA":
            if np.unique(Y_S[testUS])[0] not in IDsInGrps:
                continue
    
    if np.unique(Y_G[testUS])[0] == -1:
        continue
    assert len(np.unique(Y_G[testUS])) == 1


    if TYPE == "FV" or TYPE == "MORP":
        X_train = np.vstack((X[trainUS]))  # concatenate users
        X_test = np.vstack((X[testUS]))  
    elif TYPE == "IMG":
        X_train = np.hstack((X[trainUS]))  # concatenate users
        X_test = np.hstack((X[testUS]))  
     
    y_train = np.hstack((Y[trainUS]))
    y_test = np.hstack((Y[testUS]))
    
    tm_train = np.hstack((TS[trainUS]))
    tm_test = np.hstack((TS[testUS]))

    y_s_train = np.hstack((Y_S[trainUS]))
    y_s_test = np.hstack((Y_S[testUS]))

    y_g_train = np.hstack((Y_G[trainUS]))
    y_g_test = np.hstack((Y_G[testUS]))
    
    # remove short videos from test - only maintain data from LV
    if DATASET == "AMIGOS": 
        if VIDSELECT == "LVSV":

            y_vid_train = np.hstack((Y_VID[trainUS]))
            y_vid_test = np.hstack((Y_VID[testUS]))

            vidIDX = rmoveLV(y_vid_test)
            
            y_vid_test = y_vid_test[vidIDX]
            y_test = y_test[vidIDX]
            X_test = X_test[vidIDX]
            y_s_test = y_s_test[vidIDX]
            y_g_test = y_g_test[vidIDX]
            tm_test = tm_test[vidIDX]

            vidIDX = rmoveLV(y_vid_train)
            y_vid_train = y_vid_train[vidIDX]
            y_train = y_train[vidIDX]
            X_train = X_train[vidIDX]
            y_s_train = y_s_train[vidIDX]
            y_g_train = y_g_train[vidIDX]
            tm_train = tm_train[vidIDX]
            

    if DATASET == "AMIGOS":
        assert len(np.unique(y_vid_test)) == 4  # 4 long-videos
    assert len(np.unique(y_s_test)) == 1  # 1 test subj 

    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)

    print("training SIZE", X_train.shape)
    t0 = time.time()
    y_pred, wstd, eqLb, noCrr, _y_test = [], [], [], [], []
    _ct, _c_eql, _ct3 = 0, 0, 0
    acc, gp_f1sc, gp_f1sc_m = [], [], []
    NOGP = 0
    for timestmp in range(len(y_test)):  # iterate over timestamp for group dt
        _gp_l, ts, _vid_id, _corr, o_gp_l = [], [], [], [], []
        
        ts += [tm_test[timestmp]]
        subTest = X_test[timestmp]

        #rec = utils.rec_plot(subTest)
        #rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
        #image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
        #subTest = image_rec.resize((224, 224))

        #cm_s = np.asarray(subTest.convert("1"))
        #qa_s = np.nan_to_num(list(rp.recurrence_analysis.recurrence_quantification_analysis(cm_s, 1, 1, 1)[3:9]))   
        #if TYPE == "MORP":
        #    sm_size = int(1 *SR)
        #    subTest = bp.tools.smoother(signal=X_test[timestmp],kernel='boxzen', size=sm_size, mirror=True)[0]
        
        groupID = np.where(np.unique(y_s_test)[0] == IDsInGrps)[0]
        todlt = np.where(IDsInGrps[groupID][0] == np.unique(y_s_test)[0])[0]
        train_s = np.delete(IDsInGrps[groupID][0], todlt)
        for _usH in train_s:  # subjects in train set
            assert np.unique(y_s_test)[0] != _usH 
            # get only subject in same group
            t0 = time.time()
            train_usIDX = np.where(_usH == y_s_train)[0]  # idx of train subj
            if DATASET == "KEmoCon":
                if not len(train_usIDX):
                    print("KEMOCON, no group for", _usH, y_s_test[0])
                    NOGP = 1
                    break
            #assert len(train_usIDX) == len(y_s_test)
            if DATASET == "KEmoCon":
                if timestmp >= len(train_usIDX):
                    break
            subTrain = X_train[train_usIDX][timestmp]
            
            #rec = utils.rec_plot(subTrain)
            #rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
            #image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
            #subTrain = image_rec.resize((224, 224))

            #if TYPE == "MORP":
            #    subTrain = bp.tools.smoother(signal=X_train[train_usIDX][timestmp],kernel='boxzen', size=sm_size, mirror=True)[0]

            
            #_c = np.dot(PXX_train, PXX_test)/(np.linalg.norm(PXX_test)*np.linalg.norm(PXX_train))  # cossine
            #cm_s = np.asarray(subTrain.convert("1"))
            #qa_t = np.nan_to_num(list(rp.recurrence_analysis.recurrence_quantification_analysis(cm_s, 1, 1, 1)[3:9]))
            #_c = np.sum(np.abs(np.array(qa_t) - np.array(qa_s)))
            #_c = 1/(1+_c)
            #_c = np.dot(qa_t, qa_s)/(np.linalg.norm(qa_t)*np.linalg.norm(qa_s))  # cossine
        
            #_c = bp.signals.tools.pearson_correlation(subTrain, subTest)[0]  # norm ??
            #_c = stats.spearmanr(subTrain, subTest)[0]  # norm ??
            _c = np.dot(subTrain, subTest)/(np.linalg.norm(subTrain)*np.linalg.norm(subTest))  # cossine
            #_c = fastdtw(subTrain, subTest)[0]                
            #_c = np.sum(np.abs(np.array(subTrain) - np.array(subTest)))
            #_c = np.max(signal.correlate(subTrain, subTest, mode="full"))  # norm ??
            #_c = scipy.signal.coherence(subTrain, subTest, fs=SR, nperseg=200, noverlap=180)[1][:15].sum()
            
            #_c = 1/(1+_c)

            ts += [tm_train[train_usIDX][timestmp]]
            #_vid_id += [y_vid_train[train_usIDX][timestmp]]

            cv_pred_tm += [time.time() - t0]
            #_c = 0  # activate for non-weighted average              0 
            _c = np.nan_to_num(_c)
            if _c < 0:
                #y_train[train_usIDX[0] + timestmp] = - y_train[train_usIDX][timestmp]
                _gp_l += [-y_train[train_usIDX][timestmp]] # subj ground truth  
                _c = np.abs(_c)
                #_c = 0
            else:
                _gp_l += [y_train[train_usIDX][timestmp]] # subj ground truth  
            o_gp_l += [y_train[train_usIDX][timestmp]]
            _corr += [_c]  # k - NN instead
        if not len(_corr):
            break
        
        assert len(np.unique(ts)) # sme timestamp is being compared
        #assert len(np.unique(_vid_id))  # sme video is being compared
        assert tm_test[timestmp] == np.unique(ts)
        
        d_gp_l = np.array([1 if d>=ANN_TH else -1 for d in o_gp_l]) # transform into -1, 1 label   
        
        if len(np.unique(d_gp_l)) == 1:
            _c_eql += 1
            if DISDT:
                continue
        if not np.sum(_corr):  # if <D-2>no correlation is identified
            N = len(_gp_l)
            _corr = np.array([1/N]*N)
            _ct += 1
        else:
            _corr = _corr/np.sum(_corr)  # Norm
            #_corr = minmax_scale(_corr)  # Norm
            #_corr = np.exp(_corr)/sum(np.exp(_corr))  # autocorr
            _corr = np.nan_to_num(_corr)
        
        y_pred += [np.dot(_corr, _gp_l)]  # results across all timestamps for a group
        _y_test += [y_test[timestmp]]
        
        #y_pred += [_gp_l[np.argmax(_corr)]]  # results across all timestamps for a group
        print("corr y_train y_pred y_test")
        print(_corr, _gp_l, np.dot(_corr, _gp_l), y_test[timestmp])
        if DATASET == "AMIGOS":
            wstd += [np.std(_corr)]
        else:
            wstd += [_corr]
    
    if not len(_corr):
        continue
    eqLb += [_c_eql/len(y_test)*100]
    y_test = _y_test
    assert len(y_pred) == len(y_test)
    noCrr += [_ct/len(y_test)*100]

    print("no corr", _ct/len(y_test)*100)
    print("equal labels", _c_eql/len(y_test)*100)
     
    y_pred = np.array([1 if d>=ANN_TH else -1 for d in y_pred]) # transform into -1, 1 label   
    y_test = np.array([1 if d>=ANN_TH else -1 for d in y_test]) # transform into -1, 1 label   
    
    cv_test_acc += [(y_pred == y_test).astype(np.int).sum()/len(y_pred)*100]
    print("Acc: ", cv_test_acc[-1])
    cv_test_f1 += [f1_score(y_test, y_pred, average='weighted')*100]
    cv_test_f1_macro += [f1_score(y_test, y_pred, average='macro')*100]
    
    cv_noCorr += [noCrr]
    cv_eqlb += [eqLb]
    cv_wstd += [np.mean(wstd)]
    
    

print("#######")
print("Avg acc: ", np.round(np.mean(cv_test_acc), 2), " +- ", np.round(np.std(cv_test_acc), 2))
print("Avg weighted f1 score: ", np.round(np.mean(cv_test_f1), 2), " +- ", np.round(np.std(cv_test_f1), 2))
print("Avg macro f1 score: ", np.round(np.mean(cv_test_f1_macro), 2), " +- ", np.round(np.std(cv_test_f1_macro), 2))
print("Avg Pred time: ", np.round(np.mean(cv_pred_tm), 5), " +- ", np.round(np.std(cv_pred_tm), 5))
print("No Corr: ", np.round(np.mean(cv_noCorr), 2), " +- ", np.round(np.std(cv_noCorr), 2))
print("Weight STD: ", np.round(np.mean(cv_wstd), 2), " +- ", np.round(np.std(cv_wstd), 2))
print("Eq labels: ", np.round(np.mean(cv_eqlb), 2), " +- ", np.round(np.std(cv_eqlb), 2))
print("len cv", len(cv_test_acc))
print("\n")


print("Avg acc: ", np.round(np.mean(cv_test_acc), 2), " +- ", np.round(np.std(cv_test_acc), 2) , " & " , np.round(np.mean(cv_test_f1), 2), " +- ", np.round(np.std(cv_test_f1), 2) , " & " , np.round(np.mean(cv_test_f1_macro), 2), " +- ", np.round(np.std(cv_test_f1_macro), 2), " & " , np.round(np.mean(cv_pred_tm), 5), " +- ", np.round(np.std(cv_pred_tm), 5) , " & " , np.round(np.mean(cv_noCorr), 2), " +- ", np.round(np.std(cv_noCorr), 2) , " & ", np.round(np.mean(cv_wstd), 2), " +- ", np.round(np.std(cv_wstd), 2))



