import torch
from torch import nn
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
import resNet

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

WINDSC = 20
DATASET = "AMIGOS"


#inputX = np.hstack((RR_FV, EDA_FV))

if DATASET == "AMIGOS":
    VIDSELECT = "LVSV"
    NAME = "normPVid_woutSpks"
    VID = "LVSVO"
    
    _dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"
    _dirres ="../Results/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/Morph/"
    
else:
    _dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"
    _dirres ="../Results/" + DATASET + "/RR_redstfv/"

try:
    os.makedirs(_dirres)
except:
    pass

with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

TYPE = "MORP"  # FV, IMG, MORP
#inputX = data['EDA_redstfv'] #"EDA_redstfv" RR_redstfv
#inputX = np.hstack((data['EDA_redstfv'], data['RR_redstfv'])) #"EDA_redstfv" RR_redstfv
#print("INPUT SZ", inputX.shape)

del data['EDA_RP'] 
del data['EDA_spect']
#del data['EDA_cvx'] 
#del data['EDR'] 
#del data['EDL']
del data['ECG']
del data['EDA_stfv']
del data['EDA_fv_h']
del data['EDA_redstfv']
del data['RR_stfv']
del data['RR_fv_h']
del data['RR_redstfv']

gc.collect()

SR = data['SR']  # sampling rate
X_train, y_train, X_test, y_test = [], [], [], []


def resample(_signal):
    n_s = []
    for sidx in range(len(_signal)):
        x = np.linspace(0, (len(_signal[sidx])-1)/SR, num=len(_signal[sidx]))
        f = interpolate.interp1d(x, _signal[sidx])
        #xnew = np.linspace(0, x[-1], num=2500)  # resample to fit resnet
        xnew = np.linspace(0, (len(x)-1)/SR, num=2500)  # resample to fit resnet
        n_s += [f(xnew)]
    return np.array(n_s)


if TYPE == "FV":
    result = np.zeros((len(inputX), 224))
    result[:inputX.shape[0],:inputX.shape[1]] = inputX
    inputX = result
elif TYPE == "MORP":
    resEDA = resample(data['EDA_cvx'].copy()).astype(np.float32)
    resEDR = resample(data['EDR'].copy()).astype(np.float32)
    resEDL = resample(data['EDL'].copy()).astype(np.float32)
    inputX = np.array([resEDA, resEDR, resEDL])
    inputX = np.dstack((inputX))



def train(dataloader, _model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    _model.train()
    train_loss = []
    counter = 0
    for batch, (X, y) in enumerate(dataloader):
        counter += 1
        #X = X.reshape(len(X), 1, -1)
        
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = _model(X)
        
        loss = loss_fn(pred.squeeze(1), y)
        
        #loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        #if batch % 100 == 0:  # shows only first batch
        train_loss.append(loss)
        
        #for name, param in model.named_parameters():
        #    print(name, param.grad.abs().sum())
        #print("pred", pred)
    loss, current = loss.item(), batch * len(X)
    
    pred = pred.cpu().detach().squeeze(1)
    pred = torch.round(torch.sigmoid(pred))
    #pred = np.argmax(pred, axis=1)
    y = y.cpu().detach()
    
    acc = (pred == y).type(torch.float).sum().item()/len(y) 

    f_m_train = f1_score(pred, y, average='macro')
    #print("batch", batch)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", "lr", optimizer.param_groups[0]['lr'], batch)
    
    mean_loss = torch.tensor(train_loss).mean().item()
    #print('Training loss: %.4f' % (mean_loss))
    print("Train Acc:", acc)
    print("Train y unq", np.unique(y, return_counts=True)) 
    return mean_loss, optimizer.param_groups[0]['lr'], acc, _model, f_m_train



def test(dataloader, _model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    _model.eval()
    test_loss, correct = 0, 0
    y_test, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            #X = X.reshape(len(X), 1, -1)
            X, y = X.to(device), y.to(device)
            
            pred = _model(X)
            test_loss += loss_fn(pred.squeeze(1), y).item()
             
            pred = pred.cpu().detach().squeeze(1)
            #pred = np.argmax(pred, axis=1)
            pred = torch.round(torch.sigmoid(pred))
            y = y.cpu().detach()#.unsqueeze(1)
            
            correct += (pred == y).type(torch.float).sum().item()

            y_test += y.numpy().tolist()
            y_pred += pred.numpy().ravel().tolist()
            
    
    test_loss /= num_batches
    correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, correct, y_test, y_pred


class customDataset_FV(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # float for binary CE; long for cross entropy
        if TYPE == "MORP":
            self.X = torch.moveaxis(self.X, 2, 1)  
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

class customDataset_IMG(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)  # float for binary CE; long for cross entropy
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): 
        X = np.asarray(self.X[idx])[:, :, :3]
        #X1 = np.asarray(self.X[idx][0])[:, :, :3]
        #X2 = np.asarray(self.X[idx][1])[:, :, :3]
        #X = np.dstack((X1, X2))
        X = ToTensor()(X.astype(np.float32))
        
        return X, self.y[idx]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


inputY = data['y_a'].copy()
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

    y += [[1 if d >= 0 else 0 for d in inputY[sIDX]]]
    #y += [inputY[sIDX]]
    Y_S += [data['y_s'][sIDX]]
    Y_VID += [data['y_vid'][sIDX]]
    TS += [data['timestamp'][sIDX]]
    Y_G += [data['y_g'][sIDX]]
    #y += [[d if d >= TH else d for d in inputY[sIDX]]]

    #_sidx = samplsIDX[sIDX]  # user samples8
X = np.array(X)
Y = np.array(y)
Y_S = np.array(Y_S)
Y_VID = np.array(Y_VID)
TS = np.array(TS)
Y_G = np.array(Y_G)


def train_model(config, input_size, training_data, val_dataloader, sampler):
    if TYPE == "FV":
        #model = models.FCN(X_train.shape[1]).to(device)
        model = resNet_lin.resNet(input_size, [2, 2, 2, 2]).to(device)
    elif TYPE == "IMG":
        model = models.resnet18(pretrained=True,).to(device)
        model.fc = nn.Linear(512, 1)
        #model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif TYPE == "MORP":
        model = AMIGOS_model.Model(use_attention="True", use_non_local="True", num_class=1).to(device)
        #model = resNet.resNet(3, [2, 2, 2, 2]).to(device)
    
    train_dataloader = DataLoader(training_data, batch_size=config["batch_size"], shuffle=False, sampler=sampler, num_workers=0, drop_last=True)

    #optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["w_d"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['w_d'])
    best_score, train_best_score = None, 0.9
    delta, counter, patience, train_patience, train_counter, val_counter = 0.0, 0, config["patience"], 4, 0, 0
    epochs = config["epoch"]
    rol_fm, alp = [], 2/6
    _f_m, _acc, _train_loss, _val_loss, _train_acc, _fm_train, _rol_fm = [] ,[], [],[], [], [], []
    for t in range(epochs):
        print(f"Epoch {t+1} -------------------------------")
        
        train_loss, train_lr, train_acc, model, fm_train = train(train_dataloader, model, loss_fn, optimizer)
        val_loss, acc, _y_test, _y_pred = test(val_dataloader, model)
        f_m = f1_score(_y_test, _y_pred, average='macro')
        print("val fm", f_m)
        if not t:
            rol_fm = 0
        else:
            rol_fm = alp * f_m + (1-alp)*rol_fm
    
        _f_m += [f_m]
        _acc += [acc]
        _train_loss += [train_loss]
        _val_loss += [val_loss]
        _train_acc += [train_acc]
        _fm_train += [fm_train]
        _rol_fm += [rol_fm]

        # stops when val loss stops decreasing
        if t >= config['grace_time']:
            score = rol_fm
            if best_score is None:
                best_score = score
            elif score < best_score + delta:
                counter += 1
                if counter >= patience:
                    print("early stop")                 
                    break
            else:
                best_score = score
                counter = 0

            if fm_train > train_best_score: 
                train_counter += 1
                if train_counter >= train_patience:
                    print("train acc early stop")                 
                    break
            else:
                train_counter = 0
    report={"f1_m": _f_m, "mean_accuracy": _acc, "train_loss": _train_loss, "val_loss":_val_loss, "train_acc":_train_acc, "fm_train": _fm_train, "rol_fm":_rol_fm}

    return model, report


def rmoveLV(y_vid_test):
    VIDEOS_LABELS = ["N1", "P1", "B1", "U1"] # - 20 videos
    vidIDX = []
    for vidID in VIDEOS_LABELS:
        vidIDX += np.where(vidID == y_vid_test)[0].tolist()
    return np.sort(vidIDX)

IDsInGrps = np.array([["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"], ["P15", "P11", "P12", "P10"]])

cpIDsGP = IDsInGrps.copy()

_, idx = np.unique(np.hstack((Y_S)), return_index=True)
SUBJ_ORD = np.hstack((Y_S))[np.sort(idx)]
assert len(SUBJ_ORD) == len(Y)
import itertools

RUN_TIME = time.time()
checkpoint = {}
checkpoint["model"] = {}
checkpoint["fold_i"] = []
cv_y_pred, cv_y_test, cv_test_acc, cv_test_f1, cv_pred_tm, cv_noCorr, cv_eqlb, cv_wstd, cv_train_tm, cv_test_f1_macro = \
        [], [], [], [], [], [], [], [], [], []
gseed, fold_i = 0, -1
kf = KFold(n_splits=len(Y))  # LOSO
for trainUS, testUS in kf.split(np.arange(len(Y))):  # divide by user  
    fold_i += 1
    print("Fold: ", fold_i)
    






    train_us_byf = np.array([np.unique(Y_S[f])[0] for f in trainUS])
    assert len(train_us_byf) == 36

    if np.unique(Y_G[testUS])[0] == -1:
        continue
    assert len(np.unique(Y_G[testUS])) == 1
     
    groupID = np.where(np.unique(Y_S[testUS]) == IDsInGrps)[0]
    todlt = np.where(IDsInGrps[groupID][0] == np.unique(Y_S[testUS])[0])[0]  # delete test us from gp
    train_s = np.delete(IDsInGrps[groupID][0], todlt)  # list with group users for training + val

    assert SUBJ_ORD[testUS] == np.unique(Y_S[testUS])[0]
    
    us_to_rtrainIDX = []
    for rmU in train_s:
        us_to_rtrainIDX += [np.where(rmU == train_us_byf)[0][0]]  # remove remaining group users from training  
    
    trainUS = np.delete(trainUS, us_to_rtrainIDX)  # must have all users \ val + test

    np.random.seed(fold_i)
    val_IDX = np.random.choice(SUBJ_ORD[trainUS])

    valIDX = np.where(val_IDX == SUBJ_ORD)[0]  # fold idx of validation user

    train_us_byf = np.array([np.unique(Y_S[f])[0] for f in trainUS])
    v_trainIDX = np.where(val_IDX == train_us_byf)[0]
    trainUS = np.delete(trainUS, v_trainIDX)  # must have all users \ val + test
    
    assert len(trainUS) == 32
    assert SUBJ_ORD[testUS] not in SUBJ_ORD[trainUS]
    assert SUBJ_ORD[valIDX] not in SUBJ_ORD[trainUS]

    print("########################")
    print("group: ", IDsInGrps[groupID])  
    print("test user", testUS, SUBJ_ORD[testUS])
    print("val user", valIDX, SUBJ_ORD[valIDX])
    print("train user", SUBJ_ORD[trainUS])  

    assert len(trainUS) + len(valIDX) + len(testUS) + 3 == 37
    
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

    #tm_train = np.hstack((TS[trainUS]))
    #tm_test = np.hstack((TS[testUS]))

    y_s_train = np.hstack((Y_S[trainUS]))
    y_s_test = np.hstack((Y_S[testUS]))
    y_s_val = np.hstack((Y_S[valIDX]))

    y_vid_train = np.hstack((Y_VID[trainUS]))
    y_vid_test = np.hstack((Y_VID[testUS]))
    y_vid_val = np.hstack((Y_VID[valIDX]))
    
    # remove short videos from test - only maintain data from LV
    if DATASET == "AMIGOS": 
        if VIDSELECT == "LVSV":
            vidIDX = rmoveLV(y_vid_test)
            y_vid_test = y_vid_test[vidIDX]
            y_test = y_test[vidIDX]
            X_test = X_test[vidIDX]
            y_s_test = y_s_test[vidIDX]

          #  vidIDX = rmoveLV(y_vid_train)
          #  y_vid_train = y_vid_train[vidIDX]
          #  y_train = y_train[vidIDX]
          #  X_train = X_train[vidIDX]
          #  y_s_train = y_s_train[vidIDX]

            vidIDX = rmoveLV(y_vid_val)
            y_vid_val = y_vid_val[vidIDX]
            y_val = y_val[vidIDX]
            X_val = X_val[vidIDX]
            y_s_val = y_s_val[vidIDX]
    if DATASET == "AMIGOS":
        assert len(np.unique(y_vid_test)) == 4  # 4 long-videos
        assert len(np.unique(y_vid_val)) == 4  # 4 long-videos
    assert len(np.unique(y_s_val)) == 1  # 1 val subj 
    assert len(np.unique(y_s_test)) == 1  # 1 test subj 

    assert len(X_test) == len(y_test)
    assert len(X_val) == len(y_val)

    print("SIZE", X_train.shape)

    if TYPE == "FV" or TYPE == "MORP":
        training_data = customDataset_FV(X_train, y_train)
        test_data = customDataset_FV(X_test, y_test)  
        val_data = customDataset_FV(X_val, y_val)  
    elif TYPE == "IMG":
        training_data = customDataset_IMG(X_train, y_train)
        test_data = customDataset_IMG(X_test, y_test)  
        val_data = customDataset_IMG(X_val, y_val)  
    assert len(X_val) == len(X_test)
    class_weights = 1./np.unique(y_train, return_counts=True)[1]
    weights = [class_weights[int(t)] for t in y_train]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(y_train), replacement=True)

    val_dataloader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

    #loss_fn = torch.nn.CrossEntropyLoss().to(device)
    #num_positives = np.sum(y_train)
    #num_negatives = len(y_train) - num_positives
    #pos_weight = torch.tensor(num_negatives / num_positives).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0)
    
    search_space = {
    "w_d": tune.grid_search([0.0001, 0.05]),
    "lr": tune.grid_search([1e-5, 1e-3]),
    "patience": tune.grid_search([5, 50]),
    "batch_size": tune.grid_search([16, 256])  # 128 for features
    }
    if TYPE == "MORP":
        search_space = {
        "w_d": 0.01,
        "lr": 1e-3,
        "patience": 5,
        "batch_size": 256,  # 128 for features
        "grace_time": 400,
        "epoch": 50
        }


    if TYPE=="IMG":
        input_size = 2
    elif TYPE=="FV" or TYPE == "MORP":
        input_size = X_train.shape[1]
    t0 = time.time() 
    model, report = train_model(config=search_space, input_size=input_size, training_data=training_data, val_dataloader=val_dataloader, sampler=sampler)
    
    print("#### Test set V")
    loss, acc, _y_test, _y_pred = test(test_dataloader, model)
    print("Test set", testUS)   
    print(">>>acc: ", acc)
    print("_y_pred", _y_pred)
    print("_y_test", _y_test)
    print(classification_report(_y_test, _y_pred, labels=np.unique(_y_test)))
    
    cv_train_tm += [time.time() - t0]
    cv_test_acc += [acc*100]
    cv_test_f1 += [f1_score(_y_test, _y_pred, average='weighted')*100]
    cv_test_f1_macro += [f1_score(_y_test, _y_pred, average='macro')*100]
    cv_y_test += _y_test
    cv_y_pred += _y_pred

    EPOCH = len(report['f1_m'])
    plt.figure()
    plt.ylabel("Magnitude")
    plt.plot(np.arange(EPOCH), report['train_acc'], label='Train Acc', c="g")
    plt.plot(np.arange(EPOCH), report['train_loss'], label='Training Loss', c="b")
    plt.plot(np.arange(EPOCH), report['val_loss'], label='Val Loss', c="c")
    plt.plot(np.arange(EPOCH), report['mean_accuracy'], label='Val Acc', c="y")
    plt.plot(np.arange(EPOCH), report['f1_m'], label='Val F1-macro', c="m")
    plt.plot(np.arange(EPOCH), report['rol_fm'], label='Val F1-macro rol', c="r")
    plt.plot(EPOCH-1, acc, ".", label="Test Acc", c="k")
    plt.plot(EPOCH-1, cv_test_f1_macro[-1]*0.01, ".", label="Test f1-macro", c="r")
    plt.hlines(1, 0, EPOCH-1)
    
    plt.legend(bbox_to_anchor=(1.15, 1.05))
    plt.savefig(_dirres + str(fold_i) + "_best_rs.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close('all')
    # plot
    checkpoint["cv_test_acc"] = cv_test_acc
    checkpoint["cv_test_f1"] = cv_test_f1
    checkpoint["cv_y_test"] = cv_y_test
    checkpoint["cv_y_pred"] = cv_y_pred
    checkpoint["fold_i"] = fold_i
    checkpoint["model"][str(fold_i)] = model.state_dict()
    checkpoint["train_tm"] = cv_train_tm 

    with open(_dirres + 'checkpoint.pickle', 'wb') as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        f = open(_dirres + 'results' + str(RUN_TIME) + '.txt', 'a')
    except Exception as e:
        print(e)
        f = open(_dirres + 'results' + str(RUN_TIME) + '.txt', 'w')

    f.write(str(fold_i) + "," + str(np.mean(cv_test_acc)) + "," + str(np.std(cv_test_acc)) + "\n")
    f.write(str(fold_i) + "," + str(np.mean(cv_test_f1)) + "," + str(np.std(cv_test_f1)) + "\n")
    f.write(str(fold_i) + "," + str(np.mean(cv_test_f1_macro)) + "," + str(np.std(cv_test_f1_macro)) + "\n")
    f.write(str(fold_i) + "," + str(np.mean(cv_train_tm)) + "," + str(np.std(cv_train_tm)) + "\n")
    f.close()  
    
    


f = open(_dirres + 'results' + str(RUN_TIME) + '.txt', 'a')
f.write("cv" + "," + str(np.mean(cv_test_acc)) + "," + str(np.std(cv_test_acc)) + "\n")
f.write("cv" + "," + str(np.mean(cv_test_f1)) + "," + str(np.std(cv_test_f1)) + "\n")
f.write("cv" + "," + str(np.mean(cv_test_f1_macro)) + "," + str(np.std(cv_test_f1_macro)) + "\n")
f.write("cv" + "," + str(np.mean(cv_train_tm)) + "," + str(np.std(cv_train_tm)) + "\n")
f.close()    
    
print("Avg acc: ", np.round(np.mean(cv_test_acc), 2), " +- ", np.round(np.std(cv_test_acc), 2))
print("Avg weighted f1 score: ", np.round(np.mean(cv_test_f1), 2), " +- ", np.round(np.std(cv_test_f1), 2))
print("Avg macro f1 score: ", np.round(np.mean(cv_test_f1_macro), 2), " +- ", np.round(np.std(cv_test_f1_macro), 2))
print("Avg Pred time: ", np.round(np.mean(cv_train_tm), 5), " +- ", np.round(np.std(cv_train_tm), 5))

print("\n")
print("Eq labels: ", np.round(np.mean(cv_eqlb), 2), " +- ", np.round(np.std(cv_eqlb), 2))

print(labels, np.round(np.unique(inputY, return_counts=True)[1]/len(inputY)*100, 2))
