import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
import biosppy as bp
import sys
import pickle
from sklearn.preprocessing import minmax_scale
import cvxEDA
from scipy import signal
from scipy.fft import fftshift
#import imageio as iio
from PIL import Image
import os
from PIL import Image
from matplotlib import cm
from scipy.spatial.distance import pdist,squareform
import gc
import pandas as pd
from scipy import interpolate
import utils
# Only 37 participants took part in the long videos experiment (participants 8, 24 and 28 were not available)

# also dont have data 

SR = 128
DPI = 300
WINDSC = 20
WIND = WINDSC*SR  
NAME = "normPVid_woutSpks"
DATASET = "RECOLA"

filesDirs = glob.glob("../" + DATASET + "/RECOLA-Biosignals-recordings/*.csv")
METADirs = glob.glob("../" + DATASET + "/RECOLA-Metadata/recola_user_info.csv")[0]

_dirfilter = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/filter/"
try:
    os.makedirs(_dirfilter)
except Exception as e:
    pass
_dircvx = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/cvx/"
try:
    os.makedirs(_dircvx)
except Exception as e:
    pass
_dirECG = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/ECG/"
try:
    os.makedirs(_dirECG)
except Exception as e:
    pass
_dirsRP = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/RP/"
try:
    os.makedirs(_dirsRP)
except Exception as e:
    pass
_dirspect = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/spect/"
try:
    os.makedirs(_dirspect)
except Exception as e:
    pass

def filt_sig(sig, sampling_rate):
    _filt_signal, _, _ = bp.tools.filter_signal(signal=sig,  # filter
                                     ftype='butter',
                                     band='lowpass',
                                     order=4,
                                     frequency=5,
                                     sampling_rate=sampling_rate)  
    sm_size = int(0.25 * sampling_rate) 
    _filt_signal, _ = bp.tools.smoother(signal=_filt_signal,
                                   size=sm_size,
                                   mirror=True)
    return _filt_signal

diCTDt = {}
diCTDt['EDA_cvx'], diCTDt['EDR'] , diCTDt['EDL'] , diCTDt['ECG'] , diCTDt['y_a'] , diCTDt['y_v'], diCTDt['y_s'],\
        diCTDt['y_g'], diCTDt['EDA_RP'], diCTDt['timestamp'], diCTDt['DFT'], diCTDt['EDA_spect'], diCTDt['y_vid'] = [], [], [], [], [], [], [], [], \
        [], [], [], [], []
diCTDt['DATASET'] = "RECOLA"
diCTDt['WIND'] = WINDSC
diCTDt['SR'] = SR

data = pd.read_csv(METADirs, delimiter=";")
groupID = data.Group_id.values[:-3]
groupSID = data.User.values[:-3]

c = 0
for usIDx, USDir in enumerate(filesDirs):
    _ts = 0
    data = pd.read_csv(USDir, delimiter=";")
    _EDA = data.values[:, 1]
    raw_ECG = data.values[:, 2]
    _time = data.values[:, 0]  # app 5 min of data
    
    userID = USDir.split("/")[-1][:-4]
    ## annotations
    ANNDirs_AR = "../" + DATASET + "/RECOLA-Annotation/emotional_behaviour/arousal/"
    ANNDirs_VL = "../" + DATASET + "/RECOLA-Annotation/emotional_behaviour/valence/"
    ANNDirs_AR += userID + ".csv"
    ANNDirs_VL += userID + ".csv"
    print(ANNDirs_AR)

    ann_ar = pd.read_csv(ANNDirs_AR, delimiter=";")
    ann_vl = pd.read_csv(ANNDirs_VL, delimiter=";")

    y_a = np.mean(ann_ar.values[:, 1:], axis=1)
    y_v = np.mean(ann_vl.values[:, 1:], axis=1)
    y_t = ann_vl.values[:, 0]
    
    #SR = len(_time)/(_time[-1]-_time[0])
    
    # filter + norm
    ## ECG
    order = int(0.3 * 1000)
    _ECG, _, _ = bp.tools.filter_signal(signal=raw_ECG,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=1000)

    _ECG = (_ECG - _ECG.mean())/_ECG.std()
    f = interpolate.interp1d(_time, _ECG)
    xnew = np.arange(_time[0], _time[-1], 1/SR)
    _ECG = f(xnew)
    f = interpolate.interp1d(_time, raw_ECG)
    raw_ECG = f(xnew)

    ## EDA
    plt.figure()
    plt.plot(_time, _EDA, label="raw")
    _EDA = filt_sig(_EDA, 1000)  # filter signal
    f = interpolate.interp1d(_time, _EDA)
    xnew = np.arange(_time[0], _time[-1], 1/SR)
    _EDA = f(xnew)

    plt.plot(xnew, _EDA, label="filtered + smooth + res")
    plt.legend()
    plt.savefig(_dirfilter + str(usIDx) + "filter_EDA.pdf", format="pdf")
    plt.close('all')
    
    inputS = _EDA.astype(np.float64).copy()
    inputS = (inputS - inputS.mean())/inputS.std()
    [cvxEDR, _, cvxEDL, _, _, _, _] = cvxEDA.cvxEDA(inputS, 1./SR)  # cvxEDA only accepts double types, needs to be norm
    
    cvxEDR = (cvxEDR - cvxEDR.mean())/cvxEDR.std()
    cvxEDL = (cvxEDL - cvxEDL.mean())/cvxEDL.std()
    
    # segment
    diCTDt['EDA_cvx'] += [inputS[i:i+WIND] for i in range(0, len(inputS)-WIND, WIND - int(0.75*WIND))]
    diCTDt['EDR'] += [cvxEDR[i:i+WIND] for i in range(0, len(cvxEDR)-WIND, WIND - int(0.75*WIND))]
    diCTDt['EDL'] += [cvxEDL[i:i+WIND] for i in range(0, len(cvxEDL)-WIND, WIND - int(0.75*WIND))]
    diCTDt['ECG'] += [_ECG[i:i+WIND] for i in range(0, len(_ECG)-WIND, WIND - int(0.75*WIND))]

    # us ID
    N = len(list(range(0, len(cvxEDL)-WIND, WIND - int(0.75*WIND))))
    assert N > 0
    diCTDt['y_s'] += [userID]*N
    userGID = groupID[np.where(groupSID == float(userID[1:]))[0]]
    diCTDt['y_g'] += [userGID]*N
    diCTDt['y_vid'] += [usIDx]*N

    assert (ann_vl.values[:, 0] == ann_ar.values[:, 0]).all()
    ## interpolate ann so it has same size as data
    f = interpolate.interp1d(y_t, y_a)
    _y_a = f(xnew)
    f = interpolate.interp1d(y_t, y_v)
    _y_v = f(xnew)

    #annWIND = 1*SR
    #ann_Dt = [inputS[i:i+annWIND] for i in range(0, len(inputS), annWIND)]
    #_y_a, _y_v, _t_ann = [], [], []
    #for w in range(len(ann_Dt)):
    #    NW = len(ann_Dt[w])
    #    _y_a += [y_a[w]] * NW
    #    _y_v += [y_v[w]] * NW
    #assert len(_y_a) == len(np.hstack((ann_Dt)))
    assert len(_y_a) == len(inputS)
    
    for w in diCTDt["EDA_cvx"][-N:]:
        assert len(w) == WIND

    diCTDt['y_a'] += [np.mean(_y_a[i:i+WIND]) for i in range(0, len(_y_a)-WIND, WIND - int(0.75*WIND))]
    diCTDt['y_v'] += [np.mean(_y_v[i:i+WIND]) for i in range(0, len(_y_v)-WIND, WIND - int(0.75*WIND))]
    
    assert len(diCTDt['y_a']) == len(diCTDt['y_v'])
    assert len(diCTDt['y_s']) == len(diCTDt['y_v'])
    assert len(diCTDt['y_g']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDA_cvx']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDR']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDL']) == len(diCTDt['y_v'])
    assert len(diCTDt['ECG']) == len(diCTDt['y_v'])


    for windIdx in range(0, len(inputS)-WIND, WIND - int(0.75*WIND)):  
        plt.figure()
        plt.plot(inputS[windIdx:windIdx+WIND], label="norm EDA")
        plt.plot(cvxEDR[windIdx:windIdx+WIND], label="norm EDR")
        plt.plot(cvxEDL[windIdx:windIdx+WIND], label="norm EDL")
        plt.legend()
        plt.savefig(_dircvx + str(c) + "_cvx.pdf", format="pdf")
        plt.close('all')

        # # RP
        rec = utils.rec_plot(inputS[windIdx:windIdx+WIND])
        rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
        image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
        image_rec = image_rec.resize((224, 224))
        image_rec.save(_dirsRP + str(c) + "_RP.png")
        diCTDt['EDA_RP'] += [image_rec]

   #    # # spect
        t, f, Sxx = signal.spectrogram(inputS[windIdx:windIdx+WIND], fs=SR, 
                     nperseg=200, noverlap=180, return_onesided=True)
        FCUT = 15      
        Sxx = np.log(Sxx[:FCUT])
         
        Sxx_or = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
        image_Sxx = Image.fromarray(np.uint8(cm.viridis(Sxx_or)*255))
        image_Sxx = image_Sxx.resize((224, 224))
        image_Sxx.save(_dirspect + str(c) + "_spect.png")     
        diCTDt['EDA_spect'] += [image_Sxx] 

        #spectrum = np.log(np.abs(np.fft.fft(inputS[windIdx:windIdx+WIND])))
        #spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
        #f = np.nan_to_num(np.linspace(0, SR/2, len(spectrum)))
        #diCTDt['DFT'] += [spectrum]

        #plt.figure()
        #plt.plot(f, spectrum, label="spectrum")
        #plt.savefig(_dirspect + str(c) + "_spectrum.png", format="png")
        #plt.close('all')
        onsets = np.array(bp.signals.ecg.get_rpks(_ECG[windIdx:windIdx+WIND], SR))

        plt.figure()
        plt.plot(raw_ECG[windIdx:windIdx+WIND], label="raw ECG")
        plt.plot(_ECG[windIdx:windIdx+WIND], label="filt + res BVP")
        plt.vlines(onsets, _ECG[windIdx:windIdx+WIND].min(), _ECG[windIdx:windIdx+WIND].max(), color="r")
        plt.legend()
        plt.savefig(_dirECG + str(c) + "_BVP.png", format="png")
        plt.close('all')

        diCTDt['timestamp'] += [_ts]
        _ts += 1
        c += 1
    assert len(diCTDt['timestamp']) == len(diCTDt['EDA_spect'])
    #assert len(diCTDt['timestamp']) == len(diCTDt['DFT'])
    assert len(diCTDt['timestamp']) == len(diCTDt['EDA_RP'])
    assert len(diCTDt['timestamp']) == len(diCTDt['y_a'])
    assert len(diCTDt['timestamp']) == len(diCTDt['y_vid'])

    print(len(diCTDt["y_a"]))

for k in diCTDt.keys():
    print(k)
    diCTDt[k] = np.array(diCTDt[k])

_dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"
try:
    os.makedirs(_dir)
except Exception as e:
    pass

with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(diCTDt, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("over")
