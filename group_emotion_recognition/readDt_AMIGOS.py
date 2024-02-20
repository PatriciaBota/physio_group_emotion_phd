import scipy.io
import glob
import numpy as np
import matplotlib.pyplot as plt
import biosppy as bp
import sys
import pickle
from sklearn.preprocessing import minmax_scale
import cvxEDA
from scipy.fft import fftshift
#import imageio as iio
from PIL import Image
import os
from matplotlib import cm
import gc
from scipy import signal
import utils
import seaborn as sns
sns.set(font_scale=2)  # crazy big
# Only 37 participants took part in the long videos experiment (participants 8, 24 and 28 were not available)

# also dont have data 

SR = 128
DPI = 300
NAME = "normPVid_woutSpks"
DATASET = "AMIGOS"
VID = "LVSVO"

filesDirs = glob.glob("../" + DATASET + "/Data_Processed/*.mat")

WINDSC = 20
WIND = WINDSC*SR   

diCTDt = {}
diCTDt['EDA_cvx'], diCTDt['EDR'] , diCTDt['EDL'] , diCTDt['ECG'] , diCTDt['y_a'] , diCTDt['y_v'], diCTDt['y_s'],\
        diCTDt['y_g'], diCTDt['EDA_RP'], diCTDt['timestamp'], diCTDt['EDA_spect'], diCTDt['y_vid']  = [],\
        [], [], [], [], [], \
        [], [], [], [], [], []
diCTDt['DATASET'] = "RECOLA"
diCTDt['WIND'] = WINDSC
diCTDt['SR'] = SR

## set group IDX
IDsInGrps = np.array([["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"]
            , ["P15", "P11", "P12", "P10"]])


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


def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores

def reemovSpks(sig):
    dist = 0
    delta_intensity = [] 
    for i in np.arange(len(sig)-1):
        # https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
        dist = sig[i+1] - sig[i]
        delta_intensity.append(dist)
    delta_int = np.array(delta_intensity)

    # 1 is assigned to spikes, 0 to non-spikes:
    spikes = abs(np.array(modified_z_score(delta_int))) > 4
    
    n_p = spikes.copy()
    for i in range(len(spikes)):
        if spikes[i]:
            continue
        if i < len(spikes)-1:
            if spikes[i+1]:
                n_p[i] = True
        elif i < len(spikes)-2:
            if spikes[i+2]:
                n_p[i] = True

        elif i > 0:
            if spikes[i-1]:
                n_p[i] = True
            if spikes[i-2]:
                n_p[i] = True 

    spikes = np.array(n_p)
    to_delete = np.where(spikes == True)
    new_s = sig.copy()
    new_s[to_delete] = np.nan

    _filt_signal = np.interp(np.arange(len(new_s)), 
          np.arange(len(new_s))[np.isnan(new_s) == False], 
          new_s[np.isnan(new_s) == False])
    
    return _filt_signal

_dirfilter = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/spikes/"
try:
    os.makedirs(_dirfilter)
except Exception as e:
    pass
_dircvx = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/cvx/"
try:
    os.makedirs(_dircvx)
except Exception as e:
    pass
_dirRP = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/RP/"
try:
    os.makedirs(_dirRP)
except Exception as e:
    pass
_dirspect = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/spect/"
try:
    os.makedirs(_dirspect)
except Exception as e:
    pass
_dirECG = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/ECG/"
try:
    os.makedirs(_dirECG)
except Exception as e:
    pass



VIDEOS_LABELS = ["N1", "P1", "B1", "U1"] # - 20 videos
IDsInGrps = np.array([["P07", "P01", "P02", "P16"], ["P06", "P32", "P04", "P03"], ["P29", "P05", "P27", "P21"], ["P18", "P14", "P17", "P22"]
            , ["P15", "P11", "P12", "P10"]]) 

VID_LEN = ["4", "5", "9", "10", "13", "18", "19", "20", "23", "30", "31", "34", "36", "58", "80", "138", "N1", "P1", "B1", "U1"]
VID_LEN_C = [6, 7, 5, 6, 4, 5, 8, 5, 7, 5, 9, 5, 5, 4, 6, 7, 72, 58, 72, 44]
c = 0
for dirIDX, fileDir in enumerate(filesDirs):  # iterate over users
    userID = fileDir.split("_")[-1].split(".mat")[0]
    
    
    gID = np.where(userID == IDsInGrps)[0]
    if not len(gID):  # individual acq
        gID = -1
    else:
        gID = gID[0]

    file = scipy.io.loadmat(fileDir)
    # f: dict_keys(['__header__', '__version__', '__globals__', 'VideoIDs', 'joined_data', 'labels_ext_annotation', 'labels_selfassessment'])
    videoLst = np.hstack((file['VideoIDs'][0])).tolist()  # get idx of videos of interest
    _ts = 0
    for vdIdx in range(len(videoLst)):
        videoIDX = videoLst[vdIdx]

        if not len(file["labels_ext_annotation"][0][vdIdx][0]):
            print("No data for user", userID, "video", videoIDX, len(file["labels_ext_annotation"][0][vdIdx][0]))
            continue
        # segment_index, valence and arousal
        dt = file["joined_data"][0][vdIdx]
        
        if not len(dt):
            print("No data for user", userID, "video", videoIDX, len(file["labels_ext_annotation"][0][vdIdx][0]))
            continue

        st = 5*SR
        _EDA = dt[st:, -1]
        _ECG = dt[st:, -2]
        
        # filter signal
        order = int(0.3 * SR)
        filt_ECG, _, _ = bp.tools.filter_signal(signal=_ECG,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=SR)

        filt_ECG = (filt_ECG - filt_ECG.mean())/filt_ECG.std()

        plt.figure()
        plt.plot(_EDA, label="raw")
        _EDA = filt_sig(_EDA, SR)  # filter signal
        plt.plot(_EDA, label="filtered + smooth")

        _EDA = reemovSpks(_EDA)  # remove spikes
        plt.plot(_EDA, label="without spikes")

        plt.legend()
        plt.savefig(_dirfilter + str(c) + "filter_EDA.pdf", format="pdf")
        plt.close('all')
        
        inputS = _EDA.astype(np.float64).copy()
        inputS = (inputS - inputS.mean())/inputS.std()
        [cvxEDR, _, cvxEDL, _, _, _, _] = cvxEDA.cvxEDA(inputS, 1./SR)  # cvxEDA only accepts double types, needs to be norm
        
        cvxEDR = (cvxEDR - cvxEDR.mean())/cvxEDR.std()
        cvxEDL = (cvxEDL - cvxEDL.mean())/cvxEDL.std()
        
        annWIND = 20*SR
        _EDA_cvx = [inputS[i:i+annWIND] for i in range(0, len(inputS) - annWIND, annWIND)]
        max_t =  (len(_EDA)-1)/SR
        c_v2 = max_t/20

        subj_valence = file["labels_ext_annotation"][0][vdIdx][1:-1, 1]  # excluded first and last clip -- only for 20s window
        subj_arousal = file["labels_ext_annotation"][0][vdIdx][1:-1, 2]  # excluded first and last clip-- only for 20s window
        ann_clips_IDX = file["labels_ext_annotation"][0][vdIdx][:, 0]  # excluded first and last clip-- only for 20s window
        
        assert (np.sort(ann_clips_IDX) == ann_clips_IDX).all()
        vc_i = np.where(np.array(VID_LEN) == videoIDX)[0][0]
        assert len(_EDA_cvx) == len(subj_arousal)
        assert VID_LEN_C[vc_i]-2 == len(subj_arousal)
        
        N = 0
        _y_a, _y_v = [], []
        for w in range(len(_EDA_cvx)):
            NW = len(_EDA_cvx[w])
            _y_a += [subj_arousal[w]] * NW
            _y_v += [subj_valence[w]] * NW

        assert len(_y_a) == len(np.hstack((_EDA_cvx)))
        s_N = list(range(0, len(inputS), annWIND))[-1] #+ annWIND
        #assert len(inputS) == len(_y_a)  not equal

        assert len(np.hstack((_EDA_cvx))) == s_N
        s_N = list(range(0, len(_y_a) - WIND, WIND-int(0.75*WIND)))[-1] + WIND-int(0.75*WIND)
         
        N = len(list(range(0, s_N, WIND - int(0.75*WIND))))
        assert N > 0
        assert N == len([np.mean(_y_v[i:i+WIND]) for i in range(0, s_N, WIND - int(0.75*WIND))])
        assert N == len([np.mean(_y_v[i:i+WIND]) for i in range(0, len(_y_a) - WIND, WIND - int(0.75*WIND))])

        for i in range(0, s_N, WIND - int(0.75*WIND)):
            print(i, len(_y_a[i:i+WIND]))

        diCTDt['EDA_cvx'] += [inputS[i:i+WIND] for i in range(0, s_N, WIND - int(0.75*WIND))]
        diCTDt['EDR'] += [cvxEDR[i:i+WIND] for i in range(0, s_N, WIND - int(0.75*WIND))]
        diCTDt['EDL'] += [cvxEDL[i:i+WIND] for i in range(0, s_N, WIND - int(0.75*WIND))]
        diCTDt['ECG'] += [filt_ECG[i:i+WIND] for i in range(0, s_N, WIND - int(0.75*WIND))]
        diCTDt['y_a'] += [np.mean(_y_a[i:i+WIND]) for i in range(0, s_N, WIND - int(0.75*WIND))]
        diCTDt['y_v'] += [np.mean(_y_v[i:i+WIND]) for i in range(0, s_N, WIND - int(0.75*WIND))]
        
        diCTDt['y_vid'] += [videoIDX]*N
        diCTDt['y_s'] += [userID]*N
        diCTDt['y_g'] += [gID]*N
        assert videoIDX == np.array(videoLst)[vdIdx]

        for s in diCTDt['EDR'][-N:]:
            assert len(s) == WIND

        print("Subj", userID, "video", videoIDX)
        
        for i in range(0, s_N, WIND - int(0.75*WIND)):
            plt.figure(dpi=300)
            plt.plot(inputS[i:i+WIND], label="EDA")
            plt.plot(cvxEDR[i:i+WIND], label="EDR")
            plt.plot(cvxEDL[i:i+WIND], label="EDL")
            plt.ylabel("A.U. (Normalized)")
            plt.xlabel("Samples")
            plt.legend()
            plt.savefig(_dircvx + str(c) + "_cvx.pdf", bbox_inches='tight', format="pdf")
            plt.close('all')

            # # RP
            rec = utils.rec_plot(inputS[i:i+WIND])
            rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
            image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
            image_rec = image_rec.resize((224, 224))
            image_rec.save(_dirRP + str(c) + "_RP.png")
            diCTDt['EDA_RP'] += [image_rec]

       #    # # spect
            t, f, Sxx = signal.spectrogram(inputS[i:i+WIND], fs=SR, 
                         nperseg=200, noverlap=180, return_onesided=True)
            FCUT = 15      
            Sxx = np.log(Sxx[:FCUT])
             
            Sxx_or = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
            image_Sxx = Image.fromarray(np.uint8(cm.viridis(Sxx_or)*255))
            image_Sxx = image_Sxx.resize((224, 224))
            image_Sxx.save(_dirspect + str(c) + "_spect.png")     
            diCTDt['EDA_spect'] += [image_Sxx] 
 
            diCTDt['timestamp'] += [_ts]
            _ts += 1
            c += 1
            
            rpeaks = np.array(bp.signals.ecg.get_rpks(filt_ECG, SR))

            plt.figure(dpi=300)
            plt.plot(_ECG[i:i+WIND], label="raw ECG")
            plt.plot(filt_ECG[i:i+WIND], label="filt + norm ECG")
            plt.vlines(rpeaks, 0, 1)
            plt.ylabel("A.U. (Normalized)")
            plt.xlabel("Samples")
            plt.legend()
            plt.savefig(_dirECG + str(c) + "_ECG.pdf", bbox_inches='tight', format="pdf")
            plt.close('all')

        assert len(diCTDt['timestamp']) == len(diCTDt['EDA_spect'])
        assert len(diCTDt['timestamp']) == len(diCTDt['EDA_RP'])
        assert len(diCTDt['timestamp']) == len(diCTDt['y_a'])
        assert len(diCTDt['y_a']) == len(diCTDt['y_v'])
        assert len(diCTDt['y_s']) == len(diCTDt['y_v'])
        assert len(diCTDt['y_g']) == len(diCTDt['y_v'])
        assert len(diCTDt['EDA_cvx']) == len(diCTDt['y_v'])
        assert len(diCTDt['EDR']) == len(diCTDt['y_v'])
        assert len(diCTDt['EDL']) == len(diCTDt['y_v'])
        assert len(diCTDt['ECG']) == len(diCTDt['y_v'])
        assert len(diCTDt['ECG']) == len(diCTDt['y_vid'])
        
        print(len(diCTDt['y_a']))
        


    #assert len(np.unique(diCTDt['y_vid'])) == 20
   # assert len(file["joined_data"][0]) == len(np.unique(diCTDt['y_vid']))
    gc.collect()   


assert len(np.unique(diCTDt['y_s'])) == 37
assert len(np.unique(diCTDt['y_vid'])) == 20

for k in diCTDt.keys():
    diCTDt[k] = np.array(diCTDt[k])

_dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"
try:
    os.makedirs(_dir)
except Exception as e:
    pass


with open(_dir + str(WINDSC) + 's_data_v2.pickle', 'wb') as handle:
    pickle.dump(diCTDt, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("over")

