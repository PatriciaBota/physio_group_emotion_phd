from cycler import concat
from matplotlib.axis import GRIDLINE_INTERPOLATION_STEPS
from biosppy.signals.tools import filter_signal
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
from PIL import Image
from matplotlib import cm
import gc
from scipy import signal
import utils
import pandas as pd
from scipy import interpolate
# Only 37 participants took part in the long videos experiment (participants 8, 24 and 28 were not available)

# also dont have data 
GPbyID = np.array([["P1", "P2"], ["P3", "P4"], ["P5", "P6"], ["P7", "P8"], ["P9", "P10"], ["P11", "P12"], ["P13", "P14"], ["P15", "P16"], ["P17", "P18"], ["P19", "P20"], ["P21", "P22"], ["P22", "P23"], ["P23", "P24"], ["P25", "P26"], ["P27", "P28"], ["P29", "P30"], ["P31", "P32"]])

usersTrmv = ["P17", "P20"]
#assert len(usersTKeep) == 20
DPI = 300
DATASET = "KEmoCon"
OVERLAP = 0.75
WINDSC = 20
SR = 128
WIND = WINDSC*SR

diCTDt = {}
diCTDt['EDA_cvx'], diCTDt['EDR'] , diCTDt['EDL'] , diCTDt['BVP'] , diCTDt['y_a'] , diCTDt['y_v'], diCTDt['y_s'],\
        diCTDt['y_g'], diCTDt['EDA_RP'], diCTDt['timestamp'], diCTDt['DFT'], diCTDt['EDA_spect'], diCTDt['y_vid'], diCTDt['tm'] = [], [], [],\
        [], [], [], [], [], \
        [], [], [], [], [], []
diCTDt['DATASET'] = "RECOLA"
diCTDt['WIND'] = WINDSC
diCTDt['SR'] = SR

def filt_sig(sig, sampling_rate, CUT_F):
   # sig, _, _ = bp.tools.filter_signal(signal=sig,  # filter
   #                              ftype='butter',
   #                              band='lowpass',
   #                              order=5,
   #                              frequency=CUT_F,
   #                              sampling_rate=sampling_rate)  

    numerator_coeffs, denominator_coeffs = scipy.signal.butter(4, CUT_F, fs=sampling_rate, btype='low')
    sig = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, sig)
    
    sm_size = int(0.75 * sampling_rate) 
    sig, _ = bp.tools.smoother(signal=sig,
                                   size=sm_size,
                                   mirror=True)
    return sig


_dirfilter = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/spikes/"
try:
    os.makedirs(_dirfilter)
except Exception as e:
    pass
_dircvx = "../Plots/" + DATASET +  "/" + str(WINDSC) + "seg/cvx/"
try:
    os.makedirs(_dircvx)
except Exception as e:
    pass
_dirRP = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg//RP/"
try:
    os.makedirs(_dirRP)
except Exception as e:
    pass
_dirspect = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/spect/"
try:
    os.makedirs(_dirspect)
except Exception as e:
    pass
_dirBVP = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/BVP_filt/"
try:
    os.makedirs(_dirBVP)
except Exception as e:                                              
    pass                                                            

filesDirs = glob.glob("../K-EmoCon_dataset/e4_data/*")

c = 0
for dirIDX, fileDir in enumerate(filesDirs):  # iterate over users files
    _ts = 0
    
    SUBJID = "P" + str(fileDir.split("/")[-1])

    _EDA = pd.read_csv(fileDir + "/E4_EDA.csv").value.values
    _BVP = pd.read_csv(fileDir + "/E4_BVP.csv").value.values
    _ts_BVP = pd.read_csv(fileDir + "/E4_BVP.csv").timestamp.values
    _ts_EDA = pd.read_csv(fileDir + "/E4_EDA.csv").timestamp.values
    serial_EDA = pd.read_csv(fileDir + "/E4_EDA.csv").device_serial.values
    serial_BVP = pd.read_csv(fileDir + "/E4_BVP.csv").device_serial.values
    if SUBJID in ["P29", "P30", "P31", "P32"]:
        if SUBJID in ["P29","P31"]:
            ROWSTOKEEP_EDA = np.where(serial_EDA == "A013E1")[0]
            ROWSTOKEEP_BVP = np.where(serial_BVP == "A013E1")[0]
        else:
            ROWSTOKEEP_EDA = np.where(serial_EDA == "A01A3A")[0]
            ROWSTOKEEP_BVP = np.where(serial_BVP == "A01A3A")[0]
        _EDA = _EDA[ROWSTOKEEP_EDA]
        _ts_EDA = _ts_EDA[ROWSTOKEEP_EDA]
        _BVP = _BVP[ROWSTOKEEP_BVP]
        _ts_BVP = _ts_BVP[ROWSTOKEEP_BVP]
    if SUBJID in usersTrmv:
        plt.figure()
        plt.plot(_ts_EDA, _EDA, ".", label="raw EDA")
        plt.legend()
        plt.savefig(_dirfilter + "raw_EDA_removed_" + SUBJID + ".pdf", format="pdf")
        continue
    #plt.figure()
    #f, ps = signal.welch(_EDA, 64, nperseg=104)
    #plt.plot(f, np.log(ps))
    #plt.savefig("spect_raw_EDA.pdf", format="pdf")
    

        #plt.figure()
    #plt.plot(_ts_BVP[6*64:30*64], minmax_scale(_BVP[6*64:30*64])+0.5, label="norm raw BVP")
   # freq = 5
   # sin = np.sin(2*np.pi*freq*xnew)
   # plt.plot(xnew, sin, label="raw sinusoide 5z freq")
   # fltsin = filt_sig(sin, SR, 0.1)
    #plt.plot(xnew, fltsin, label="filtered sinusoide cut off at 0.1hz")
    #plt.plot(filtered_signal, label="filtered")
   # plt.ylabel("Amplitude (a.u)")
   # plt.xlabel("Time (s)")
   # plt.legend()
   # plt.show()
    
    ## debate start and end times
    ts_deb_matrix = pd.read_csv("../K-EmoCon_dataset/metadata/subjects.csv")
    idx_db_m_sjb = np.where(int(SUBJID[1:]) == ts_deb_matrix.pid)[0][0]
    db_st = ts_deb_matrix.values[idx_db_m_sjb][2]
    db_end = ts_deb_matrix.values[idx_db_m_sjb][3]

    ann = pd.read_csv("../K-EmoCon_dataset/emotion_annotations/aggregated_external_annotations/" + SUBJID + ".external.csv")
    y_a = ann.arousal.values
    y_v = ann.valence.values
    ann_ts = ann.seconds.values  # assert with _ts_EDA[-1]

    print("raw ann ts", ann_ts[0], ann_ts[-1])
    print("raw data ts", _ts_EDA[0], _ts_EDA[-1])
    print("debate ts", db_st, db_end)
    print("###############") 

    # cut data to debate time 
    st_t_i = np.where(_ts_EDA >= db_st)[0][0]
    try:
        end_t_i = np.where(_ts_EDA > db_end)[0][0]
    except Exception as e:
        print(e)
        end_t_i = len(_ts_EDA)
    
    _EDA = _EDA[st_t_i:end_t_i]
    _ts_EDA = _ts_EDA[st_t_i:end_t_i]
    
    print("ct db ann ts", ann_ts[0], ann_ts[-1])
    print("ct db data ts", _ts_EDA[0], _ts_EDA[-1])
    print("ct db debate ts", db_st, db_end)
    
    st_t_i = np.where(_ts_BVP >= db_st)[0][0]
    try:
        end_t_i = np.where(_ts_BVP > db_end)[0][0]
    except Exception as e:
        print(e)
        end_t_i = len(_ts_BVP)
    _BVP = _BVP[st_t_i:end_t_i]
    _ts_BVP = _ts_BVP[st_t_i:end_t_i]

    _ts_EDA -= _ts_EDA[0]
    _ts_BVP -= _ts_BVP[0]
    _ts_BVP *= 0.001  # convert to seconds
    _ts_EDA *= 0.001

    print("s ct db ann ts", ann_ts[0], ann_ts[-1])
    print("s ct db data ts", _ts_EDA[0], _ts_EDA[-1])
    print("s ct db debate ts", db_st * 0.001, (db_end - db_st)*0.001)
    print("obtained SR EDA", len(_ts_EDA)/(_ts_EDA[-1] - _ts_EDA[0]))
    print("obtained SR BVP", len(_ts_BVP)/(_ts_BVP[-1] - _ts_BVP[0]))
    # filter EDA
    plt.figure()
    plt.plot(_ts_EDA, _EDA, label="raw EDA")
    _EDA = filt_sig(_EDA, 4, CUT_F=0.1)  # filter signal↲
    f = interpolate.interp1d(_ts_EDA, _EDA)
    xnew = np.arange(_ts_EDA[0], _ts_EDA[-1], 1/SR)
    _EDA = f(xnew)
    plt.plot(xnew, _EDA, label="EDA filtered + resampled")
    plt.legend()
    plt.title(SUBJID)
    plt.savefig(_dirfilter + str(c) + "_" + SUBJID + "filter_EDA.pdf", format="pdf")
    plt.close('all')


    #f, sp = bp.signals.tools.welch_spectrum(signal=_EDA, sampling_rate=4, size=150, overlap=None, window='hanning', window_kwargs=None, pad=None, decibel=True)
    annWIND = 5*SR
    _EDA_cvx = [_EDA[i:i+annWIND] for i in range(0, len(_EDA) - annWIND, annWIND)]
    #_ts_seg_EDA = [_ts_EDA[i:i+annWIND] for i in range(0, len(_ts_EDA) - annWIND, annWIND)]
    #assert len(ann_ts) == len(_EDA_cvx)

    N = 0
    _y_a, _y_v, _t_ann = [], [], []
    for w in range(len(_EDA_cvx)):
        NW = len(_EDA_cvx[w])
        _y_a += [y_a[w]] * NW
        _y_v += [y_v[w]] * NW
        _t_ann += [ann_ts[w]] * NW
    assert len(_y_a) == len(np.hstack((_EDA_cvx)))
    s_N = list(range(0, len(_EDA) - annWIND, annWIND))[-1] + annWIND
    assert len(np.hstack((_EDA_cvx))) == s_N
    print("subs" + SUBJID)

    #print("af seg", _ts_seg_EDA[-1][-1], _t_ann[-1])

    inputS = _EDA.astype(np.float64).copy()
    inputS = (inputS - inputS.mean())/inputS.std()

    [cvxEDR, _, cvxEDL, _, _, _, _] = cvxEDA.cvxEDA(inputS, 1./SR)  # cvxEDA only accepts double types, needs to be norm↲
    cvxEDR = (cvxEDR - cvxEDR.mean())/cvxEDR.std()
    cvxEDL = (cvxEDL - cvxEDL.mean())/cvxEDL.std()
    
    s_N = min(len(_y_a), len(inputS)) - WIND

    WIND = WINDSC*SR 
    diCTDt['EDA_cvx'] += [inputS[i:i+WIND] for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    diCTDt['EDR'] += [cvxEDR[i:i+WIND] for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    diCTDt['EDL'] += [cvxEDL[i:i+WIND] for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    diCTDt['y_a'] += [np.mean(_y_a[i:i+WIND]) for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    diCTDt['y_v'] += [np.mean(_y_v[i:i+WIND]) for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    diCTDt['tm'] += [np.mean(_t_ann[i:i+WIND]) for i in range(0, s_N, WIND - int(OVERLAP*WIND))]
    
    print("stored ann tm", diCTDt['tm'][-1])
    
    sN = len([np.mean(_y_v[i:i+WIND]) for i in range(0, s_N, WIND - int(OVERLAP*WIND))])
    diCTDt['y_s'] += [SUBJID]*sN
    diCTDt['y_vid'] += [0]*sN
    
    gID = np.where(GPbyID == SUBJID)[0][0]
    usInG = GPbyID[gID]
    SUBJID_i = np.where(usInG == SUBJID)[0]
    otherM = np.delete(usInG, SUBJID_i)
    
    if otherM in usersTrmv:
        gID = -1
    diCTDt['y_g'] += [gID]*sN
    
    # filter  BVP signal
    #BVP_SR = 64


   # sm_size = int(0.06 * SR)
   # filtBVP, _ = bp.signals.tools.smoother(_BVP,
   #                       kernel='boxzen',
   #                       size=sm_size,
   #                       mirror=False)

        
    f = interpolate.interp1d(_ts_BVP, _BVP)
    xnew = np.arange(_ts_BVP[0], _ts_BVP[-1], 1/SR)
    filt_BVP = f(xnew)

    filt_BVP, _, _ = bp.tools.filter_signal(signal=filt_BVP,
                                      ftype='butter',
                                      band='bandpass',
                                      order=4,
                                      frequency=[1, 8],
                                      sampling_rate=SR)


    
    diCTDt['BVP'] += [filt_BVP[i:i+WIND] for i in range(0, s_N, WIND - int(OVERLAP*WIND))]

    for i in range(0, s_N, WIND - int(OVERLAP*WIND)):
        assert len(filt_BVP[i:i+WIND]) == WIND
        onsets, _ = bp.signals.ppg.find_onsets_elgendi2013(signal=filt_BVP[i:i+WIND], sampling_rate=SR)
        
        plt.figure()
        plt.plot(inputS[i:i+WIND], label="norm EDA")
        plt.plot(cvxEDR[i:i+WIND], label="norm EDR")
        plt.plot(cvxEDL[i:i+WIND], label="norm EDL")
        plt.plot(minmax_scale(filt_BVP[i:i+WIND]), label="NORM BVP")
        plt.legend()
        plt.savefig(_dircvx + str(c) + "_" + SUBJID +  "_cvx.pdf", format="pdf")
        plt.close('all')

        # # RP
        rec = utils.rec_plot(inputS[i:i+WIND])
        rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
        image_rec = Image.fromarray(np.uint8(cm.viridis.reversed()(rec)*255))
        image_rec = image_rec.resize((224, 224))
        image_rec.save(_dirRP + str(c) + "_RP.png")
        diCTDt['EDA_RP'] += [image_rec]

        # # spect
        t, f, Sxx = signal.spectrogram(inputS[i:i+WIND], fs=SR,
                     nperseg=100, noverlap=60, return_onesided=True)
        FCUT = 15
        Sxx = np.log(Sxx[:FCUT])

        Sxx_or = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
        image_Sxx = Image.fromarray(np.uint8(cm.viridis(Sxx_or)*255))
        image_Sxx = image_Sxx.resize((224, 224))
        image_Sxx.save(_dirspect + str(c) + "_spect.png")
        diCTDt['EDA_spect'] += [image_Sxx]

        #spectrum = np.log(np.abs(np.fft.fft(inputS[i:i+WIND])))
        #spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
        #f = np.nan_to_num(np.linspace(0, SR/2, len(spectrum)))
        #t = np.arange(0, (len(inputS[i:i+WIND])-1)/SR, 1/SR)
        #f, PXX = scipy.signal.welch(inputS[i:i+WIND], fs=SR, nperseg=100, noverlap=80)

        #Sxx = np.log(PXX)
        #PXX = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
        #PXX = np.nan_to_num(PXX[:FCUT])
        #f = f[:FCUT]

        #diCTDt['DFT'] += [PXX]
        #plt.figure()
        #plt.plot(f, PXX, label="spectrum")
        #plt.savefig(_dirspect + str(c) + "_spectrum.png", format="png")
        
        plt.figure()
        plt.plot(_BVP[i:i+WIND], label="raw BVP")
        plt.plot(filt_BVP[i:i+WIND], label="filt + res BVP")
        plt.vlines(onsets, filt_BVP[i:i+WIND].min(), filt_BVP[i:i+WIND].max(), color="r")
        plt.legend()
        plt.savefig(_dirBVP + str(c) + "_BVP.png", format="png")
        plt.close('all')
        
        diCTDt['timestamp'] += [_ts]
        _ts += 1
        c += 1

    assert len(diCTDt['timestamp']) == len(diCTDt['EDA_spect'])
    assert len(diCTDt['timestamp']) == len(diCTDt['EDA_RP'])
    assert len(diCTDt['timestamp']) == len(diCTDt['y_a'])
    assert len(diCTDt['y_a']) == len(diCTDt['y_v'])
    assert len(diCTDt['y_s']) == len(diCTDt['y_v'])
    assert len(diCTDt['y_g']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDA_cvx']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDR']) == len(diCTDt['y_v'])
    assert len(diCTDt['EDL']) == len(diCTDt['y_v'])
    assert len(diCTDt['BVP']) == len(diCTDt['y_v'])
     

for k in diCTDt.keys():
    diCTDt[k] = np.array(diCTDt[k])

_dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/" 
try:
    os.makedirs(_dir)
except Exception as e:
    pass


with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(diCTDt, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("over")

