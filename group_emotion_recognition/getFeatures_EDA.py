import numpy as np
import matplotlib.pyplot as plt
import biosppy as bp
import pickle
import pandas as pd
import pandas_profiling
import utils
import tsfel
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
import scipy
import os
import psutil
import gc
# open pickle with data


WINDSC = 20
DATASET = "RECOLA"

if DATASET == "AMIGOS":
    NAME = "normPVid_woutSpks"
    VID = "LVSVO"
    
    _dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"
    _dirpks = "../Plots/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/pks/"
else:
    _dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"
    _dirpks = "../Plots/" + DATASET + "/" + str(WINDSC) + "seg/pks/"

try:
    os.makedirs(_dirpks)
except Exception as e:
    pass

try:
    os.makedirs(_dir)
except Exception as e:
    pass

with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

SR = data['SR']  # sampling rate

def mob(sig):
    der = np.diff(sig)
    return np.sqrt(sig.var()*der.var())


def getEDAfts(inputEDA, inputEDR, SR, c):
    # EDA characteristic
    sm_size = int(0.4 * SR)
    downSignal, _ = bp.tools.smoother(signal=inputEDA,
                                  kernel='boxzen',
                                  size=sm_size,
                                  mirror=True)


    edaChrs = bp.features.eda_features.eda_features(signal=minmax_scale(downSignal), TH=0.04, sampling_rate=SR)  
    onsets, pks, amps, end  = bp.signals.eda.get_eda_param(downSignal, min_amplitude=0.04, sampling_rate=SR)
    half_rec, six_rec, hf_ts, six_ts = bp.signals.eda.Mar_edr_times(inputEDA, onsets, pks)
    
    plt.figure() # check onsets and pks are well obtained
    plt.plot(minmax_scale(downSignal))
    plt.vlines(edaChrs["onsets"], 0, 1, label="onset", color="g")
    plt.vlines(edaChrs["pks"], 0, 1, label="pks", color="r")
    plt.vlines(pks, 0, 1, label="pks2", color="orange")
    plt.legend()
    plt.savefig(_dirpks + str(c) + "_onpk.pdf", format="pdf")
    plt.close('all')

    fv, fv_header = [], []
    fv += [len(edaChrs["pks"])]  # peak count
    fv_header += ["len_pks"]
    fv += [np.mean(edaChrs["amps"])]  # mean peak amplitude
    fv_header += ["pks_amp"]
    
    fv += [np.mean(edaChrs["rise_ts"])]  # mean rise time 
    fv_header += ["rise_ts"]
    
    fv += [np.sum(edaChrs["amps"])]  # sum pks amplitude
    fv_header += ["sum_pks_amp"]
    
    fv += [np.sum(edaChrs["rise_ts"])]  # sum rise time
    fv_header += ["sum_rise_ts"]
    
    area = 0
    for pkIdx in range(len(pks)):
        st = onsets[pkIdx]
        try:
            _end = hf_ts[pkIdx] + pks[pkIdx]
        except:
            _end = end[pkIdx] 
        area += np.trapz(inputEDA[st:_end])
    fv += [area]  # sum areas
    fv_header += ["sum_areas"]
    
    #  area under curve within a time response window of 5 sec after each stimulus onset
    area = 0
    for pkIdx in range(len(pks)):
        st = onsets[pkIdx]
        _end = st + 5*SR
        try:
            area += np.trapz(inputEDA[st:_end])
        except:
            area += 0

    fv += [area]
    fv_header += ["sum_area5s"]
    
    fv += [np.mean(inputEDA)]  # mean EDA
    fv_header += ["mean_EDA"]
    
    fv += [np.std(inputEDA)]  # std EDA
    fv_header += ["std_EDA"]
    
    fv += [scipy.stats.kurtosis(inputEDA)]  # kurtosis EDA
    fv_header += ["kurtosis_EDA"]
    
    fv += [scipy.stats.skew(inputEDA)]  # skewness EDA
    fv_header += ["skew_EDA"]
    
    der = np.diff(inputEDA)
    fv += [np.mean(der)]  # mean derivative EDA
    fv_header += ["mean_1sder"]
    
    negIDX = np.where(der < 0)[0]
    try:
        fv += [np.mean(der[negIDX])]  # mean negative derivative EDA
    except:
        fv += [0]
    fv_header += ["mean_neg_1sder"]
    
    # signal energy
    #spectrum_signal = np.abs(np.fft.fft(inputEDA, SR))**2
    #spectrum = np.nan_to_num(spectrum_signal[:len(spectrum_signal)//2])
    spectrum = np.log(np.abs(np.fft.fft(inputEDA)))
    spectrum = np.nan_to_num(spectrum[:len(spectrum)//2])
    f = np.nan_to_num(np.linspace(0, SR/2, len(spectrum)))

     # or  total area under the rising time curve is treated as the GSR response energy
    area = 0
    for pkIdx in range(len(pks)):
        st = onsets[pkIdx]
        _end = pks[pkIdx]
        try:
            area += np.trapz(inputEDR[st:_end])
        except:
            area += 0
    fv += [area]
    fv_header += ["GSR_respEnerg"]
    
    # 5 spectral power in the [0-0.5]Hz ban
    fv += [np.sum(spectrum[:10])]  # TODO
    fv_header += ["sum_spec"]

    energy = np.nan_to_num(bp.tools.signal_energy(spectrum, f)[:][0])
    try:
        spect_var = np.convolve(energy)
        spect_var /= np.max(np.abs(spect_var))
    except:
        spect_var = 0
    fv += [spect_var]  # variance spectral power
    fv_header += ["spectPower_var"]
    
    #signal magnitude area
    fv += [np.sum(inputEDR**2)]
    fv_header += ["EDR_area"]
   
    sum_spectrum = np.sum(spectrum)
    norm_spectrum = np.nan_to_num(spectrum / sum_spectrum)

    try:
        spectral_centroid = np.nan_to_num(np.dot(f, norm_spectrum))
    except:
        spectral_centroid = 0

    # spectrum kurtosis
    try:
        spectral_spread = np.nan_to_num(np.dot(((f - spectral_centroid) ** 2), norm_spectrum))
        
        spectral_kurtosis = np.nan_to_num(np.sum(((f - spectral_centroid) ** 4) * norm_spectrum) / (spectral_spread**2))
    except:
        spectral_kurtosis = 0
    fv += [spectral_kurtosis]
    fv_header += ["spect_kurt"]
    
    fv += [mob(inputEDA)] # mobility  # CHECK
    fv_header += ["mobility"]
    
    fv += [mob(inputEDA)*mob(der)] # complexity CHECK
    fv_header += ["complexity"]

    fv += [len(np.where(np.diff(np.sign(inputEDA)))[0])]  # zero cross - wasnt in list
    fv_header += ["zeroCross"]

    _mfcc = tsfel.feature_extraction.features.mfcc(inputEDA, SR)

    fv += [scipy.stats.kurtosis(_mfcc)]
    fv_header += ["mfcc_kurt"]
    
    fv += [scipy.stats.skew(_mfcc)]
    fv_header += ["mfcc_skew"]
    
    fv += [np.mean(_mfcc)]
    fv_header += ["mfcc_mean"]
    
    fv += [np.std(_mfcc)]
    fv_header += ["mfcc_std"]
    
    fv += [np.median(_mfcc)]
    fv_header += ["mfcc_median"]

    fv = np.array(fv)
    fv = np.nan_to_num(fv.astype(np.float))   #

    gc.collect()
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg() 
    cpu_usage = (load15/os.cpu_count()) * 100
    print("The CPU usage is : ", cpu_usage)
    
    return fv, np.array(fv_header)


# iterate over data and extract features
data['EDA_stfv'] = []
for i_seg, d_seg in enumerate(data['EDA_cvx']):
    #print("seg idx", i_seg)
    inputEDA = data['EDA_cvx'][i_seg]
    inputEDR = data['EDR'][i_seg]

    fv, data['EDA_fv_h'] = getEDAfts(inputEDA, inputEDR, SR, i_seg)

    if not len(data['EDA_stfv']):
        data['EDA_stfv'] = fv
    else:
        data['EDA_stfv'] = np.vstack((data['EDA_stfv'], fv))
    print("stfv shape", data['EDA_stfv'].shape)

print("End FE")       
with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
#    data = pickle.load(handle)
print("Start FS")       
print("initial size", data['EDA_stfv'].shape[1]) 

redstfv, data['EDA_red_fv_h'] = utils.remove_correlatedFeatures(data['EDA_stfv'], data['EDA_fv_h'], threshold=0.85)
print("After FS size", redstfv.shape[1])# 

## norm by video
n_y_a, n_fv = [], []
_, idx = np.unique(data['y_s'], return_index=True)
for sID in data['y_s'][np.sort(idx)]:
    sIDX = np.where(sID == data['y_s'])[0]

    if DATASET == "AMIGOS":
        _, idx = np.unique(data['y_vid'][sIDX], return_index=True)    
        for vidID in data['y_vid'][sIDX][np.sort(idx)]:
            vidIDX = np.where(vidID == data['y_vid'][sIDX])[0]
            
            n_y_a += data['y_a'][sIDX][vidIDX].tolist()
            n_fv += minmax_scale(redstfv[sIDX][vidIDX].astype(np.float32)).tolist()
    else:        
        n_y_a += data['y_a'][sIDX].tolist()
        n_fv += minmax_scale(np.nan_to_num(redstfv[sIDX].astype(np.float32))).tolist()

n_y_a = np.array(n_y_a)
data['EDA_redstfv'] = np.array(n_fv).copy()

assert (n_y_a == data['y_a']).all()
assert len(data['y_a']) == len(data['EDA_redstfv'])

with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("End")       

