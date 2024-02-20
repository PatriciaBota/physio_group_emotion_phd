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
import pyhrv
import gc
import psutil

# open pickle with data

print("Before st")
print('RAM memory % used:', psutil.virtual_memory()[2])
# Getting loadover15 minutes
load1, load5, load15 = psutil.getloadavg() 
cpu_usage = (load15/os.cpu_count()) * 100
print("The CPU usage is : ", cpu_usage)


WINDSC = 20
DATASET = "RECOLA" # AMIGOS, RECOLA

if DATASET == "AMIGOS":
    NAME = "normPVid_woutSpks"
    VID = "LVSVO" 
    _dir = "../Input/" + DATASET + "/" + VID + "/" + str(WINDSC) + "seg/" + NAME + "/"

else:
    _dir = "../Input/" + DATASET + "/" + str(WINDSC) + "seg/"



with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

SR = data['SR']  # sampling rate


def mob(sig):
    der = np.diff(sig)
    return np.sqrt(sig.var()*der.var())


def getEDAfts(inputECG, SR, c):

    fv, fv_header = [], []
    #fv += [len(edaChrs["pks"])]  # peak count
    #fv_header += ["len_pks"]
    
    try:
        if DATASET == "KEmoCon":
            rpeaks, _ = bp.signals.ppg.find_onsets_elgendi2013(signal=inputECG, sampling_rate=SR)
        else:
            rpeaks = np.array(bp.signals.ecg.get_rpks(inputECG, SR))
        length = len(inputECG)
        T = (length - 1) / SR
        ts = np.linspace(0, T, length, endpoint=True)

    except Exception as e:
        print(e)
        rpeaks = []
        ts = []

    #out = pyhrv.hrv(rpeaks=ts[rpeaks], sampling_rate=SR, plot_tachogram=False, plot_ecg=False, show=False)
    try:
        out = pyhrv.time_domain.nni_parameters(rpeaks=ts[rpeaks])
    except Exception as e:
        print(e)
        out = {}
        out["nni_counter"], out["nni_mean"], out["nni_min"], out["nni_max"] = 0, 0, 0, 0

    fv_header += ["nni_counter"]
    fv += [out["nni_counter"]]

    #fv_header += ["nni_mean"]
    #fv += [out["nni_mean"]]

    #fv_header += ["nni_min"]
    #fv += [out["nni_min"]]

    #fv_header += ["nni_max"]
    #fv += [out["nni_max"]]

    try:
        out = pyhrv.time_domain.hr_parameters(rpeaks=ts[rpeaks])
    except Exception as e:
        print(e)
        out = {}
        out["hr_mean"], out["hr_min"], out["hr_max"], out["hr_std"] = 0, 0, 0, 0

    fv_header += ["hr_mean"]
    fv += [out["hr_mean"]]

    fv_header += ["hr_min"]
    fv += [out["hr_min"]]

    fv_header += ["hr_max"]
    fv += [out["hr_max"]]

    fv_header += ["hr_std"]
    fv += [out["hr_std"]]

    try:
        out = pyhrv.time_domain.nni_differences_parameters(rpeaks=ts[rpeaks])    
    except Exception as e:
        print(e)
        out = {}
        out["nni_diff_mean"], out["nni_diff_min"], out["nni_diff_max"] = 0, 0, 0


    fv_header += ["nni_diff_mean"]
    fv += [out["nni_diff_mean"]]

    fv_header += ["nni_diff_min"]
    fv += [out["nni_diff_min"]]

    fv_header += ["nni_diff_max"]
    fv += [out["nni_diff_max"]]

    try:
        out = pyhrv.time_domain.sdnn(rpeaks=ts[rpeaks])    
    except Exception as e:
        print(e)
        out = {}
        out["sdnn"] = 0

    fv_header += ["sdnn"]
    fv += [out["sdnn"]]

    try:
        out = pyhrv.time_domain.sdnn_index(rpeaks=ts[rpeaks])    
    except Exception as e:
        print(e)
        out = {}
        out["sdnn_index"] = 0
    
    fv_header += ["sdnn_index"]
    fv += [out["sdnn_index"]]

    try:
        out = pyhrv.time_domain.sdann(rpeaks=ts[rpeaks])        
    except Exception as e:
        print(e)
        out = {}
        out["sdann"] = 0

    fv_header += ["sdann"]
    fv += [out["sdann"]]

    try:
        out = pyhrv.time_domain.rmssd(rpeaks=ts[rpeaks])        
    except Exception as e:
        print(e)
        out = {}
        out["rmssd"] = 0

    fv_header += ["rmssd"]
    fv += [out["rmssd"]]
    
    try:
        out = pyhrv.time_domain.sdsd(rpeaks=ts[rpeaks])        
    except Exception as e:
        print(e)
        out = {}
        out["sdsd"] = 0

    fv_header += ["sdsd"]
    fv += [out["sdsd"]]

    try:
        out = pyhrv.time_domain.nnXX(rpeaks=ts[rpeaks], threshold=50)            
    except Exception as e:
        print(e)
        out = {}
        out["nn50"], out["pnn50"] = 0, 0

    fv_header += ["nn50"]
    fv += [out["nn50"]]

    fv_header += ["pnn50"]
    fv += [out["pnn50"]]

    try:
        out = pyhrv.time_domain.nnXX(rpeaks=ts[rpeaks], threshold=20)                
    except Exception as e:
        print(e)
        out = {}
        out["nn20"], out["pnn20"] = 0, 0

    fv_header += ["nn20"]
    fv += [out["nn20"]]

    fv_header += ["pnn20"]
    fv += [out["pnn20"]]

    try:
        out = pyhrv.time_domain.tinn(rpeaks=ts[rpeaks], plot=False, show=False)                
    except Exception as e:
        print(e)
        out = {}
        out["tinn_n"], out["tinn_m"], out["tinn"] = 0, 0, 0

    fv_header += ["tinn_n"]
    fv += [out["tinn_n"]]

    fv_header += ["tinn_m"]
    fv += [out["tinn_m"]]

    fv_header += ["tinn"]
    fv += [out["tinn"]]
    
    try:
        out = pyhrv.time_domain.triangular_index(rpeaks=ts[rpeaks], plot=False, show=False)   
    except Exception as e:
        print(e)
        out = {}
        out["tri_index"] = 0

    fv_header += ["tri_index"]
    fv += [out["tri_index"]]

    try:
        out = pyhrv.frequency_domain.welch_psd(rpeaks=ts[rpeaks], show_param=False, show=False)                
    except Exception as e:
        print(e)
        out = {}
        out["fft_peak"], out["fft_abs"], out["fft_rel"], out["fft_log"], out["fft_norm"], \
                out["fft_ratio"], out["fft_total"] = [0]*3, [0]*3, [0]*3, [0]*3, [0]*2, 0, 0

    fv_header += ["fft_peak_VLF"]
    fv += [out["fft_peak"][0]]
    fv_header += ["fft_peak_LF"]
    fv += [out["fft_peak"][1]]
    fv_header += ["fft_peak_HF"]
    fv += [out["fft_peak"][2]]

    fv_header += ["fft_abs_VLF"]
    fv += [out["fft_abs"][0]]
    fv_header += ["fft_abs_LF"]
    fv += [out["fft_abs"][1]]
    fv_header += ["fft_abs_HF"]
    fv += [out["fft_abs"][2]]

    fv_header += ["fft_rel_VLF"]
    fv += [out["fft_rel"][0]]
    fv_header += ["fft_rel_LF"]
    fv += [out["fft_rel"][1]]
    fv_header += ["fft_rel_HF"]
    fv += [out["fft_rel"][2]]

    fv_header += ["fft_log_VLF"]
    fv += [out["fft_log"][0]]
    fv_header += ["fft_log_LF"]
    fv += [out["fft_log"][1]]
    fv_header += ["fft_log_HF"]
    fv += [out["fft_log"][2]]

    fv_header += ["fft_norm_LF"]
    fv += [out["fft_norm"][0]]
    fv_header += ["fft_norm_HF"]
    fv += [out["fft_norm"][1]]

    fv_header += ["fft_ratio"]
    fv += [out["fft_ratio"]]

    fv_header += ["fft_total"]
    fv += [out["fft_total"]]

    try:
        out = pyhrv.frequency_domain.lomb_psd(rpeaks=ts[rpeaks], show_param=False, show=False)                
    except Exception as e:
        print(e)
        out = {}
        out["lomb_peak"], out["lomb_abs"], out["lomb_rel"], out["lomb_log"], out["lomb_norm"], \
                out["lomb_ratio"], out["lomb_total"] = [0]*3, [0]*3, [0]*3, [0]*3, [0]*2, 0, 0

    fv_header += ["lomb_peak_VLF"]
    fv += [out["lomb_peak"][0]]
    fv_header += ["lomb_peak_LF"]
    fv += [out["lomb_peak"][1]]
    fv_header += ["lomb_peak_HF"]
    fv += [out["lomb_peak"][2]]

    fv_header += ["lomb_abs_VLF"]
    fv += [out["lomb_abs"][0]]
    fv_header += ["lomb_abs_LF"]
    fv += [out["lomb_abs"][1]]
    fv_header += ["lomb_abs_HF"]
    fv += [out["lomb_abs"][2]]

    fv_header += ["lomb_rel_VLF"]
    fv += [out["lomb_rel"][0]]
    fv_header += ["lomb_rel_LF"]
    fv += [out["lomb_rel"][1]]
    fv_header += ["lomb_rel_HF"]
    fv += [out["lomb_rel"][2]]

    fv_header += ["lomb_log_VLF"]
    fv += [out["lomb_log"][0]]
    fv_header += ["lomb_log_LF"]
    fv += [out["lomb_log"][1]]
    fv_header += ["lomb_log_HF"]
    fv += [out["lomb_log"][2]]

    fv_header += ["lomb_norm_LF"]
    fv += [out["lomb_norm"][0]]
    fv_header += ["lomb_norm_HF"]
    fv += [out["lomb_norm"][1]]

    fv_header += ["lomb_ratio"]
    fv += [out["lomb_ratio"]]

    fv_header += ["lomb_total"]
    fv += [out["lomb_total"]]

    try:
        out = pyhrv.frequency_domain.ar_psd(rpeaks=ts[rpeaks], show_param=False, show=False) 
    except Exception as e:
        print(e)
        out = {}
        out["ar_peak"], out["ar_abs"], out["ar_rel"], out["ar_log"], out["ar_norm"], out["ar_ratio"], \
                out["ar_total"] = [0]*3, [0]*3, [0]*3, [0]*3, [0]*2, 0, 0

    fv_header += ["ar_peak_VLF"]
    fv += [out["ar_peak"][0]]
    fv_header += ["ar_peak_LF"]
    fv += [out["ar_peak"][1]]
    fv_header += ["ar_peak_HF"]
    fv += [out["ar_peak"][2]]

    fv_header += ["ar_abs_VLF"]
    fv += [out["ar_abs"][0]]
    fv_header += ["ar_abs_LF"]
    fv += [out["ar_abs"][1]]
    fv_header += ["ar_abs_HF"]
    fv += [out["ar_abs"][2]]

    fv_header += ["ar_rel_VLF"]
    fv += [out["ar_rel"][0]]
    fv_header += ["ar_rel_LF"]
    fv += [out["ar_rel"][1]]
    fv_header += ["ar_rel_HF"]
    fv += [out["ar_rel"][2]]

    fv_header += ["ar_log_VLF"]
    fv += [out["ar_log"][0]]
    fv_header += ["ar_log_LF"]
    fv += [out["ar_log"][1]]
    fv_header += ["ar_log_HF"]
    fv += [out["ar_log"][2]]

    fv_header += ["ar_norm_LF"]
    fv += [out["ar_norm"][0]]
    fv_header += ["ar_norm_HF"]
    fv += [out["ar_norm"][1]]

    fv_header += ["ar_ratio"]
    fv += [out["ar_ratio"]]

    fv_header += ["ar_total"]
    fv += [out["ar_total"]]

    try:
        out = pyhrv.nonlinear.poincare(rpeaks=ts[rpeaks], show=False)                
    except Exception as e:
        print(e)
        out = {}
        out["sd1"],  out["sd2"], out["sd_ratio"], out["ellipse_area"] = 0, 0, 0, 0

    fv_header += ["sd1"]
    fv += [out["sd1"]]

    fv_header += ["sd2"]
    fv += [out["sd2"]]

    fv_header += ["sd_ratio"]
    fv += [out["sd_ratio"]]

    fv_header += ["ellipse_area"]
    fv += [out["ellipse_area"]]

    try:
        out = pyhrv.nonlinear.sample_entropy(rpeaks=ts[rpeaks])                
    except Exception as e:
        print(e)
        out = {}
        out["sampen"] = 0

    fv_header += ["sampen"]
    fv += [out["sampen"]]

    try:
        out = pyhrv.nonlinear.dfa(rpeaks=ts[rpeaks], show=False)                
    except Exception as e:
        print(e)
        out = {}
        out["dfa_alpha1"], out["dfa_alpha2"] = 0, 0

    fv_header += ["dfa_alpha1"]
    fv += [out["dfa_alpha1"]]

    fv_header += ["dfa_alpha2"]
    fv += [out["dfa_alpha2"]]
    
    fv = np.array(fv).astype(np.float32)
    try:
        fv = np.nan_to_num(fv)   #
    except Exception as e:
        print(e)

    plt.close('all')
    gc.collect()
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg() 
    cpu_usage = (load15/os.cpu_count()) * 100
    print("The CPU usage is : ", cpu_usage)


    return fv, np.array(fv_header)


data['RR_stfv'] = []
# iterate over data and extract features
for i_seg, d_seg in enumerate(data['y_a']):
    #print("seg idx", i_seg)
    if DATASET == "KEmoCon":
        inputECG = data['BVP'][i_seg]
    else:
        inputECG = data['ECG'][i_seg]

    fv, data['RR_fv_h'] = getEDAfts(inputECG, SR, i_seg)
    
    if not len(data['RR_stfv']):
        data['RR_stfv'] = fv
    else:
        data['RR_stfv'] = np.vstack((data['RR_stfv'], fv))
    print("RR_stfv shape", data['RR_stfv'].shape)


print("End FE")       
with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open(_dir + str(WINDSC) + 's_data.pickle', 'rb') as handle:
#    data = pickle.load(handle)
print("after removal nni size", data['RR_stfv'].shape, data['RR_fv_h'].shape) 

#delIDX = []
#to_dl = ["nni_min", "nni_max", "nni_mean"]
#for f in to_dl:
#    delIDX +=  np.where(data['RR_fv_h'] == f)[0].tolist()
#print("to delete", delIDX)
#data['RR_stfv'] = np.delete(data['RR_stfv'], delIDX, axis=1)
#data['RR_fv_h'] = np.delete(data['RR_fv_h'], delIDX)
print("Start FS")       
print("after removal nni size", data['RR_stfv'].shape, data['RR_fv_h'].shape) 

redstfv, data['RR_red_fv_h'] = utils.remove_correlatedFeatures(data['RR_stfv'], data['RR_fv_h'], threshold=0.85)
print("After FS size", redstfv.shape[1]) 

assert len(data['y_a']) == len(data['RR_stfv'])

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
            n_fv += minmax_scale(redstfv[sIDX][vidIDX]).tolist()
    else:        
        n_y_a += data['y_a'][sIDX].tolist()
        n_fv += minmax_scale(redstfv[sIDX]).tolist()

n_y_a = np.array(n_y_a)
data['RR_redstfv'] = np.array(n_fv)

assert (n_y_a == data['y_a']).all()
assert len(data['y_a']) == len(data['RR_redstfv'])
with open(_dir + str(WINDSC) + 's_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("End")     

print("final fts: ", data['RR_red_fv_h'])

