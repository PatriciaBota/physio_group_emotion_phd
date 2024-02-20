# -*- coding: utf-8 -*-

"""
This file analyses the quality of the EDA data
Returns: 
    Quality information in csv table
    Low quality samples idx
"""


# Imports
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import pandas as pd
import scipy
from scipy.signal import welch
import pdb 

# local
import biosppy as bp
import get_loss


FLAG = "segments"  # "session", "segments"


if FLAG == "session":
    DATA_DIR = '../3_Physio/Transformed/physio_trans_data_session.pickle'
    QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_session.pickle'
    STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_session.pickle'
else:
    DATA_DIR = '../3_Physio/Transformed/physio_trans_data_segments.pickle'
    QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_segments.pickle'
    STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_segments.pickle'


files = glob.glob("../6_Results/PPG/" + FLAG + "/Plots/*")
for f in files:
    os.remove(f)

with open(DATA_DIR, 'rb') as handle:  # read data
    data = pickle.load(handle)
with open(STIMU_DATA_DIR, 'rb') as handle:  # read data
    stimu_data = pickle.load(handle)
with open(QUEST_DATA_DIR, 'rb') as handle:  # read data
    quest_data = pickle.load(handle)


def full_scale(dt, sampling_rate, FLAG="segments"):
    pSat = len(np.where(dt >= 2**12-10)[0])
    if FLAG == "session":
        TH = 7
        pSat = pSat/len(dt)*100
    else:
        TH = 4*sampling_rate
    isSt = 0
    if pSat > TH:
        isSt = 1
    if FLAG == "segments":
        pSat = pSat/len(dt)*100
    return np.round(pSat,2), isSt



def zero(dt, sampling_rate, FLAG="segments"):
    pHyp = len(np.where(dt < 0.01)[0])
    isHyp = 0
    #if (np.max(dt) - np.mean(dt)) < 0.5:
    if FLAG == "session":
        TH = 7
        pHyp = pHyp/len(dt)*100
    else:
        TH = 4*sampling_rate
    if pHyp > TH:
        isHyp = 1
    if FLAG == "segments":
        pHyp = pHyp/len(dt)*100
    return np.round(pHyp,2), isHyp



def abn_hr(dt, FLAG="segments"):
    isflat = 0
    if not len(dt):
        return 1

    if FLAG == "segments":
        if len(np.where(dt < 40)[0]) + len(np.where(dt > 200)[0]) > 4*100:
            isflat = 1
    else:
        if len(np.where(dt < 40)[0])/len(dt)*100 + len(np.where(dt > 200)[0])/len(dt)*100 > 7 :
            isflat = 1
    return isflat


def dtLoss(a, sampling_rate, FLAG="segments"):
    ordered = get_loss.order(a)
    loss = get_loss.getError(ordered, sampling_rate)
    is_loss = 0
    if loss > 7:
        is_loss = 1
    return is_loss, loss


def spectral_entropy(x, sfreq):
    nperseg = int(4 * sfreq)  # 4 s window
    fmin = 0.1  # Hz
    fmax = 3  # Hz

    if len(x) < nperseg:  # if segment smaller than 4s
        nperseg = len(x)
    noverlap = int(0.9375 * nperseg)  # if nperseg = 4s, then 3.75 s of overlap
    f, psd = welch(x, sfreq, nperseg=nperseg, noverlap=noverlap)
    psd = np.nan_to_num(psd)
    idx_min = np.argmin(np.abs(f - fmin))
    idx_max = np.argmin(np.abs(f - fmax))
    psd = psd[idx_min:idx_max]
    psd /= np.sum(psd)  # normalize the PSD
    entropy = -np.sum(psd * np.nan_to_num(np.log2(psd)))
    N = idx_max - idx_min
    entropy_norm = entropy / np.log2(N)
    
    isEnt = 0
    if entropy_norm > 0.8:
        isEnt = 1
    return isEnt, np.nan_to_num(entropy_norm)

def quality(raw_y, filtY, noise, hr, packet_number, ts, sampling_rate, FLAG="segments"):
    #SNR = np.round(np.sum(y**2)/np.sum(filtY**2), 2)
    try:
        if np.sum(noise**2) > 0:
            pS = np.sum(filtY**2)/len(filtY)
            pN = np.sum(noise**2)/len(noise)
            SNR = np.nan_to_num(10*np.log(pS/pN)) #np.round(np.sum(filtY**2)/np.sum(y**2), 2)
        else:
            SNR = 0
    except Exception as e:
        print(e)
    pSat, isSt = full_scale(filtY, sampling_rate, FLAG)  # y
    pDisc, isDisc = zero(raw_y, sampling_rate, FLAG)

    matrix = np.concatenate((filtY.reshape(-1, 1), packet_number.reshape(-1, 1), ts.reshape(-1, 1)), axis=1)
    is_loss, loss = dtLoss(matrix.tolist(), sampling_rate)

    isHR = abn_hr(hr, sampling_rate, FLAG)
    isEnt, entropy = spectral_entropy(filtY, sampling_rate)

    return SNR, pSat, isSt, pDisc, isDisc, is_loss, loss, isHR, isEnt, entropy


def remvSat(_signal):
    #signal_norm = (_signal - _signal.min()) / (_signal.max() - _signal.min())
    signal_norm = _signal
    der = np.diff(signal_norm)
    sat_left = scipy.signal.find_peaks(der, height=0.05, distance=35)[0] + 1
    sat_right = scipy.signal.find_peaks(-der, height=0.05, distance=35)[0] -1 
    #%% find nearest point to the left or to the right
    third_point = []


    # iterate over lefts
    sTH = (0.18*(np.max(_signal) - np.mean(_signal)) + np.mean(_signal))
    _sat_left, _sat_right, left_usd = [], [], []
    for l_idx in range(len(sat_left)):
        for r_idx in range(len(sat_right)):                                
            if sat_left[l_idx] < sat_right[r_idx]:
                if (sat_right[r_idx] - sat_left[l_idx]) > 35:
                        continue
        
                if _signal[sat_right[r_idx]] < sTH:
                    continue
                if _signal[sat_left[l_idx]] < sTH:
                    continue

                if sat_left[l_idx] not in left_usd:
                    _sat_left += [sat_left[l_idx]]
                    _sat_right += [sat_right[r_idx]]
                    left_usd += [sat_left[l_idx]]
                    break

    for i in range(len(_sat_left)):  # pressupoe que para cada sat_left há um sat_right
        
        diff_left = _signal[_sat_left[i]] - _signal[_sat_left[i]-1]
        diff_right = _signal[_sat_right[i]+1] + _signal[_sat_right[i]]
        
        if diff_left > diff_right:
            third_point.append(_sat_right[i]+1)
        else:
            third_point.append(_sat_left[i]-1)

    new_signal = _signal
    for i in range(len(_sat_left)):
        xp = np.sort(np.array([_sat_left[i], _sat_right[i], third_point[i]]))
        fp = _signal[xp] 
        coef = np.polyfit(xp, fp, 2)
        
        poly = np.poly1d(coef)
        _x = np.arange(xp[0], xp[-1])
        y = poly(_x)
        
        new_signal[xp[0]: xp[-1]] = y # replace points by extrapolated ones

    return new_signal


idx_to_keep, qualt_inf, dataQlt, allstats, alldt = [], [], {}, [], []
for seg_idx in range(len(data["filt_PPG"])):
    sampling_rate = data["sampling_rate"][seg_idx]
    ID = str(quest_data["ID"][seg_idx])
    raw_y = data["raw_PPG"][seg_idx]
    y = data["filt_PPG"][seg_idx]
    x = data["ts"][seg_idx]
    packet_number = data["packet_number"][seg_idx]
    movie = stimu_data["movie"][seg_idx]
    hr = data["hr"][seg_idx]
    hr_idx = data["hr_idx"][seg_idx]

    y = remvSat(y)
    
    noise, _, _ = bp.signals.tools.filter_signal(
        signal=raw_y,
        ftype="butter",
        band="highpass",
        order=4,
        frequency=15,
        sampling_rate=sampling_rate,
    )

    SNR, pSat, isSt, pDisc, isDisc, is_loss, loss, noPks, isEnt, entropy = quality(raw_y, y, noise, hr, packet_number, x*100000, sampling_rate, FLAG)

    qualt_inf += [[ID + "_" + movie, SNR, pSat, pDisc, loss, isEnt]]

    dataQlt[ID + "_" + movie + "_" + str(seg_idx)] = {}
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["File"] = ID + "_" + movie + "_" + str(seg_idx)
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["SNR"] = SNR
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["pSat"] = pSat
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["pDisc"] = pDisc
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Loss"] = loss
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Abnormal"] = noPks*100
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Spectral Entropy"] = entropy
    _max = np.max(y)
    _min = np.min(y)
    _mean = np.mean(y)
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Max"] = _max
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Min"] = _min
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Mean"] = _mean
    allstats += [dataQlt[ID + "_" + movie + "_" + str(seg_idx)]]

    alldt += [[ID + "_" + movie + "_" + str(seg_idx), SNR, pSat, pDisc, loss, noPks*100, entropy, _max, _mean, _min]]

    dtAssessment = [isSt, isDisc, is_loss, noPks]
    print("dt", dtAssessment)

    plt.figure(dpi=200) # EDA plot
    plt.title("M: " + movie + "; D: " + ID)

    if np.sum(dtAssessment) >= 1:
        plt.plot(x, raw_y, label="Raw PPG", color="#073b4c")
        plt.plot(x, y, label="Filtered PPG", color="red")
    else:
        plt.plot(x, raw_y, label="Raw PPG", color="#073b4c")
        plt.plot(x, y, label="Filtered PPG", color="#2a9d8f")
        idx_to_keep += [seg_idx] 
    if len(hr_idx) > 1:  # if a ppg HR-peak was found
        plt.vlines(x=x[hr_idx], ymin=y.min(), ymax=y.max(), color="#264653", alpha=0.3, label="HR Peak")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (bits)")
    #plt.show()
    plt.savefig("../6_Results/PPG/" + FLAG +"/Plots/D" + ID + "_M" + movie + "_idx" + str(seg_idx) + "_" + FLAG +"_PPG.png", format="png", bbox_inches="tight")

data["PPG_quality_idx"] = np.array(idx_to_keep)

with open(DATA_DIR, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


alldt = np.array(alldt)

label = ["User", "SNR", "pSat", "pDisc", "Loss", "Abnormal", "Spectral Entropy", "Max", "Mean", "Min"]

keep_dt = alldt[idx_to_keep]
res_all = ["Average", str(np.round(np.mean(keep_dt[:, 1].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 1].astype(float)), 2)),\
           str(np.round(np.mean(keep_dt[:, 2].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 2].astype(float)),2 )) ,\
           str(np.round(np.mean(keep_dt[:, 3].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 3].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)) + "+-" + str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)),
           str(np.round(np.mean(keep_dt[:, 5].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 5].astype(float)), 2)),\
           str(np.round(np.mean(keep_dt[:, 6].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 6].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 7].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 7].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 8].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 8].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 9].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 9].astype(float)), 2))]


keep_dt = keep_dt.tolist()
_keep_dt = [*res_all]
keep_dt += [_keep_dt]
df = pd.DataFrame.from_dict(keep_dt) 
df.to_csv ('../6_Results/PPG/' + FLAG + '/Quality/PPG_quality_good_' + FLAG + '.csv', index = True, header=label)

to_rmv = np.delete(np.arange(len(alldt)), idx_to_keep)
keep_dt = alldt[to_rmv]
res_all = ["Average", str(np.round(np.mean(keep_dt[:, 1].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 1].astype(float)), 2)),\
           str(np.round(np.mean(keep_dt[:, 2].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 2].astype(float)),2 )) ,\
           str(np.round(np.mean(keep_dt[:, 3].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 3].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)) + "+-" + str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)),
           str(np.round(np.mean(keep_dt[:, 5].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 5].astype(float)), 2)),\
           str(np.round(np.mean(keep_dt[:, 6].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 6].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 7].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 7].astype(float)), 2)), \
           str(np.round(np.mean(keep_dt[:, 8].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 8].astype(float)), 2)), \
            str(np.round(np.mean(keep_dt[:, 9].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 9].astype(float)), 2))]

keep_dt = keep_dt.tolist()
_keep_dt = [*res_all]
keep_dt += [_keep_dt]
#allstats_f = np.array(allstats)[[-1]]
df = pd.DataFrame.from_dict(keep_dt) 
df.to_csv ('../6_Results/PPG/' + FLAG + '/Quality/PPG_quality_bad_' + FLAG + '.csv', index = True, header=label)
print("Done")
print("Flag:", FLAG, " PPG")
pdb.set_trace()