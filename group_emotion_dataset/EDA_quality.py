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
import pdb

# local
import biosppy as bp
import get_loss


FLAG = "session"  # "session", "segments"
if FLAG == "session":
    DATA_DIR = '../3_Physio/Transformed/physio_trans_data_session.pickle'
    QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_session.pickle'
    STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_session.pickle'
else:
    DATA_DIR = '../3_Physio/Transformed/physio_trans_data_segments.pickle'
    QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_segments.pickle'
    STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_segments.pickle'


files = glob.glob("../6_Results/EDA/" + FLAG + "/Plots/*")
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
    dt = ((dt/2**12) * 3.3)/0.12
    pHyp = len(np.where(dt < 0.05)[0])
    isHyp = 0
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



def dtLoss(a, sampling_rate, FLAG="segments"):
    ordered = get_loss.order(a)
    loss = get_loss.getError(ordered, sampling_rate)
    is_loss = 0
    if loss > 7:
        is_loss = 1
    return is_loss, loss


def quality(y, raw_y, noise, packet_number, ts, sampling_rate, FLAG="segments"):
    try:
        if np.sum(noise**2) > 0:
            pS = np.sum(y**2)/len(y)
            pN = np.sum(noise**2)/len(noise)
            SNR = 10*np.log(pS/pN) 
        else:
            SNR = 0
    except Exception as e:
        print(e)
    pSat, isSt = full_scale(y, sampling_rate, FLAG)  # y
    pDisc, isDisc = zero(raw_y, sampling_rate, FLAG)
    matrix = np.concatenate((y.reshape(-1, 1), packet_number.reshape(-1, 1), ts.reshape(-1, 1)), axis=1)
    is_loss, loss = dtLoss(matrix.tolist(), sampling_rate, FLAG)

    return SNR, pSat, isSt, pDisc, isDisc, is_loss, loss


idx_to_keep, qualt_inf, dataQlt, allstats, alldt = [], [], {}, [], []
for seg_idx in range(len(data["filt_EDA"])):
    sampling_rate = data["sampling_rate"][seg_idx]
    ID = str(quest_data["ID"][seg_idx])
    y = data["filt_EDA"][seg_idx]
    raw_y = data["raw_EDA"][seg_idx]
    x = data["ts"][seg_idx]
    packet_number = data["packet_number"][seg_idx]
    movie = stimu_data["movie"][seg_idx]
    
    # filter signal
    noise, _, _ = bp.signals.tools.filter_signal(
        signal=raw_y,
        ftype="butter",
        band="bandpass",
        order=3,
        frequency=[2, 10],
        sampling_rate=sampling_rate,
    )
    
    SNR, pSat, isSt, pDisc, isDisc, is_loss, loss = quality(y, raw_y, noise, packet_number, x*100000, sampling_rate, FLAG)

    qualt_inf += [[ID + "_" + movie, pSat, pDisc, SNR,  loss]]

    dataQlt[ID + "_" + movie + "_" + str(seg_idx)] = {}
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["File"] = ID + "_" + movie + "_" + str(seg_idx)
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["SNR"] = SNR
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["pSat"] = pSat
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["pDisc"] = pDisc
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Loss"] = loss
    dtAssessment = [isSt, isDisc, is_loss]

    print("dt", dtAssessment)
    plt.figure(dpi=200) # EDA plot
    plt.title("M: " + movie + "; D: " + ID)

    if np.sum(dtAssessment) >= 1:
        plt.plot(x, raw_y, ".", label="Raw EDA", color="red")
        plt.plot(x, y, ".", label="Filtered EDA", color="red")
    else:
        plt.plot(x, raw_y, ".", label="Raw EDA", color="#073b4c")
        plt.plot(x, y, ".", label="Filtered EDA", color="#2a9d8f")       
        idx_to_keep += [seg_idx] 

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (bits)")
    #plt.show()
    plt.savefig("../6_Results/EDA/" + FLAG +"/Plots/D" + ID + "_M" + movie + "_idx" + str(seg_idx) + "_seg_EDA.png", format="png", bbox_inches="tight")

    y = ((y/2**12) * 3.3)/0.12
    _max = np.max(y)
    _min = np.min(y)
    _mean = np.mean(y)
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Max"] = _max
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Min"] = _min
    dataQlt[ID + "_" + movie + "_" + str(seg_idx)]["Mean"] = _mean

    allstats += [dataQlt[ID + "_" + movie + "_" + str(seg_idx)]]
    alldt += [[ID + "_" + movie + "_" + str(seg_idx), SNR, pSat, pDisc, loss, _max, _mean, _min]]

data["EDA_quality_idx"] = np.array(idx_to_keep)

with open(DATA_DIR, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


alldt = np.array(alldt)

label = ["User","SNR", "pSat", "pDisc","Loss", "Max", "Mean", "Min"]

keep_dt = alldt[idx_to_keep]
res_all = ["Average", str(np.round(np.mean(keep_dt[:, 1].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 1].astype(float)), 2)), \
       str(np.round(np.mean(keep_dt[:, 2].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 2].astype(float)),2 )), \
          str(np.round(np.mean(keep_dt[:, 3].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 3].astype(float)), 2)),  \
            str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)) + "+-" + str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)),  \
                    str(np.round(np.mean(keep_dt[:, 5].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 5].astype(float)), 2)), \
                        str(np.round(np.mean(keep_dt[:, 6].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 7].astype(float)), 2)), \
                            str(np.round(np.mean(keep_dt[:, 7].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 7].astype(float)), 2))]
_keep_dt = [*res_all]
keep_dt = keep_dt.tolist()
keep_dt += [_keep_dt]

df = pd.DataFrame.from_dict(keep_dt) 
df.to_csv ('../6_Results/EDA/' + FLAG + '/Quality/EDA_quality_good_' + FLAG + '.csv', index = True, header=label)

to_rmv = np.delete(np.arange(len(alldt)), idx_to_keep)
keep_dt = alldt[to_rmv]
res_all = ["Average", str(np.round(np.mean(keep_dt[:, 1].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 1].astype(float)), 2)), \
       str(np.round(np.mean(keep_dt[:, 2].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 2].astype(float)),2 )), \
          str(np.round(np.mean(keep_dt[:, 3].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 3].astype(float)), 2)),  \
            str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)) + "+-" + str(np.round(np.mean(keep_dt[:, 4].astype(float)), 2)),  \
                    str(np.round(np.mean(keep_dt[:, 5].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 5].astype(float)), 2)), \
                        str(np.round(np.mean(keep_dt[:, 6].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 6].astype(float)), 2)), \
                            str(np.round(np.mean(keep_dt[:, 7].astype(float)), 2)) + "+-" + str(np.round(np.std(keep_dt[:, 7].astype(float)), 2))]
_keep_dt = [*res_all]
keep_dt = keep_dt.tolist()
keep_dt += [_keep_dt]

df = pd.DataFrame.from_dict(keep_dt) 
df.to_csv ('../6_Results/EDA/' + FLAG + '/Quality/EDA_quality_bad_' + FLAG + '.csv', index = True, header=label)

print("Done")
print("Flag:", FLAG, " EDA")
pdb.set_trace()



