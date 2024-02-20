# -*- coding: utf-8 -*-
"""
This file receives as input the volunteers questionnaire data, the raw physiological data and video data, and returns the transformed physiological data, questionnaire data, stimuli and annotations per session and segments.
"""

# Imports
# 3rd party
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from matplotlib.patches import Rectangle
import os
import pandas as pd

# local
import biosppy as bp


colors = ["#e9c46a", "#073b4c", "#2a9d8f", "#75a3a3", "#264653", "#e63946"]


files = np.hstack((glob.glob("../6_Results/EDA/session/Plots/*"), glob.glob("../6_Results/PPG/session/Plots/*"), glob.glob("../6_Results/EDA/segments/Plots/*"), glob.glob("../6_Results/PPG/segments/Plots/*")))
for f in files:
    os.remove(f)


genre_dic = pd.read_csv('../1_Stimuli/Raw/video_info.csv', index_col=0)
users_info = pd.read_csv('../2_Questionnaire/Raw/quest_raw_data.csv', index_col=0)

DIR = np.sort(glob.glob("../3_Physio/Raw/*.hdf5"))  # replace with directory where the file is

Loss, dur = [], []
data, quest_trans_data, stimu_trans_data, quest_trans_data_session, stimuli_trans_data_session, ann_data, session_data  = {}, {}, {}, {}, {}, {}, {}

data["filt_EDA"], data["filt_PPG"], data["ts"], data["sampling_rate"], data["packet_number"], data["EDR"], data["hr"], data["raw_EDA"], data["raw_PPG"], data["hr_idx"] = [], [], [], [], [], [], [], [], [], []
ann_data["ar_seg"], ann_data["vl_seg"], ann_data["unc_seg"], ann_data["ts_seg"] = [], [], [], []
quest_trans_data["ID"], quest_trans_data["device"] = [], []
stimu_trans_data["movie"], stimu_trans_data["genre"], stimu_trans_data["session"] = [], [], []
session_data["filt_EDA"], session_data["filt_PPG"], session_data["ts"], session_data["packet_number"], session_data["raw_EDA"], session_data["sampling_rate"], session_data["raw_PPG"], session_data["hr"], session_data["EDR"], session_data["hr_idx"] = [], [], [], [], [], [], [], [], [], []
stimuli_trans_data_session["movie"], stimuli_trans_data_session["genre"], stimuli_trans_data_session["session"] = [], [], []
quest_trans_data_session["ID"], quest_trans_data_session["device"] = [], []

skipped_user = []
no_acc_c = 0
for d in DIR:  # iterate over file
    f = h5py.File(d, 'r')  # open file
    d = d.split("/")[-1]
    d_s = d.split("_")
    d = d_s[0] + "_" + d_s[1]
    print("File: ", f)
    for ID in list(f.keys()):  # iterate over devices
        if ID != "LOGs" and ID != "Flags":  # is a device
            print("ID", ID)
            try:
                ar_ann_m = f[ID]["Arousal EDA"] 
                vl_ann_m = f[ID]["Valence EDA"] 
            except Exception as e:
                print(e)
                print("USER DID NOT ANNOTATE")
                no_acc_c += 1
                continue

            ts_ar_ann = ar_ann_m[:, 1]
            ts_vl_ann = vl_ann_m[:, 1]

            unique_ann_ts = set(ts_ar_ann) & set(ts_vl_ann)
            
            # sync annotated segments start time
            un_idx_ar_ann, un_idx_vl_ann  = [], []
            for ann_ts in unique_ann_ts:
                un_idx_ar_ann += [np.where(ann_ts == ts_ar_ann)[0][-1]]
                un_idx_vl_ann += [np.where(ann_ts == ts_vl_ann)[0][-1]]
            if not len(un_idx_ar_ann) and not len(un_idx_vl_ann):
                print("USER DID NOT ANNOTATE")
                no_acc_c += 1
                continue

            assert len(un_idx_ar_ann) > 0
            assert len(un_idx_vl_ann) > 0

            ts_ar_ann = ts_ar_ann[un_idx_ar_ann]
            ts_vl_ann = ts_vl_ann[un_idx_vl_ann]
            un_idx_ar_ann = np.sort(un_idx_ar_ann)
            un_idx_vl_ann = np.sort(un_idx_vl_ann)
            assert len(ts_ar_ann) == len(ts_vl_ann)
            assert (ts_ar_ann == ts_vl_ann).all()
            assert len(np.unique(ts_ar_ann)) == len(np.unique(ts_vl_ann))

            u_k = []
            for k_i, key in enumerate(list(f[ID].keys())):
                if "Uncertainty" in key:
                    u_k += [list(f[ID].keys())[k_i]]
                
            if len(u_k) > 1:
                u_ar_ann_ts = f[ID][u_k[0]][:, 1]
                u_vl_ann_ts = f[ID][u_k[1]][:, 1]

                u_ar_ann_vl = f[ID][u_k[0]][:, -1]
                u_vl_ann_vl = f[ID][u_k[1]][:, -1]
                
                # sync annotated segments start time
                unique_ann_ts = set(u_ar_ann_ts) & set(u_vl_ann_ts)
                un_idx_ar_ann_u, un_idx_vl_ann_u  = [], []
                for ann_ts in unique_ann_ts:
                    un_idx_ar_ann_u += [np.where(ann_ts == u_ar_ann_ts)[0][-1]]
                    un_idx_vl_ann_u += [np.where(ann_ts == u_vl_ann_ts)[0][-1]]
                un_idx_ar_ann_u = np.sort(un_idx_ar_ann_u)
                un_idx_vl_ann_u = np.sort(un_idx_vl_ann_u)

                u_ar_ann_ts = u_ar_ann_ts[un_idx_ar_ann_u]
                u_vl_ann_ts = u_vl_ann_ts[un_idx_vl_ann_u]
                assert len(u_ar_ann_ts) == len(u_vl_ann_ts)

                u_ar_ann_vl = u_ar_ann_vl[un_idx_ar_ann_u]
                u_vl_ann_vl = u_vl_ann_vl[un_idx_vl_ann_u]

                u_ann_ts = np.mean(np.hstack((u_ar_ann_ts.reshape(-1, 1), u_vl_ann_ts.reshape(-1, 1))), axis=1)
                u_ann_vl = np.mean(np.hstack((u_ar_ann_vl.reshape(-1, 1), u_vl_ann_vl.reshape(-1, 1))), axis=1)
                
            elif len(u_k) > 0:  # user annotated
                u_ar_ann_ts = f[ID][u_k[0]][:, 1]
                u_ar_ann_vl = f[ID][u_k[0]][:, -1]

                u_ann_ts, u_ann_vl = [], []
                for ann_ts in ts_ar_ann:
                    a_ts_i = np.where(ann_ts == u_ar_ann_ts)[0]
                    if len(a_ts_i) > 1:
                        u_ann_ts += [u_ar_ann_ts[a_ts_i[-1]]]
                        u_ann_vl += [u_ar_ann_vl[a_ts_i[-1]]]
                    else:
                        u_ann_ts += [ann_ts]
                        u_ann_vl += [None]
                u_ann_vl = np.array(u_ann_vl)
                u_ann_ts = np.array(u_ann_ts)
            else:
                u_ann_vl = np.array([None])
                u_ann_ts = np.array([None])
            
            # assert all annotations (unc, ar, vl) are synchronised
            _u_ann_ts, _u_ann_vl  = [], []
            for ann_ts in ts_ar_ann:  # iterate over ann ts
                idx_ts = np.where(ann_ts == u_ann_ts)[0]
                if not len(idx_ts):
                    _u_ann_ts += [ann_ts]
                    _u_ann_vl += [None]
                else:
                    _u_ann_ts += [u_ann_ts[idx_ts][0]]
                    _u_ann_vl += [u_ann_vl[idx_ts][0]]

            u_ann_ts = np.array(_u_ann_ts)
            u_ann_vl = np.array(_u_ann_vl)

            assert (u_ann_ts == ts_ar_ann).all()
            
            sampling_rate = f[ID].attrs["sampling rate"]
            movie = f[ID].attrs["movie"] 
            genre = genre_dic['genre'][np.where(genre_dic["movie_name"].values == movie)[0][0]]

            # cut segments in eda signal
            EDA = f[ID]['data'][:, 0]
            PPG = f[ID]['data'][:, 1]
            packet_number = f[ID]['data'][:, -2]
            ts_or = f[ID]['data'][:, -1]
            assert len(EDA) == len(PPG)
            ts = np.arange(0, len(EDA)/sampling_rate, 1/sampling_rate)
            dur += [ts[-1]]
            d_0 = np.round(dur[-1]) 
            d_1 = np.round(f[ID]['data'][-1, -1]*0.000001- f[ID]['data'][0, -1]*0.000001)
 
            ID = ID[2:]
            idx = list(set(users_info[users_info["movie"]==movie].index.tolist()) & set(users_info[users_info["device"].values.astype(int).astype(str)==ID].index.tolist()))

            if not len(idx) > 0:
                mv_idx = []
                for m_i, mv, in enumerate(users_info["movie"].values.astype(str)):
                    if movie in mv:
                        assert movie in mv
                        if users_info["device"].values[m_i].astype(str) == ID:
                            idx = [m_i]
                            break

            
            if not len(idx) > 0:
                print("skipped user for no data", movie, ID)
                skipped_user += [[movie, ID]]
                no_acc_c += 1
                continue
            
            assert users_info["device"][idx].values[0].astype(str) == ID
            if not len(ann_data["ar_seg"]):
                ann_data["ar_seg"] = ar_ann_m[un_idx_ar_ann][:, -1].tolist()
                ann_data["vl_seg"] = vl_ann_m[un_idx_vl_ann][:, -1].tolist()
                ann_data["ts_seg"] = ts_ar_ann.tolist()
                ann_data["unc_seg"] = u_ann_vl.tolist()

                data["sampling_rate"] = [sampling_rate] * len(un_idx_ar_ann)
                quest_trans_data["device"] = [ID] * len(un_idx_ar_ann)
                stimu_trans_data["movie"] = [movie] * len(un_idx_ar_ann)
                stimu_trans_data["genre"] = [genre] * len(un_idx_ar_ann)
                quest_trans_data["ID"] = [users_info["ID"].values[idx][0]] * len(un_idx_ar_ann)
                stimu_trans_data["session"] = [users_info["session"].values[idx][0]] * len(un_idx_ar_ann)
            else:
                ann_data["ar_seg"] += ar_ann_m[un_idx_ar_ann][:, -1].tolist()
                ann_data["vl_seg"] += vl_ann_m[un_idx_vl_ann][:, -1].tolist()
                ann_data["ts_seg"] += vl_ann_m[un_idx_vl_ann][:, 1].tolist()
                ann_data["unc_seg"] += u_ann_vl.tolist()

                data["sampling_rate"] += [sampling_rate] * len(u_ann_vl)
                quest_trans_data["device"] += [ID] * len(u_ann_vl)
                stimu_trans_data["movie"] += [movie] * len(u_ann_vl)
                stimu_trans_data["genre"] += [genre] * len(u_ann_vl)
                quest_trans_data["ID"] += [users_info["ID"].values[idx][0]] * len(u_ann_vl)
                stimu_trans_data["session"] += [users_info["session"].values[idx][0]] * len(u_ann_vl)

            assert len(EDA) == len(ts)
            # assert all sizes match
            assert len(stimu_trans_data["movie"]) == len(ann_data["ar_seg"]) == len(quest_trans_data["device"]) == len(stimu_trans_data["genre"]) == len(stimu_trans_data["session"]) == len(quest_trans_data["ID"]) == len(ann_data["unc_seg"]) == len(ann_data["ts_seg"]) == len(data["sampling_rate"])

            # Create figure and axes
            fig, ax = plt.subplots(dpi=200)
            plt.title("M: " + movie + "; D: " + ID)
            
            # filter signal
            aux, _, _ = bp.signals.tools.filter_signal(
                signal=EDA,
                ftype="butter",
                band="lowpass",
                order=4,
                frequency=5,
                sampling_rate=sampling_rate,
            )

            # smooth
            sm_size = int(.75 * sampling_rate)
            EDA_filt, _ = bp.signals.tools.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

            eda_m = EDA_filt.mean()
            eda_std = EDA_filt.std()

            edr = bp.signals.eda.biosppy_decomposition(EDA_filt, sampling_rate=sampling_rate)["edr"]
            edr_m = edr.mean()
            edr_std = edr.std()
            
            plt.plot(ts, EDA, ".", label="Raw EDA", color="#073b4c")
            plt.plot(ts, EDA_filt, label="Filtered EDA", color="#2a9d8f")
            for seg in range(len(ts_ar_ann)):
                left, bottom, width, height = (ts_ar_ann[seg], np.min(EDA), 20, np.max(EDA) + 1)
                rect = Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1, label="Annotations")
                ax.add_patch(rect)
                plt.axvline(x=ts_ar_ann[seg], ymin=np.min(EDA), ymax=np.max(EDA)+1, color="#264653", alpha=0.3)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (bits)")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            #plt.show()
            plt.savefig("../6_Results/EDA/session/Plots/D" + ID + "_M" + movie + "_session_EDA.png", format="png", bbox_inches = "tight")
            
            # filter signal            
            PPG_filt, _, _ = bp.signals.tools.filter_signal(signal=PPG,
                                              ftype='butter',
                                              band='bandpass',
                                              order=4,
                                              frequency=[1, 8],
                                              sampling_rate=sampling_rate)
            PPG_m = PPG_filt.mean()
            PPG_std = PPG_filt.std()
      
            plt.figure(dpi=200) # PPG plot
            plt.title("M: " + movie + "; D: " + ID)
            plt.plot(ts, PPG ,".", label="Raw PPG", color="#073b4c")
            plt.plot(ts, PPG_filt ,".", label="Filtered PPG", color="#2a9d8f")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (bits)")
            #plt.show()
            plt.savefig("../6_Results/PPG/session/Plots/D" + ID + "_M" + movie + "_session_PPG.png", format="png", bbox_inches = "tight")
            plt.close("all") 

            # get average hr

            
            for ann_ts in ts_ar_ann:
                color_seg = "g"
                idx_y_ann_ts = np.where(ts >= ann_ts)[0][0]
                _eda = EDA_filt[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]
                _edr = edr[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]
                _ppg = PPG_filt[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]

                data["raw_EDA"] += [EDA[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]]
                data["filt_EDA"] += [_eda]
                
                data["EDR"] += [_edr]
                
                data["raw_PPG"] += [PPG[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]]
                data["filt_PPG"] += [_ppg]

                # find onsets
                try:
                    onsets, _ = bp.signals.ppg.find_onsets_elgendi2013(signal=data["filt_PPG"][-1], sampling_rate=sampling_rate)

                    # compute heart rate
                    hr_idx, hr = bp.signals.tools.get_heart_rate(beats=onsets,
                                               sampling_rate=sampling_rate,
                                               smooth=True,
                                               size=3)
                except Exception as e:
                    print(e, "exception obtaining r-peaks in segment")
                    hr = np.array([])
                    hr_idx = np.array([])

                    color_seg = "red"
                    
                data["hr"] += [hr]
                data["hr_idx"] += [hr_idx]

                data["ts"] += [ts[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]]

                data["packet_number"] += [packet_number[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate]]
                
                plt.figure(dpi=200) # EDA plot
                plt.title("M: " + movie + "; D: " + ID)
                plt.plot(data["ts"][-1], EDA[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate], ".", label="Raw EDA", color="#073b4c")
                plt.plot(data["ts"][-1], _eda, ".", label="Filtered EDA", color="#2a9d8f")
                plt.legend()
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude (bits)")
                #plt.show()
                plt.savefig("../6_Results/EDA/segments/Plots/D" + ID + "_M" + movie + "_idx" + str(len(data["ts"])) + "_segments_EDA.png", format="png", bbox_inches="tight")
                plt.close("all") 
                
                # PPG plot
                plt.figure(dpi=200)
                plt.title("M: " + movie + "; D: " + ID)
                plt.plot(data["ts"][-1], PPG[idx_y_ann_ts:idx_y_ann_ts+20*sampling_rate], ".", label="Raw PPG", color="#073b4c")
                plt.plot(data["ts"][-1], _ppg, ".", label="Filtered PPG", color="#2a9d8f")
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude (bits)")
                if len(hr_idx) > 1:  # if a ppg HR-peak was found
                    plt.vlines(x=data["ts"][-1][hr_idx], ymin=_ppg.min(), ymax=_ppg.max(), color="#264653", alpha=0.3, label="HR Peak")
                plt.legend()
                #plt.show()
                plt.savefig("../6_Results/PPG/segments/Plots/D" + ID + "_M" + movie + "_idx" + str(len(data["ts"])) + "_segments_PPG.png", format="png", bbox_inches="tight")
                plt.close("all") 

            assert len(stimu_trans_data["movie"]) == len(data["filt_EDA"])
            
            session_data["filt_EDA"] += [EDA_filt]
            session_data["raw_EDA"] += [EDA]
            session_data["EDR"] += [edr]
            stimuli_trans_data_session["session"] += [users_info["session"].values[idx][0]]
            session_data["filt_PPG"] += [PPG_filt]
            session_data["raw_PPG"] += [PPG]
            session_data["ts"] += [ts]
            session_data["packet_number"] += [packet_number]
            stimuli_trans_data_session["movie"] += [movie]
            quest_trans_data_session["ID"] += [users_info["ID"].values[idx][0]]
            quest_trans_data_session["device"] += [ID]
            stimuli_trans_data_session["genre"] += [genre]
            session_data["sampling_rate"] += [sampling_rate]

            # find onsets
            try:
                onsets, _ = bp.signals.ppg.find_onsets_elgendi2013(signal=PPG_filt, sampling_rate=sampling_rate)

                # compute heart rate
                hr_idx, hr = bp.signals.tools.get_heart_rate(beats=onsets,
                                           sampling_rate=sampling_rate,
                                           smooth=True,
                                           size=3)
            except Exception as e:
                print(e, "exception obtaining r-peaks in session data")

                hr = np.array([])
                hr_idx = np.array([])

            session_data["hr"] += [hr]
            session_data["hr_idx"] += [hr_idx]

            assert len(data["filt_EDA"]) == len(data["filt_PPG"]) and len(data["filt_EDA"]) == len(data["ts"]) and len(data["filt_EDA"]) == len(data["packet_number"]) and len(data["filt_EDA"]) == len(data["hr"]) and len(data["filt_EDA"]) == len(data["hr_idx"])
            assert len(session_data["filt_EDA"]) == len(stimuli_trans_data_session["session"]) and len(session_data["filt_EDA"]) == len(session_data["raw_EDA"]) and len(session_data["filt_EDA"]) == len(session_data["EDR"]) and len(session_data["filt_EDA"]) == len(session_data["filt_PPG"]) and len(session_data["filt_EDA"]) == len(session_data["raw_PPG"]) and len(session_data["filt_EDA"]) == len(session_data["ts"]) and len(session_data["filt_EDA"]) == len(session_data["packet_number"]) and len(session_data["filt_EDA"]) == len(session_data["hr"]) and len(session_data["filt_EDA"]) == len(session_data["hr_idx"]) and len(session_data["filt_EDA"]) == len(session_data["sampling_rate"])


print("skipped users: ",  skipped_user)
# store data in pickle
with open('../4_Annotation/Transformed/ann_trans_data_segments.pickle', 'wb') as handle:
    pickle.dump(ann_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../3_Physio/Transformed/physio_trans_data_segments.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../3_Physio/Transformed/physio_trans_data_session.pickle', 'wb') as handle:
    pickle.dump(session_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../2_Questionnaire/Transformed/quest_trans_data_segments.pickle', 'wb') as handle:
    pickle.dump(quest_trans_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../2_Questionnaire/Transformed/quest_trans_data_session.pickle', 'wb') as handle:
    pickle.dump(quest_trans_data_session, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../1_Stimuli/Transformed/stimu_trans_data_session.pickle', 'wb') as handle:
    pickle.dump(stimuli_trans_data_session, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../1_Stimuli/Transformed/stimu_trans_data_segments.pickle', 'wb') as handle:
    pickle.dump(stimu_trans_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("Users that did not annotate", no_acc_c) 
print("Number of segment samples", len(data["filt_EDA"]))
print("Number of session samples", len(session_data["filt_EDA"]))
