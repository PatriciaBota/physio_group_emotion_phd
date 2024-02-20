# -*- coding: utf-8 -*-
"""
This file analyses the annotations and physiological data across the annotation dimensions. 
Returns: 
    Annotation 2d plot
    Histogram of annotations, genre, and violin plots of eda and hr across annotation score.
"""

# Imports
# 3rd party
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pickle
import os
import glob
import pandas as pd
from scipy import stats
import pdb

# local
import biosppy as bp


# personality analysis
users_info = pd.read_csv('../2_Questionnaire/Raw/quest_raw_data.csv', index_col=0)
pers_matrix = users_info['personality'].values

pers_m = []
for row in pers_matrix:
    nr = row.split(",")
    try:
        pers_m += [np.array([float(nr[0][1:]), float(nr[1]), float(nr[2]), float(nr[3]), float(nr[4][:-1])])] # 1 and -1 remove [ and ]
    except:  # none
        continue
pers_m = np.array(pers_m, dtype=object)
print("len users with personality", len(pers_m))

LABEL_PERS = ["Extraversion", "Neuroticism", "Openness", "Conscientiousness", "Agreeableness"]

def change_vio_color(violin):
    # Set the color of the violin patches
    for pc in violin['bodies']:
        pc.set_facecolor("#2a9d8f")
    # Set the color of the median lines
    violin['cmedians'].set_colors(["#073b4c"])

violin_dt = []
for i in range(len(LABEL_PERS)):
    violin_dt += [pers_m[:, i].tolist()]
x = list(range(len(LABEL_PERS)))

replaced_data = []

for sublist in violin_dt:
    new_sublist = []
    for value in sublist:
        if np.isnan(value):
            continue
        else:
            new_sublist.append(value)
    replaced_data.append(new_sublist)

plt.figure(dpi=200)
violin = plt.violinplot(replaced_data, x, showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(violin_dt[0], violin_dt[1], violin_dt[2], violin_dt[3], violin_dt[4])))
#plt.title("Personality")
#plt.xlabel(LABEL_PERS)
plt.xticks(x, labels=LABEL_PERS, rotation = 45)
plt.ylim(0, 1)
plt.ylabel("Personality Score")
plt.savefig("../6_Results/Analysis/pers_violin.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

age = users_info['age'].values

age_counts = {value: list(age).count(value) for value in set(age)}

plt.figure(dpi=200)
# Sort the keys in ascending order
sorted_keys = sorted(age_counts.keys(), key=lambda x: int(x.split('-')[0]))  # assuming the keys are in 'start-end' format
# Extract corresponding values
sorted_values = [age_counts[key] for key in sorted_keys]
plt.bar(sorted_keys, sorted_values, color='#2a9d8f', width=0.4)
plt.xlabel("Age Range")
plt.ylabel("No. of Users")
plt.savefig("../6_Results/Analysis/users_age.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()
#print("pers_violin: " + "f_oneway: " + str(stats.f_oneway(*violin_dt)))


SESSION_DATA_DIR = '../3_Physio/Transformed/physio_trans_data_session.pickle'
SESSION_QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_session.pickle'
SESSION_STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_session.pickle'

SEG_DATA_DIR = '../3_Physio/Transformed/physio_trans_data_segments.pickle'
SEG_QUEST_DATA_DIR = '../2_Questionnaire/Transformed/quest_trans_data_segments.pickle'
SEG_STIMU_DATA_DIR = '../1_Stimuli/Transformed/stimu_trans_data_segments.pickle'
SEG_ANN_DATA_DIR = '../4_Annotation/Transformed/ann_trans_data_segments.pickle'

with open(SEG_DATA_DIR, 'rb') as handle:  # read data
    seg_data = pickle.load(handle)
with open(SEG_STIMU_DATA_DIR, 'rb') as handle:  # read data
    seg_stimu_data = pickle.load(handle)
with open(SEG_QUEST_DATA_DIR, 'rb') as handle:  # read data
    seg_quest_data = pickle.load(handle)
with open(SEG_ANN_DATA_DIR, 'rb') as handle:  # read data
    seg_ann_data = pickle.load(handle)

with open(SESSION_DATA_DIR, 'rb') as handle:  # read data
    session_data = pickle.load(handle)
with open(SESSION_STIMU_DATA_DIR, 'rb') as handle:  # read data
    session_stimu_data = pickle.load(handle)
with open(SESSION_QUEST_DATA_DIR, 'rb') as handle:  # read data
    session_quest_data = pickle.load(handle)

files = np.hstack((glob.glob("../6_Results/EDA/" + "session" + "/Analysis/*"), glob.glob("../6_Results/PPG/" + "session" + "/Analysis/*"), glob.glob("../6_Results/EDA/" + "segments" + "/Analysis/*"), glob.glob("../6_Results/PPG/" + "segments" + "/Analysis/*")))
for f in files:
    os.remove(f)

print("Before outlier removal")

idx_to_keep_EDA = seg_data["EDA_quality_idx"]
idx_to_keep_PPG = seg_data["PPG_quality_idx"]
print("seg idx to keep (EDA PPG):", len(idx_to_keep_EDA), len(idx_to_keep_PPG))
idx_to_keep = list(set(idx_to_keep_EDA) & set(idx_to_keep_PPG))
good_dt = len(idx_to_keep)
print("Total len samples after outliers removal (seg): ", good_dt)


raw_EDA_seg = np.array(seg_data["raw_EDA"])#[idx_to_keep]
EDA_seg = np.array(seg_data["filt_EDA"])#[idx_to_keep]
ar_seg = np.array(seg_ann_data["ar_seg"])#[idx_to_keep]
unc_seg = np.array(seg_ann_data["unc_seg"], dtype=object)#[idx_to_keep]
vl_seg = np.array(seg_ann_data["vl_seg"], dtype=object)#[idx_to_keep]
hr_seg = np.array(seg_data["hr"], dtype=object)#[idx_to_keep]
movies = np.array(seg_stimu_data["movie"])#[idx_to_keep]
genres = np.array(seg_stimu_data["genre"])#[idx_to_keep]
ID = np.array(seg_quest_data["ID"])#[idx_to_keep]
session_seg = np.array(seg_stimu_data["session"])#[idx_to_keep]
devices = np.array(seg_quest_data["device"])#[idx_to_keep]

assert len(raw_EDA_seg) == len(EDA_seg) == len(ar_seg) == len(unc_seg) == len(vl_seg) == len(hr_seg) == len(movies) == len(genres) == len(ID) == len(session_seg) == len(devices)

# session data
idx_to_keep_EDA = session_data["EDA_quality_idx"]
idx_to_keep_PPG = session_data["PPG_quality_idx"]
print("session idx to keep (EDA PGG):", len(idx_to_keep_EDA), len(idx_to_keep_PPG))
idx_to_keep_sess = list(set(idx_to_keep_EDA) & set(idx_to_keep_PPG))
print("idx to keep all: ", len(idx_to_keep_sess))


session_EDA = np.array(session_data["filt_EDA"], dtype=object)#[idx_to_keep]
session_ID = np.array(session_quest_data["ID"], dtype=object)#[idx_to_keep]
session_genre = np.array(session_stimu_data["genre"], dtype=object)#[idx_to_keep]
session_movie = np.array(session_stimu_data["movie"], dtype=object)#idx_to_keep]
session_session = np.array(session_stimu_data["session"])#[idx_to_keep]
session_hr = np.array(session_data["hr"], dtype=object)#[idx_to_keep]
session_EDR = np.array(session_data["EDR"], dtype=object)#[idx_to_keep]
session_ts = np.array(session_data["ts"], dtype=object)#[idx_to_keep]

assert len(session_EDA) == len(session_ID) == len(session_genre) == len(session_movie) == len(session_session) == len(session_hr) == len(session_EDR) == len(session_ts)

good_dt = len(idx_to_keep_sess)

# session before processing
#print("Session")
#print("Num users:", len(np.unique(session_ID)))
#print("Num movies:", len(np.unique(session_movie)))
total_time = 0
for sample in session_ts:
    total_time += (sample[-1] - sample[0])
print("total session time: ", total_time, total_time/3600)
print("After Processing")
# session after processing
#print("Num users:", len(np.unique(session_ID[idx_to_keep_sess])))
#print("Num movies:", len(np.unique(session_movie[idx_to_keep_sess])))
total_time = 0
for sample in session_ts[idx_to_keep_sess]:
    total_time += (sample[-1] - sample[0])
print("total session time: ", total_time, total_time/3600)

print("\n")
print("Segments")
# segments
#print("Num users:", len(np.unique(ID)))
#print("Num movies:", len(np.unique(movies)))
#print("Annotated segments: ", len(ar_seg), len(vl_seg), len(unc_seg))
# hours collected
# segment
total_time = 0
for sample in seg_data["ts"]:
    total_time += (sample[-1] - sample[0])
print("total seg time: ", total_time, total_time/3600)
print("After Processing")
print("Table ID: ", len(np.unique(session_ID)), "&", len(np.unique(ID)), "&", len(np.unique(session_ID[idx_to_keep_sess])), "&", len(np.unique(ID[idx_to_keep])))
print("Table Movie: ", len(np.unique(session_movie)), "&", len(np.unique(movies)), "&", len(np.unique(session_movie[idx_to_keep_sess])), "&", len(np.unique(movies[idx_to_keep])))
print("Table Session: ", len(np.unique(session_session)), "&", len(np.unique(session_seg)), "&", len(np.unique(session_session[idx_to_keep_sess])), "&", len(np.unique(session_seg[idx_to_keep])))
print("Table samples: ", len(session_session), "&", len(session_seg), "&", len(session_session[idx_to_keep_sess]), "&", len(session_seg[idx_to_keep]))

#print("Total number of samples after outliers removal (session): ", good_dt)
#print("len sessions data:", len(session_EDA))
#print("len sessions data (after noise removal):", len(session_EDA[idx_to_keep]))
#print("Num movies:", len(np.unique(session_movie)))

# remove outliers
raw_EDA_seg = raw_EDA_seg[idx_to_keep]
EDA_seg = EDA_seg[idx_to_keep]
ar_seg = ar_seg[idx_to_keep].astype(int)
unc_seg = unc_seg[idx_to_keep]
vl_seg = vl_seg[idx_to_keep].astype(int)
hr_seg = hr_seg[idx_to_keep]
movies = movies[idx_to_keep]
genres = genres[idx_to_keep]
ID = ID[idx_to_keep]
session_seg = session_seg[idx_to_keep]
devices = devices[idx_to_keep]

# segments
#print("Num users:", len(np.unique(ID)))
#print("Num movies:", len(np.unique(movies)))
#print("Annotated segments: ", len(ar_seg), len(vl_seg), len(unc_seg))
# hours collected
# segment
total_time = 0
for sample in np.array(seg_data["ts"])[idx_to_keep]:
    total_time += (sample[-1] - sample[0])
print("total seg time: ", total_time, total_time/3600)

users_counts = {int(value): list(ID).count(value) for value in set(ID)}

## user histogram
plt.figure(dpi=200)
plt.bar(list(users_counts.keys()), list(users_counts.values()), color ='#2a9d8f',
        width = 0.4)
plt.xlabel("Users")
plt.ylabel("No. of Annotations")
#plt.show()
plt.savefig("../6_Results/Analysis/users_hist.pdf", format="pdf", dpi=200, bbox_inches="tight")
## statistics on annotation distribution 
plt.close()

print("ar ann stats (table 7): ", np.round(np.mean(ar_seg), 2), "&", np.round(np.std(ar_seg),2), "&", np.round(stats.skew(ar_seg),2) , " &" , np.round(stats.kurtosis(ar_seg), 2))
print("vl ann stats (table 7 ): ", np.round(np.mean(vl_seg), 2), "&", np.round(np.std(vl_seg),2), "& ", np.round(stats.skew(vl_seg),2), " &" , np.round(stats.kurtosis(vl_seg), 2))
print("\n")

# histograma com genre
genre_counts = {value: list(genres).count(value) for value in set(genres)}

plt.figure(dpi=200)
plt.bar(list(genre_counts.keys()), list(genre_counts.values()), color ='#2a9d8f',
        width = 0.4)
plt.xlabel("Movie Genre")
plt.ylabel("No. of Annotations")
x = list(range(len(list(genre_counts.keys()))))
plt.xticks(x, labels=list(genre_counts.keys()), rotation = 45)
#plt.show()
plt.savefig("../6_Results/Analysis/genre_hist_all_samples.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

## arousal + valence histogram
ar_counts = {int(value): list(ar_seg).count(value) for value in set(ar_seg)}
vl_counts = {int(value): list(vl_seg).count(value) for value in set(vl_seg)}

plt.figure(dpi=200)
plt.bar(list(ar_counts.keys()), list(ar_counts.values()), color ='#2a9d8f',
        width = 0.4)
plt.xlabel("Arousal")
plt.ylabel("No. of Annotations")
#plt.show()
plt.savefig("../6_Results/Analysis/ar_hist.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()


plt.figure(dpi=200)
plt.bar(list(vl_counts.keys()), list(vl_counts.values()), width = 0.4, color ='#2a9d8f')
plt.xlabel("Valence")
plt.ylabel("No. of Annotations")
#plt.show()
plt.savefig("../6_Results/Analysis/vl_hist.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close("all")


## Annotation distribution stat-tests

# statistical tests
# arousal - EDA/HR

# normalise EDA data per subject
EDA_seg = np.nan_to_num(EDA_seg)
session_EDA = np.nan_to_num(session_EDA)
session_hr = np.nan_to_num(session_hr)

norm_EDA_seg  = []
for seg in range(len(EDA_seg)):
    sess_idx_1 = np.where(movies[seg] == session_movie)[0]
    sess_idx_2 = np.where(ID[seg] == session_ID)[0]
    sess_idx = list(set(sess_idx_1) & set(sess_idx_2))

    assert set(sess_idx).issubset(sess_idx_1) 
    assert set(sess_idx).issubset(sess_idx_2) 
    
    sess_dt = np.hstack((session_EDA[sess_idx]))
    if sess_dt.std() == 0:
        norm_EDA_seg += [((EDA_seg[seg] - sess_dt.mean())).tolist()]
    else:
        norm_EDA_seg += [((EDA_seg[seg] - sess_dt.mean())/sess_dt.std()).tolist()]
norm_EDA_seg = np.nan_to_num(norm_EDA_seg)

assert len(norm_EDA_seg) == len(EDA_seg)

violin_dt = []
for ann_sc in list(ar_counts.keys()):
    sc_idx = np.where(ar_seg == ann_sc)[0]
    violin_dt += [np.mean(norm_EDA_seg[sc_idx], axis=1)]


## mean EDA
# arousal
plt.figure(dpi=200)
#plt.title(str(stats.kruskal(violin_dt[0], violin_dt[1], violin_dt[2], violin_dt[3], violin_dt[4])))
#plt.title(str(stats.f_oneway(violin_dt)))
violin = plt.violinplot(violin_dt, list(ar_counts.keys()), showmedians=True, showextrema=False)
change_vio_color(violin)
plt.xlabel("Arousal")
plt.ylabel(r'$\mu$ Stand. (p/ subj. sess.) EDA')
#plt.show()
plt.savefig("../6_Results/Analysis/ar_violin_meanEDA.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

print("ar_violin_meanEDA: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")
# valence
violin_dt = []
for ann_sc in list(vl_counts.keys()):
    sc_idx = np.where(vl_seg == ann_sc)[0]
    violin_dt += [np.mean(norm_EDA_seg[sc_idx], axis=1)]

plt.figure(dpi=200)
violin = plt.violinplot(violin_dt, list(ar_counts.keys()), showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(violin_dt[0], violin_dt[1], violin_dt[2], violin_dt[3], violin_dt[4])))
plt.xlabel("Valence")
plt.ylabel(r'$\mu$ Stand. (p/ subj. sess.) EDA')
plt.tight_layout()
#plt.show()
plt.savefig("../6_Results/Analysis/vl_violin_meanEDA.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

print("vl_violin_meanEDA: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)

if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")
# genre
# normalize data per user
users_ID = np.unique(session_ID)
us_dt = {}
for _id in users_ID:
    us_idx = np.where(session_ID == _id)[0]
    us_dt[_id] = {}
    us_dt[_id]["mean"] = np.hstack((session_EDA[us_idx])).mean()
    us_dt[_id]["std"] = np.hstack((session_EDA[us_idx])).std()

norm_data, norm_genre = [], []
for seg in range(len(EDA_seg)):
    us = ID[seg]
    if us_dt[us]["std"] == 0:
        norm_data += [(EDA_seg[seg] - us_dt[us]["mean"]).tolist()]
    else:
        norm_data += [(EDA_seg[seg] - us_dt[us]["mean"])/us_dt[us]["std"]]
    norm_genre += [genres[seg]]
norm_data = np.nan_to_num(norm_data)

genre_counts_n = {value: norm_genre.count(value) for value in set(norm_genre)}
violin_dt = []
for ann_sc in list(set(genre_counts_n.keys())):
    sc_idx = np.where(genres == ann_sc)[0]
    violin_dt += [np.mean(norm_data[sc_idx], axis=1)]
x = list(range(len(list(genre_counts_n.keys()))))
plt.figure(dpi=200)
violin = plt.violinplot(violin_dt, x, showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(*violin_dt)))
plt.xlabel("Movie Genre")
plt.ylabel(r'$\mu$ Stand. (p/ subj. sess.) EDA')
plt.xticks(x, labels=list(genre_counts_n.keys()), rotation = 45)
#plt.show()
plt.savefig("../6_Results/Analysis/genre_violin_meanEDA.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()
print("genre_violin_meanEDA: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")

## mean HR
# norm hr data
norm_hr_seg  = []
for seg in range(len(hr_seg)):
    sess_idx_1 = np.where(movies[seg] == session_movie)[0]
    sess_idx_2 = np.where(ID[seg] == session_ID)[0]
    sess_idx = list(set(sess_idx_1) & set(sess_idx_2))
    
    assert set(sess_idx).issubset(sess_idx_1) 
    assert set(sess_idx).issubset(sess_idx_2) 

    sess_dt = np.hstack((session_hr[sess_idx]))
    if sess_dt.mean() == 0:
        norm_hr_seg += [hr_seg[seg]]
    else:
        norm_hr_seg += [hr_seg[seg]/sess_dt.mean()]
norm_hr_seg = np.array(norm_hr_seg, dtype=object)
norm_hr_seg = np.nan_to_num(norm_hr_seg)

assert len(norm_hr_seg) == len(EDA_seg)

# Arousal
violin_dt, ar_k = [], []
for ann_sc in list(ar_counts.keys()):
    sc_idx = np.where(ar_seg == ann_sc)[0]
    _dt = []
    for ann_i in range(len(sc_idx)):
        if len(norm_hr_seg[sc_idx][ann_i]) > 0:
            _dt += [np.mean(norm_hr_seg[sc_idx][ann_i])]
    violin_dt += [_dt]
    ar_k += [ann_sc]

plt.figure(dpi=200)
violin = plt.violinplot(violin_dt, ar_k, showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(violin_dt[0], violin_dt[1], violin_dt[2], violin_dt[3], violin_dt[4])))
plt.xlabel("Arousal")
plt.ylabel(r'$\mu$ Norm. (p/ subj. sess.) HR')
#plt.show()
plt.savefig("../6_Results/Analysis/ar_violin_meanHR.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()
print("ar_violin_meanHR: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")


# valence
violin_dt, vl_k = [], []
for ann_sc in list(vl_counts.keys()):
    sc_idx = np.where(vl_seg == ann_sc)[0]
    _dt = []
    for ann_i in range(len(sc_idx)):
        if len(norm_hr_seg[sc_idx][ann_i]) > 0:
            _dt += [np.mean(norm_hr_seg[sc_idx][ann_i])]
    violin_dt += [_dt]
    vl_k += [ann_sc]

plt.figure(dpi=200)
violin = plt.violinplot(violin_dt, vl_k, showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(violin_dt[0], violin_dt[1], violin_dt[2], violin_dt[3], violin_dt[4])))
plt.xlabel("Valence")
plt.ylabel(r'$\mu$ Norm. (p/ subj. sess.) HR')
#plt.show()
plt.savefig("../6_Results/Analysis/vl_violin_meanHR.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()
print("vl_violin_meanHR: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")

# genre
# normalize data per user
assert len(session_ID) == len(session_hr)
users_ID = np.unique(session_ID)
for _id in users_ID:
    us_idx = np.where(session_ID == _id)[0]
    if len(np.hstack((session_hr[us_idx]))) > 0:
        us_dt[_id]["mean"] = np.hstack((session_hr[us_idx])).mean()
    else:
        us_dt[_id]["mean"] = 1
norm_data, norm_genre = [], []
for seg in range(len(norm_hr_seg)):
    us = ID[seg]
    if us_dt[us]["mean"] == 0:
        norm_data += [hr_seg[seg]]
    else:
        norm_data += [hr_seg[seg]/us_dt[us]["mean"]]
    norm_genre += [genres[seg]]
norm_data = np.array(norm_data, dtype=object)
norm_data = np.nan_to_num(norm_data)

violin_dt, ar_k = [], []
for ann_sc in list(genre_counts.keys()):
    sc_idx = np.where(genres == ann_sc)[0]
    _dt = []
    for ann_i in range(len(sc_idx)):
        if len(norm_data[sc_idx][ann_i]) > 0:
            _dt += [np.mean(norm_data[sc_idx][ann_i])]
    violin_dt += [_dt]
    ar_k += [ann_sc]
x = list(range(len(list(genre_counts.keys()))))

plt.figure(dpi=200)
violin = plt.violinplot(violin_dt, list(range(len(ar_k))), showmedians=True, showextrema=False)
change_vio_color(violin)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(*violin_dt)))
plt.xlabel("Movie Genre")
plt.ylabel(r'$\mu$ Norm. (p/ subj. sess.) HR')
plt.xticks(x, labels=list(genre_counts.keys()), rotation = 45)
#plt.show()
plt.savefig("../6_Results/Analysis/genre_violin_meanHR.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

print("genre_violin_meanHR: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")
## genre

## movie genre and annotations
violin_dt = []
for ann_sc in list(genre_counts.keys()):
    sc_idx = np.where(genres == ann_sc)[0]
    violin_dt += [ar_seg[sc_idx]]

plt.figure(dpi=200)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(*violin_dt)))
violin = plt.violinplot(violin_dt, x, showmedians=True, showextrema=False)
change_vio_color(violin)
plt.xlabel("Movie Genre")
plt.ylabel("Arousal")
x = list(range(len(list(genre_counts.keys()))))
plt.xticks(x, labels=list(genre_counts.keys()), rotation = 45)
#plt.show()
plt.savefig("../6_Results/Analysis/genre_violin_arousal.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()

print("genre_violin_arousal: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")
violin_dt = []
for ann_sc in list(genre_counts.keys()):
    sc_idx = np.where(genres == ann_sc)[0]
    violin_dt += [vl_seg[sc_idx].tolist()]
plt.figure(dpi=200)
#plt.title(str(stats.f_oneway(violin_dt)))
#plt.title(str(stats.kruskal(*violin_dt)))
x = list(range(len(list(genre_counts.keys()))))
violin = plt.violinplot(violin_dt, x, showmedians=True, showextrema=False)
change_vio_color(violin)
plt.xlabel("Movie Genre")
plt.ylabel("Valence")
plt.xticks(x, labels=list(genre_counts.keys()), rotation = 45)
#plt.show()
plt.savefig("../6_Results/Analysis/genre_violin_valence.pdf", format="pdf", dpi=200, bbox_inches="tight")
plt.close()
print("genre_violin_valence: ", end="")
norm_t = []
for i in range(len(violin_dt)):
    norm_t += [np.round(stats.shapiro(violin_dt[i])[1], 2)]
print("(" + str(norm_t) + ") ", end="")
norm_t = np.array(norm_t)
if len(np.where(norm_t < 0.05)[0]) > 0:
    print(" & ", str(np.round(stats.kruskal(*violin_dt)[1], 2)), end="")
else:
    print(" & ", str(np.round(stats.f_oneway(*violin_dt)[1], 2)), end="")
print(")")

pdb.set_trace()
# > 0.05 -> normal distribution