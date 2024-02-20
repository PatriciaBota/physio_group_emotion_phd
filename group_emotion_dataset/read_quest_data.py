# -*- coding: utf-8 -*-
"""
This file reads the users demographic data. 

Returns: 
    A csv file the user info ans its ID
    
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
import json
import pdb

# local
import biosppy as bp


DIR = glob.glob("../2_Questionnaire/Raw/*.xlsx")[0]  # replace with directory where the file is


df_samples = pd.read_excel(DIR, engine='openpyxl')

matrix = df_samples.values

emails = matrix[:, 4].astype(str)
age = matrix[:, 5]
gender = matrix[:, 6]
device = matrix[:, 7].astype(str)
movie = matrix[:, 21]
unusable = matrix[:, 23]
date_acq = matrix[:, 0]

print(emails)

pers_matrix = pd.read_excel(open(DIR, 'rb'), sheet_name='Personality')
persn_sess_date = pers_matrix['Unnamed: 0'].values
extra_perc = pers_matrix['Unnamed: 54'].values
neuro_perc = pers_matrix['Unnamed: 56'].values
open_perc = pers_matrix['Unnamed: 58'].values
consc_perc = pers_matrix['Unnamed: 60'].values
agreeb_perc = pers_matrix['Unnamed: 62'].values
pers_ID = pers_matrix['ID'].values


# create database with users email, s01, device, movie and ID
last_email = 'FILL@mail.com'  # CHANGE TO LAST USER EMAIL
start_email = "FILL@gmail.com"  # CHANGE TO FIRST USER EMAIL
size_end = np.where(emails == last_email)[0][0] + 1
size_st = np.where(emails == start_email)[0][0]
print("last email", emails[size_end])
print("start email", emails[size_st])
A_quest = matrix[:, 14].astype(str)
C_quest = matrix[:, 16].astype(str)

session_c, subj_ID_c, table, data, unusable_c = 0, 0, [], {}, 0
_dt = []
for l_ds in range(size_st, size_end, 1):
    print(l_ds)
    if emails[l_ds] == 'nan':
        print("change session")
        session_c += 1
        print("continue", emails[l_ds])
        continue

    if device[l_ds] == "nan" or A_quest[l_ds] == "nan"  or C_quest[l_ds] == "nan": #if unusable[l_ds] == "x":
        unusable_c += 1
        print("Unusable user: ", unusable_c, device[l_ds], A_quest[l_ds], C_quest[l_ds])
        continue
    if l_ds > size_st: 
        if emails[l_ds] in np.array(_dt)[:, 1]:
            print("Repeated subject")
            subj_ID_idx = np.where(emails[l_ds] == np.array(_dt)[:, 1])[0][0]
            subj_ID = _dt[subj_ID_idx][0]
        else:
            subj_ID = subj_ID_c
            subj_ID_c += 1
    else:
        subj_ID = subj_ID_c
        subj_ID_c += 1
    
    data = {}
    data["ID"] = subj_ID
    #data["email"] = emails[l_ds]
    data["movie"] = movie[l_ds] 
    data["device"] = int(float(device[l_ds]))
    data["session"] = session_c
    data["age"] = matrix[l_ds, 5]
    data["gender"] = matrix[l_ds, 6]
    data["friends"] = matrix[l_ds, 9]
    data["pre-viewing"] = [matrix[l_ds, 10], matrix[l_ds, 11]]
    data["pos-viewing"] = [matrix[l_ds, 12], matrix[l_ds, 13]]
    idx_sess_pers_m = np.where(date_acq[l_ds] == persn_sess_date)[0]
    if len(idx_sess_pers_m) > 0:
        idx_dv_sess_pers_i = np.where(data['device'] == pers_ID[idx_sess_pers_m])[0]
        if len(idx_dv_sess_pers_i) > 0:
            e_p = extra_perc[idx_sess_pers_m][idx_dv_sess_pers_i][0]
            n_p = neuro_perc[idx_sess_pers_m][idx_dv_sess_pers_i][0]
            o_p = open_perc[idx_sess_pers_m][idx_dv_sess_pers_i][0]
            c_p = consc_perc[idx_sess_pers_m][idx_dv_sess_pers_i][0]
            a_p = agreeb_perc[idx_sess_pers_m][idx_dv_sess_pers_i][0]
            pers = [e_p, n_p, o_p, c_p, a_p]
        else:
            pers = [None]*5
    else:
        pers = [None]*5
    data["personality"] = pers

    _dt += [[subj_ID, emails[l_ds]]]
    table += [data]

print("Unusable users counter: ", unusable_c)
print("Usable users counter: ", len(table))


label = ["ID", "movie", "device", "session", "age", "gender", "friends", "pre-viewing", "pos-viewing", "personality"] 
df = pd.DataFrame.from_dict(table) 
df.to_csv ('../2_Questionnaire/Raw/quest_raw_data.csv', index = True, header=label)


with open('../2_Questionnaire/Raw/quest_raw_data.json', 'w') as file_object:  #open the file in write mode
    json.dump(table, file_object, indent=4)


label = ["movie_ID", "movie_name", "duration", "bit_rate", "fps", "encoding", "video_resolution_width", "video_resolution_height", "genre"]
f = open("../1_Stimuli/Raw/video_info.json")
video_info = json.load(f)
df = pd.DataFrame.from_dict(video_info) 
df.to_csv ('../1_Stimuli/Raw/video_info.csv', index=True, header=label)
