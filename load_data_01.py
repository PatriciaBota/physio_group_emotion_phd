# Import libraries
from __future__ import absolute_import, division, print_function
import biosppy as bs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, AffinityPropagation, \
    DBSCAN, Birch
from sklearn.mixture import GaussianMixture
import seaborn as sb
import pickle
import numba
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import minmax_scale
from scipy.stats.mstats import gmean
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from sklearn.model_selection import train_test_split
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [14, 7]
sb.set()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import glob
import h5py
import pyhrv
import os
sep = os.sep
from scipy import stats
import mne
from xml.dom import minidom
from scipy import signal

def load_joana():
    directories = np.sort(np.array(glob.glob("/Users/patriciabota/DATA/Joana/Par*.hdf5")))  # EDA fingers / EDA hand
    ignore_users = [1, 6, 8, 15, 18]  # 14
    _labels = np.loadtxt("input_Joana" + sep + 'user_ann_200819.txt')
    # users = np.arange(0, 23, 1)
    # users = np.delete(users, [1, 6, 8, 14, 15, 18])
    # _labels = _labels[users]

    arousal = _labels[:, 1::2]
    valence = _labels[:, ::2]
    # arousal = valence

    SR = 1000  # sampling rate
    windows_time = 40
    print("Windows time: ", windows_time)
    wind_size = windows_time * SR
    SENSOR = "BVP"
    # GT_labels = [0, 1, 1, ]

    users_df, labels_a, labels_v = [], [], []
    for i, user in enumerate(directories):
        if i in ignore_users:
            continue
        print(user)
        user_l_a, user_l_v, user_data = [], [], []
        # Open file
        f = h5py.File(user, 'r')
        for _i, p in enumerate(list(f.keys())):  # section of the protocol
            if p != 'Calibration1' and p != 'Calibration2' and p != 'Calibration3' and p != 'Calibration4':
                print(p)
                for sensor in list(f[p]['1'].keys()):
                    if SENSOR in sensor: # and "hand" not in sensor:
                        # Get data
                        data = np.nan_to_num(f[p]['1'][sensor].value.astype(np.float))
                        sensor = str(sensor).replace(' ', '_')

                        print('sensor: ', sensor)

                        if SENSOR == "ECG":
                            data = np.nan_to_num(bs.signals.ecg.get_filt_ecg(data, SR))
                        elif SENSOR == "EDA":
                            data = np.nan_to_num(bs.signals.eda.get_filt_eda(data, SR))
                        elif SENSOR == "BVP":
                            data = np.nan_to_num(bs.signals.bvp.get_filt_bvp(data, SR))
                        elif SENSOR == "Resp":
                            data = np.nan_to_num(bs.signals.resp.get_filt_resp(data, SR))

                        # cut = len(data) // 20
                        # data = data[cut:-cut]
                        # Segment data
                        SEG = 0
                        if SEG:
                            rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
                            # extract templates
                            _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                                        rpeaks=rpeaks,
                                                                        sampling_rate=SR,
                                                                        before=0.2,
                                                                        after=0.4)
                        else:
                            _seg = [data[i:i + wind_size] for i in
                                    range(0, len(data) - wind_size, int(wind_size * 0.25))]
                        if not len(_seg):
                            continue
                            # _seg = [data.reshape(1, -1)]

                        _c = len(_seg)
                        if p == 'Video1':
                            _sec = 0
                        elif p == 'Video2':
                            _sec = 1
                        elif p == 'Video3':
                            _sec = 2
                        elif p == 'Video4':
                            _sec = 3
                        elif p == 'Video5':
                            _sec = 4
                        elif p == 'Video6':
                            _sec = 5
                        elif p == 'Video7':
                            _sec = 6
                        user_l_a += [int(arousal[i][_sec])] * _c
                        user_l_v += [int(valence[i][_sec])] * _c

                        # if int(arousal[i][_sec]) <= 5:
                        #     user_l_a += [0] * _c
                        # else:
                        #     user_l_a += [1] * _c
                        #
                        # if int(valence[i][_sec]) <= 5:
                        #     user_l_v += [0] * _c
                        # else:
                        #     user_l_v += [1] * _c

                        _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)

                        print("Modality: " + SENSOR + str(_ECG_f.values.shape))
                        a

                        print("!", _ECG_f.values.shape)
                        if not len(user_data):
                            user_data = _ECG_f.values
                        else:
                            user_data = np.vstack((user_data, _ECG_f.values))

        users_df += [user_data]
        labels_a += [user_l_a]
        labels_v += [user_l_v]

    for i in range(len(users_df)):
        print(len(labels_a[i]))
        print(len(users_df[i]))

    labels_a = np.array(labels_a)
    labels_v = np.array(labels_v)
    users_df = np.array(users_df)

    pickle.dump(users_df, open("input_Joana" + sep + str(SENSOR) + 'fingers_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'wb'))
    pickle.dump(labels_a, open("input_Joana" + sep + str(SENSOR) +'fingers_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal', 'wb'))
    pickle.dump(labels_v, open("input_Joana" + sep + str(SENSOR) +'fingers_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'wb'))
    pickle.dump(np.array(_ECG_f.columns), open("input_Joana" + sep + str(SENSOR) + 'fingers_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels', 'wb'))


def load_WESAD():
    print("WESAD DATABASE")
    SENSOR = "Resp"
    print("SENSOR: ", SENSOR)
    dir = "/Users/patriciabota/DATA/WESAD/"
    windows_time = 40
    print("Windows time: ", windows_time)
    # wind_size = windows_time * SR

    all_files = np.sort(os.listdir(dir))
    print(all_files)

    users_df, labels_a, labels_v = [], [], []
    for d in all_files:
        if d == ".DS_Store":
            continue
        else:
            de = d[0] + d[2] if d[1] == '0' else d
            print('Extracting data from ' + str(d))
            e = pickle.load(open(dir + d + sep + de + '.pkl', 'rb'), encoding='bytes')

            if SENSOR == "ECG":
                SR = 700
                data = e[b'signal'][b'chest'][b'ECG'].ravel()
                data = np.nan_to_num(bs.signals.ecg.get_filt_ecg(data, SR))
            elif SENSOR == "EDA":
                SR = 4
                data = e[b'signal'][b'wrist'][b'EDA'].ravel()
                data = np.nan_to_num(bs.signals.eda.get_filt_eda(data, SR))
            elif SENSOR == "BVP":
                SR = 64
                data = e[b'signal'][b'wrist'][b'BVP'].ravel()
                data = np.nan_to_num(bs.signals.bvp.get_filt_bvp(data, SR))
            elif SENSOR == "Resp":
                SR = 700
                data = e[b'signal'][b'chest'][b'Resp'].ravel()
                data = np.nan_to_num(bs.signals.resp.get_filt_resp(data, SR))

            wind_size = windows_time*SR

            # cut = len(data) // 20
            # data = data[cut:-cut]
            #
            # cut = len(e[b'label']) // 20
            # e[b'label'] = e[b'label'][cut:-cut]

            # m = min(len(e[b'label']), len(data))
            # e[b'label'] = e[b'label'][:m]
            # data = data[:m]

            # Segment data
            SEG = 0
            if SEG:
                if SENSOR == "EDA":
                    data = signal.resample(data, len(data)*25)
                    SR = 100
                if SENSOR == "Resp":
                    data = signal.resample(data, len(data)//7)
                    SR = 100
                rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
                # extract templates
                _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                                      rpeaks=rpeaks,
                                                                      sampling_rate=SR,
                                                                      before=0.2,
                                                                      after=0.4)
            else:
                _seg = [data[i:i + wind_size] for i in
                        range(0, len(data) - wind_size, int(wind_size * 0.25))]

            wind_size = windows_time*700

            try:
                user_l_v = [stats.mode(e[b'label'][i:i + wind_size])[0][0] for i in
                        range(0, len(e[b'label']) - wind_size, int(wind_size * 0.25))]
            except:
                user_l_v = [stats.mode(e[b'label'][i:i + wind_size])[0][0][0] for i in
                        range(0, len(e[b'label']) - wind_size, int(wind_size * 0.25))]

            m = min(len(_seg), len(user_l_v))
            user_l_v = user_l_v[:m]
            _seg = _seg[:m]

            _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)

        users_df += [_ECG_f.values]
        labels_v += [user_l_v]

        print(len(labels_v[-1]))
        print(len(users_df[-1]))

    for i in range(len(users_df)):
        print(len(labels_v[i]))
        print(len(users_df[i]))

    # labels_a = np.array(labels_a)
    labels_v = np.array(labels_v)
    users_df = np.array(users_df)

    pickle.dump(users_df, open("input_WESAD" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'wb'))
    # pickle.dump(labels_a, open("input_Joana" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal', 'wb'))
    pickle.dump(labels_v, open("input_WESAD" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels', 'wb'))
    pickle.dump(np.array(_ECG_f.columns), open("input_WESAD" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels', 'wb'))


def load_DEAP():
    print("DEAP")
    windows_time = 40
    print("Windows time: ", windows_time)
    SR = 128
    wind_size = windows_time * SR

    SENSOR = "BVP"
    if SENSOR == "EDA":
        sensor_idx = 37
    if SENSOR == "Resp":
        sensor_idx = 38
    if SENSOR == "BVP":
        sensor_idx = 39

    # annotations = pd.read_csv('/Users/patriciabota/DATA/DEAP/metadata_csv/participant_ratings.csv')
    dirs = np.sort(os.listdir("/Users/patriciabota/DATA/DEAP/data_preprocessed_python/"))
    users_labels_v, users_labels_a, users_df, labels_v, labels_a = [], [], [], [], []
    for u, user in enumerate(dirs):
        users_labels_v = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['labels'][:, 0]
        users_labels_a = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['labels'][:, 1]
        _user_data = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['data']
        user_l_v, user_l_a, ECG_f = [], [], []
        for v, video in enumerate(_user_data):
            if SENSOR == "EDA":
                data = np.nan_to_num(bs.signals.eda.get_filt_eda(video[sensor_idx], SR))
            elif SENSOR == "BVP":
                data = np.nan_to_num(bs.signals.bvp.get_filt_bvp(video[sensor_idx], SR))
            elif SENSOR == "Resp":
                data = np.nan_to_num(bs.signals.resp.get_filt_resp(video[sensor_idx], SR))

            SEG = 0
            if SEG:
                rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
                # extract templates
                _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                            rpeaks=rpeaks,
                                                            sampling_rate=SR,
                                                            before=0.2,
                                                            after=0.4)
            else:
                _seg = [data[i:i + wind_size] for i in
                        range(0, len(data) - wind_size, int(wind_size * 0.25))]

            _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)

            user_l_v += [users_labels_v[v]]*len(_seg)
            user_l_a += [users_labels_a[v]]*len(_seg)

            if not len(ECG_f):
                ECG_f = _ECG_f.values
            else:
                ECG_f = np.vstack((ECG_f, _ECG_f.values))

        users_df += [ECG_f]
        labels_v += [user_l_v]
        labels_a += [user_l_a]
        print(len(user_l_a))
        print(ECG_f.shape)

    labels_a = np.array(labels_a)
    labels_v = np.array(labels_v)
    users_df = np.array(users_df)

    pickle.dump(users_df, open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'wb'))
    pickle.dump(labels_a, open("input_DEAP" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal', 'wb'))
    pickle.dump(labels_v, open("input_DEAP" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'wb'))
    pickle.dump(np.array(_ECG_f.columns), open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels', 'wb'))


def load_DEAP_vo():
    print("DEAP")
    windows_time = 40
    print("Windows time: ", windows_time)
    SR = 128
    wind_size = windows_time * SR

    SENSOR = "BVP"
    if SENSOR == "EDA":
        sensor_idx = 37
    if SENSOR == "Resp":
        sensor_idx = 38
    if SENSOR == "BVP":
        sensor_idx = 39

    # annotations = pd.read_csv('/Users/patriciabota/DATA/DEAP/metadata_csv/participant_ratings.csv')
    dirs = np.sort(os.listdir("/Users/patriciabota/DATA/DEAP/data_preprocessed_python/"))
    users_labels_v, users_labels_a, users_df, labels_v, labels_a = [], [], [], [], []
    for u, user in enumerate(dirs):
        users_labels_v = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['labels'][:, 0]
        users_labels_a = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['labels'][:, 1]
        _user_data = pickle.load(open('/Users/patriciabota/DATA/DEAP/data_preprocessed_python/' + str(user), 'rb'), encoding='iso-8859-1')['data']

        for v, video in enumerate(_user_data):
            user_l_v, user_l_a, ECG_f = [], [], []

            if SENSOR == "EDA":
                data = np.nan_to_num(bs.signals.eda.get_filt_eda(video[sensor_idx], SR))
            elif SENSOR == "BVP":
                data = np.nan_to_num(bs.signals.bvp.get_filt_bvp(video[sensor_idx], SR))
            elif SENSOR == "Resp":
                data = np.nan_to_num(bs.signals.resp.get_filt_resp(video[sensor_idx], SR))

            SEG = 0
            if SEG:
                rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
                # extract templates
                _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                            rpeaks=rpeaks,
                                                            sampling_rate=SR,
                                                            before=0.2,
                                                            after=0.4)
            else:
                _seg = [data[i:i + wind_size] for i in
                        range(0, len(data) - wind_size, int(wind_size * 0.25))]

            _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)

            user_l_v += [users_labels_v[v]]*len(_seg)
            user_l_a += [users_labels_a[v]]*len(_seg)

            # if not len(ECG_f):
            #     ECG_f = _ECG_f.values
            # else:
            #     ECG_f = np.vstack((ECG_f, _ECG_f.values))

            users_df += [_ECG_f.values]
            labels_v += [user_l_v]
            labels_a += [user_l_a]
            print(len(user_l_a))
            print(_ECG_f.values.shape)

    labels_a = np.array(labels_a)
    labels_v = np.array(labels_v)
    users_df = np.array(users_df)

    pickle.dump(users_df, open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data_v', 'wb'))
    pickle.dump(labels_a, open("input_DEAP" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal_v', 'wb'))
    pickle.dump(labels_v, open("input_DEAP" + sep + str(SENSOR) +'_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence_v', 'wb'))
    pickle.dump(np.array(_ECG_f.columns), open("input_DEAP" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels_v', 'wb'))


def load_HCI():
    print("HCI")
    dirs = np.sort(os.listdir("/Users/patriciabota/DATA/HCI/Sessions/"))[1:]
    SENSOR = "ECG"
    if SENSOR == "EDA":
        sensor_idx = 41
    if SENSOR == "Resp":
        sensor_idx = 45
    if SENSOR == "ECG":
        sensor_idx = 31
    windows_time = 5
    print("Windows time: ", windows_time)
    SR = 256
    wind_size = windows_time * SR
    users_labels_v, users_labels_a, users_df, labels_v, labels_a = [], [], [], [], []
    subj_id = 0
    all_users_ids = []
    user_l_v, user_l_a, ECG_f = [], [], []
    # for i in range(29):
    #     user_l_v.append([])
    #     user_l_a.append([])
    #     ECG_f.append([])
    for d, dir in enumerate(dirs):
        try:
            mydoc = minidom.parse(glob.glob("/Users/patriciabota/DATA/HCI/Sessions/" + str(dir) + "/*.xml")[0])
            items = mydoc.getElementsByTagName('subject')
            curr_user_id = int(items[0].attributes['id'].value)
            if subj_id != curr_user_id:
                all_users_ids += [curr_user_id]
        except:
            continue

    for i in range(len(np.unique(all_users_ids))):
        user_l_v.append([])
        user_l_a.append([])
        ECG_f.append([])

    for d, dir in enumerate(dirs):
        try:
            data = mne.io.read_raw_edf(glob.glob("/Users/patriciabota/DATA/HCI/Sessions/" + str(dir) + "/*.bdf")[0])
            # parse an xml file by name
            mydoc = minidom.parse(glob.glob("/Users/patriciabota/DATA/HCI/Sessions/" + str(dir) + "/*.xml")[0])
            data = data.get_data()
            # info = data.info
        except:
            continue
        try:
            items = mydoc.getElementsByTagName('session')
            arousal = int(items[0].attributes['feltArsl'].value)
            valence = int(items[0].attributes['feltVlnc'].value)
            items = mydoc.getElementsByTagName('subject')
            curr_user_id = int(items[0].attributes['id'].value)
        except:
            continue
        offset = 30*SR
        data = data[sensor_idx][offset:-offset]

        SEG = 0
        if SEG:
            rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
            # extract templates
            _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                        rpeaks=rpeaks,
                                                        sampling_rate=SR,
                                                        before=0.2,
                                                        after=0.4)
        else:
            _seg = [data[i:i + wind_size] for i in
                    range(0, len(data) - wind_size, int(wind_size * 0.25))]

        if not len(_seg):
            _seg = [data]
        _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)
        try:
            if not len(ECG_f[curr_user_id-1]):
                ECG_f[curr_user_id-1] = _ECG_f.values
            else:
                ECG_f[curr_user_id-1] = np.vstack((ECG_f[curr_user_id-1], _ECG_f.values))

            user_l_v[curr_user_id - 1] += [valence] * len(_seg)
            user_l_a[curr_user_id - 1] += [arousal] * len(_seg)
            # if subj_id != curr_user_id:
            # #     print(">>>>>", len(user_l_a))
            # #     print(">>>>>", ECG_f.shape)
            # #
            # #     users_df += [ECG_f]
            # #     labels_v += [user_l_v]
            # #     labels_a += [user_l_a]
            # #     subj_id = curr_user_id
            # #     user_l_v, user_l_a, ECG_f = [], [], []
            #     all_users_ids += [curr_user_id]
        except:
            continue

    labels_a = np.array(user_l_a)
    labels_v = np.array(user_l_v)
    users_df = np.array(ECG_f)

    print("all_users_ids: ", np.unique(all_users_ids))

    pickle.dump(users_df,
                open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data_2', 'wb'))
    pickle.dump(labels_a,
                open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal_2',
                     'wb'))
    pickle.dump(labels_v,
                open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence_2', 'wb'))
    pickle.dump(np.array(_ECG_f.columns),
                open("input_HCI" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels_2', 'wb'))

def load_eight():
    print("EIGHT EMOTIONS")

    import scipy.io

    print("EIGHT DATABASE")
    SENSOR = "Resp"
    print("SENSOR: ", SENSOR)
    dir = "/Users/patriciabota/DATA/EightEmotionSenticsData/MAS622dataSetA/*.mat"

    dirs = np.sort(glob.glob(dir))
    print(dirs)

    arousal = [0, 1, 1, 0, 1, 0, 1, 0]
    valence = [1, 1, 1, 1, 0, 0, 0, 0]

    SENSOR = "BVP"
    if SENSOR == "Resp":
        sensor_idx = [0,1,2,3,4,5,6,7]
    if SENSOR == "EDA":
        sensor_idx = [8,9,10,11,12,13,14,15]
    if SENSOR == "BVP":
        sensor_idx = [16,17,18,19,21,22,23,24]

    windows_time = 40
    print("Windows time: ", windows_time)
    SR = 256
    wind_size = windows_time * SR
    users_labels_v, users_labels_a, users_df, labels_v, labels_a = [], [], [], [], []

    for d, dir in enumerate(dirs):
        print("DIR: ", dir)
        user_l_v, user_l_a, ECG_f = [], [], []
        mat = scipy.io.loadmat(dir)
        for c, col in enumerate(sensor_idx):
            data = mat[[*mat][0]][:, col]
            # a_lab += [arousal[c]]
            # v_lab += [valence[c]]

            SEG = 0
            if SEG:
                rpeaks = bs.signals.ecg.get_rpks(np.array(data), SR)
                # extract templates
                _seg, _ = bs.signals.ecg.extract_heartbeats(signal=np.array(data),
                                                            rpeaks=rpeaks,
                                                            sampling_rate=SR,
                                                            before=0.2,
                                                            after=0.4)
            else:
                _seg = [data[i:i + wind_size] for i in
                        range(0, len(data) - wind_size, int(wind_size * 0.25))]

            if not len(_seg):
                _seg = [data]

            _ECG_f, _ = bs.features.feature_vector.get_feat(_seg, SENSOR, SR, segment=False)

            user_l_v += [valence[c]] * len(_seg)
            user_l_a += [arousal[c]] * len(_seg)

            if not len(ECG_f):
                ECG_f = _ECG_f.values
            else:
                ECG_f = np.vstack((ECG_f, _ECG_f.values))

        users_df += [ECG_f]
        labels_v += [user_l_v]
        labels_a += [user_l_a]
        print(len(user_l_a))
        print(_ECG_f.values.shape)

    labels_a = np.array(labels_a)
    labels_v = np.array(labels_v)
    users_df = np.array(users_df)

    pickle.dump(users_df,
                open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_data', 'wb'))
    pickle.dump(labels_a,
                open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_arousal',
                     'wb'))
    pickle.dump(labels_v,
                open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_users_labels_valence', 'wb'))
    pickle.dump(np.array(_ECG_f.columns),
                open("input_eight" + sep + str(SENSOR) + '_wo_cut2c_' + str(windows_time) + 's_25ol_features_labels', 'wb'))


load_joana()
