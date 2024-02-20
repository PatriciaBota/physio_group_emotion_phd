import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from sklearn.preprocessing import minmax_scale
import scipy
from scipy.interpolate import interp1d
import pandas as pd
from biosppy.signals import eda
import datetime
from biosppy import tools
import seaborn as sb
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import krippendorff
from sklearn.metrics import mean_squared_error
import biosppy as bp
from scipy.stats import pearsonr, spearmanr, ttest_ind, wilcoxon, shapiro, ttest_rel
from statsmodels.stats import weightstats as stests
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from itertools import combinations
import dtw
import pdb



def clean_dataset(data):
    # d = {str(lab): data[:, idx] for idx, lab in enumerate(labels_name)}
    try:
        df = pd.DataFrame(data=data)
        df = df.replace([np.inf, -np.inf, np.nan], 0.0)
        df = df.values.ravel()
    except:
        df = []
    return df


def FMCI_fileProc(data):
    return np.array(data)[:, 0], np.array(data)[:, 1]


def normEDA(sig):
    #sig = np.append(sig, [0])
    sig = minmax_scale(sig, (1, 5))
    return sig.tolist()


def getEDA(fileDir, ID):
    f = h5py.File(fileDir, 'r')
    FS = int(f[ID].attrs["sampling rate"])        
    if f[ID]['EDA'].shape[1] == 2: # FMCI DEV saved in hdf5 file
        x, y = FMCI_fileProc(f[ID]['EDA'])
        interpf = interp1d(x, y, kind='cubic')  # resample
        FS = 10  # new SR            
        xnew = np.arange(x[0], x[-1], 1/FS)  # all samples with same SR
        y = interpf(xnew)
        x = xnew
        y = clean_dataset(np.array(y))
        y = eda.get_filt_eda(y, FS, 10)
        y, _ = tools.smoother(signal=y, kernel='bartlett', size=int(20 * FS), mirror=True)  
    else:
        x = f[ID]['EDA'][:, -1]*0.001
        if f[ID]['EDA'].shape[1] == 14:  # RIOT
            y = f[ID]['EDA'][:, 5]  # EDA on A0
            y = clean_dataset(np.array(y))
            y = eda.get_filt_eda(y, FS, 10)

    x -= x[0]  # synchronize with movie
    return x, normEDA(y), FS


def getusers_EDA(MOVIEACQ):
    filesDIR = glob.glob("../DATA/" + MOVIEACQ + "/*.hdf5")
    all_users_x, all_users_y, all_u_FS = [], [], []
    for file in filesDIR:
        f = h5py.File(file, 'r')
        for ID in f.keys():
            #print("FILE", file, " (ID ", ID, ")")
            if ID == "LOGs" or ID == "Flags":  # not an ID
                continue
            x, y, FS = getEDA(file, ID)  # get data
            
            all_users_y += [y]
            all_users_x += [x]
            all_u_FS += [FS]
    return np.array(all_users_x), np.array(all_users_y), np.array(all_u_FS)


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def norm(sig):
    sig = np.append(sig, [1, 5])
    sig = minmax_scale(sig, (-1, 1))
    return sig[:-2].tolist()


def unNorm(sig):
    sig = np.append(sig, [-1, 1])
    sig = minmax_scale(sig, (1, 5))
    return sig[:-2].tolist()

def UNC_getSTAT(ALG, d, MOVIEACQ):
    print("ALG: ", ALG, d)
    vl, u_st, u_end, u_t = [], [], [], []
    filesDIR = glob.glob("../DATA/" + MOVIEACQ + "/*.hdf5")
    for file in filesDIR:
        f = h5py.File(file, 'r')
        for ID in f.keys():
            dC = 0
            if ID == "LOGs" or ID == "Flags":  # not an ID
                continue

            v_stTM = f[ID][d + " " + ALG][:][:, 1]
            v_endTM = f[ID][d + " " + ALG][:][:, 2]
            v_t = f[ID][d + " " + ALG][:][:, 0]

            v_endTM = v_stTM + 60  # change to 5
            #v_endTM = v_stTM + 5  # 5 in old TO REMOVE!!!! TODO!!!!!
            
            _v = f[ID][d  + " " + ALG][:][:, 3].astype(np.int)                 
            
            for v in np.unique(v_stTM):
                dupl = duplicates(v_stTM, v)  # find duplicates
                if len(dupl) > 1:
                    dC += len(dupl)-1
                    # if duplicate found delete last entry
                    for i_tr, IDXtoDel in enumerate(dupl[:-1]):
                        v_stTM = np.delete(v_stTM, IDXtoDel - i_tr)
                        _v = np.delete(_v, IDXtoDel - i_tr)
                        v_endTM = np.delete(v_endTM, IDXtoDel - i_tr)
                        v_t = np.delete(v_t, IDXtoDel - i_tr)

            for idx_v, annT in enumerate(v_stTM):  # correct start-end times
                if v_endTM[idx_v] <= v_stTM[idx_v]:
                    print("ERROR", file, ID, v_stTM[idx_v], v_endTM[idx_v], ALG, d)
                    v_endTM[idx_v] = v_stTM[idx_v] + 60
                         
            vl += [_v]
            u_st += [v_stTM]
            u_end += [v_endTM]
            u_t += [v_t]
    return vl, u_st, u_end, u_t


def getTM_idx(data, TH):
    for _i, t in enumerate(data):
        if t >= TH:
            return _i


ALG = ["Movie", "Random", "EDA"]
DIM = ["Arousal", "Valence"]

fDIC = {}
EDAfDIC = {}
fstats_DIC = {}
fDIC["Dimension"] = []
EDAfDIC["Dimension"] = []
fstats_DIC["Dimension"] = []

fstats_DIC["Ann. Time"], fstats_DIC["Num. Ann."], fstats_DIC["Total Anns."] = [], [], []
hist_DIC = {}
for d in DIM:
    for lg in ALG:     
        stats_DIC = {}
        stats_DIC["Ann. Time"], stats_DIC["Num. Ann."], stats_DIC["Total Anns."] = [], [], []
        
        fDIC["Dimension"] += [d + " " + lg] 
        EDAfDIC["Dimension"] += [d + " " + lg] 
        fstats_DIC["Dimension"] += [d + " " + lg] 
        #_MOVIEACQ = ["After_The_Rain", "Chatter", "Big_Buck_Bunny", "Lesson_Learned", "Elephant_s_Dream"]
        _MOVIEACQ = ["After_The_Rain", "Elephant_s_Dream", "Tears_Of_Steel"]
        u_GT, u_ann, u_eda, u_t, t_ann, t_ind, v_ann, ar_ann = [], [], [], [], 0, [], [], []    
        
        for MOVIEACQ in _MOVIEACQ: 
            valAnn = np.loadtxt("../annotations/" + MOVIEACQ + "_" + d + ".txt",  skiprows=1)
            _tL = valAnn[:, 0].astype(np.int)  # ground truth time
            _labels = valAnn[:, 1]  # ground truth labels  # WRONG BY USER     
            _labels = np.array(unNorm(_labels))
            
            vl, st, end, _t = UNC_getSTAT(lg, d, MOVIEACQ)  # get volunteers self-report
            
            all_users_x, all_users_y, all_u_FS = getusers_EDA(MOVIEACQ)
            
            for user_idx in range(len(vl)):  # iterate over users idx
                eda_seg, ann, GT_seg, _u_t = [], [], [], []                   
                for seg_idx in range(len(vl[user_idx])):  # iterate over ann segments
                    _st = st[user_idx][seg_idx]
                    _end = end[user_idx][seg_idx]
                    
                    if _st >= len(_labels):
                        continue
                    
                    _s = getTM_idx(all_users_x[user_idx], _st)
                    e = getTM_idx(all_users_x[user_idx], _end)

                    #eda_seg += [int(np.round(np.nan_to_num(np.mean(all_users_y[user_idx][_s:e]))))]  # ATTENTION 
                    eda_seg += [np.nan_to_num(np.mean(all_users_y[user_idx][_s:e]))]  # ATTENTION 
                    ann += [vl[user_idx][seg_idx]]

                    _s = getTM_idx(_tL, _st)
                    e = getTM_idx(_tL, _end)
                    
                    if _s == e:
                        e += 1
                    
                    GT_seg += [np.mean(_labels[_s:e])]  # ground-truth labels for segment
                    #_u_t = [_t[user_idx][-1]- _t[user_idx][0]]
                    _u_t = [np.nan_to_num(np.round(np.mean(np.diff(_t[user_idx]))), 2)]  # doesn't take consideration doubles

                u_GT += [GT_seg]
                u_ann += [len(ann)]
                u_eda += [eda_seg]
                u_t += [_u_t]
                t_ann += len(ann)
            
            stats_DIC["Num. Ann."] += [np.mean(u_ann)]
            stats_DIC["Ann. Time"] += [np.mean(u_t)]
            stats_DIC["Total Anns."] += [t_ann]     
            try:       
                hist_DIC[d + " Ann." + lg] += ann            
            except Exception as e:
                print(e)
                hist_DIC[d + " Ann." + lg] = ann
        for k in stats_DIC.keys():
            fstats_DIC[k] += [str(np.round(np.mean(stats_DIC[k]), 2)) + " +- " + str(np.round(np.std(stats_DIC[k]), 2))]


print("Annotations Statistics")
print(pd.DataFrame(fstats_DIC).to_latex(index=False))

for k in hist_DIC.keys():
    ar_counts = {int(value): hist_DIC[k].count(value) for value in set(hist_DIC[k])}
    plt.figure(dpi=200)
    plt.bar(list(ar_counts.keys()), list(ar_counts.values()), color ='#2a9d8f',
            width = 0.4)
    plt.xlabel(k)
    plt.ylabel("No. of Annotations")
    #plt.show()
    plt.savefig(k + "_ann_hist.pdf", format="pdf", dpi=200, bbox_inches="tight")
    plt.close()


