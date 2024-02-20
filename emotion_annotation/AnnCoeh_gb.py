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
sb.set(font_scale=1.5)


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
    #print(fileDir)
    FS = int(f[ID].attrs["sampling rate"])        
    if f[ID]['EDA'].shape[1] == 2: # FMCI DEV saved in hdf5 file
        x, y = FMCI_fileProc(f[ID]['EDA'])
        #y = eda.get_filt_eda(y, FS, 10)
        #y, _ = tools.smoother(signal=y, kernel='bartlett', size=int(20 * FS), mirror=True)  
    else:
        x = f[ID]['EDA'][:, -1]*0.001
        if f[ID]['EDA'].shape[1] == 14:  # RIOT
            y = f[ID]['EDA'][:, 5].ravel()  # EDA on A0
            #y = clean_dataset(np.array(y))
            #y = eda.get_filt_eda(y, FS, 10)
    x -= x[0]  # synchronize with movie
    
    nidx = np.unique(x, return_index=True)[1]
    x = x[nidx]
    y = y[nidx]
    
    interpf = interp1d(x, y, kind='cubic')  # resample
    FS = 32  # new SR            
    xnew = np.arange(x[0], x[-1], 1/FS)  # all samples with same SR
    y = interpf(xnew)
    x = xnew
    y = clean_dataset(np.array(y))
    #plt.figure()
    #plt.plot(x, y, label="interpolated")
    y = tools.filter_signal(signal=y, ftype="FIR", band='lowpass', order=3, frequency=2, sampling_rate=FS)[0]
    
    #plt.plot(dx, dy, label="derivative")
    #plt.legend()
    #plt.show()
    # overlap window
    #wind = 10*FS  # s
    #overlp = 5*FS  #s
    #windy = [np.mean(dy[i:i+wind]) for i in range(0, len(dy)-wind, overlp)]
    #windx = [np.mean(dx[i:i+wind]) for i in range(0, len(dx)-wind, overlp)]

    # normalization so area is 1
    #indvInt = np.trapz(windy)
    #windy = windy/indvInt.ravel()
    
    #plt.figure()
    #plt.plot(windx, windy, label="avg")

    #thIDX = np.argwhere(windy >= 0.66*np.max(windy)).ravel()
    #plt.hlines(0.66*np.max(windy), 0, windx[-1])
    #windy = windy[thIDX]
    #windx = np.array(windx)[thIDX]

    #plt.plot(x, y/np.trapz(y), label="filtered")
    #plt.plot(windx, windy, "o", c="g", label="th")
    #plt.legend()
    #plt.show() 
    
    return x, y, FS


def getTM_idx(data, TH):
    for _i, t in enumerate(data):
        if t >= TH:
            return _i


def getusers_EDA(MOVIEACQ):
    filesDIR = glob.glob("../DATA/"+MOVIEACQ+"/*.hdf5")
    all_users_x, all_users_y, all_u_FS, all_u_int, xmax, xmin = [], [], [], [], [], []
    for file in filesDIR:
        f = h5py.File(file, 'r')
        for ID in f.keys():
            #print("FILE", file, " (ID ", ID, ")")
            if ID == "LOGs" or ID == "Flags":  # not an ID
                continue
            x, y, FS = getEDA(file, ID)  # get data
            #x = x - x[0]
            xmax += [x[-1]]
            xmin += [x[0]]

            all_users_y += [y]
            all_users_x += [x]
            all_u_FS += [FS]
    
    ce = np.min(xmax)
    cs = np.max(xmin)

    _all_users_y, _all_users_x, _all_u_FS, confSc = all_users_y[:], all_users_x[:], all_u_FS[:], []
    for i in range(len(all_users_y)):
        _e = getTM_idx(all_users_x[i], ce)
        _s = getTM_idx(all_users_x[i], cs)
        if not _s:
            _s = 1
        if not _e:
            _e = len(all_users_x[i])-2
        
        interpf = interp1d(all_users_x[i][_s-1:_e+1], all_users_y[i][_s-1:_e+1], kind='cubic')  # resample
        FS = 50  # new SR   
        
        xnew = np.arange(cs, ce, 1/FS)  # all samples with same SR
        _all_users_y[i] = interpf(xnew)
        _all_users_x[i] = xnew
        _all_u_FS[i] = FS

        _all_users_y[i] = np.diff(_all_users_y[i])
        posIDX = np.argwhere(_all_users_y[i] < 0)
        _all_users_y[i][posIDX] = 0

        # overlap window
        wind = 10*FS  # s
        overlp = 5*FS  #s
        
        _all_users_y[i] = [np.mean(_all_users_y[i][k:k+wind]) for k in range(0, len(_all_users_y[i])-wind, wind-overlp)]
        _all_users_x[i] = [np.mean(_all_users_x[i][k:k+wind]) for k in range(0, len(_all_users_x[i])-wind, wind-overlp)]

        # normalization so area is 1
        indvInt = np.trapz(_all_users_y[i])
        _all_users_y[i] = _all_users_y[i]/indvInt.ravel()
        
    ## Remove outliers
    aggmt = []
    for user1 in range(len(_all_users_y)):
        aggmt_i = []
        for user2 in range(len(_all_users_y)):
            if user1 == user2:
                continue
            aggmt_i += [spearmanr(_all_users_y[user1], _all_users_y[user2])[0]]
        aggmt += [np.sum(aggmt_i)]
    aggmt = np.array(aggmt)/(len(_all_users_y)-1)
    
    usersTRmt = []
    for user1 in range(len(_all_users_y)):
        if aggmt[user1] >= (np.mean(aggmt) - (0.5*np.std(aggmt))):
            usersTRmt += [user1]
    
    print("len us INIT", len(_all_users_y))
   
    _all_users_y = np.array(_all_users_y)[usersTRmt]
    _all_users_x = np.array(_all_users_x)[usersTRmt]
    all_users_y = np.array(all_users_y)[usersTRmt]
    all_users_x = np.array(all_users_x)[usersTRmt]
    print("len us", len(_all_users_y))
    #avgInt = np.sum(all_u_int)/len(all_u_int)
    #avgRec = 1/len(all_u_int) * np.sum(_all_users_y, axis=0)
    
    for i in range(len(_all_users_y)):
        confSc += [tools.pearson_correlation(_all_users_y[i], 1/len(_all_users_y)*np.sum(_all_users_y, axis=0))[0]]
    
    confSc = np.array(confSc).reshape(-1, 1)
    _all_users_y = np.array(_all_users_y)
    wMP = np.sum(confSc*_all_users_y, axis=0)
    wMP = wMP/np.sum(confSc)
  
    #plt.show()
    
    #plt.figure(dpi=300)
    #plt.plot(_all_users_x[0], wMP, label="avg")
    T = 0.
   
    wMP = normEDA(wMP)  # NORM
    #plt.plot(_all_users_x[0], wMP, c="#66ff00", label="Weighted mean GSR profile")
    
    thIDX = np.argwhere(wMP >= T*np.max(wMP)).ravel()
   # plt.hlines(T*np.max(wMP), 0, xnew[-1])
    wMP = np.array(wMP)[thIDX]
    
   # plt.plot(np.array(_all_users_x[0])[thIDX], wMP, "o", color="b", label="th")
   # plt.legend()
#  #  plt.show() 
   # plt.savefig("PLOTS/GSTProfile_TH_BBB" + ".pdf", format="pdf", bbox_inches='tight')
    
    x_time = []
    for i in range(len(_all_users_x)):
        x_time += [_all_users_x[i][thIDX]]
    
    return np.array(x_time), _all_users_y, np.array(all_u_FS), wMP, T


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


def stdNorm(sig):
    sig = np.array(sig)
    return (sig - np.mean(sig))/(np.std(sig))

def UNC_getSTAT(ALG, d, MOVIEACQ):
    print("ALG: ", ALG, d)
    vl, u_st, u_end = [], [], []
    filesDIR = glob.glob("../DATA/" +MOVIEACQ+"/*.hdf5")
    WIDNT = 80
    print("WIDNT", WIDNT)    
    for file in filesDIR:
        print(file)
        f = h5py.File(file, 'r')
        for ID in f.keys():
            dC = 0
            if ID == "LOGs" or ID == "Flags":  # not an ID
                continue

            v_stTM = f[ID][d + " " + ALG][:][:, 1]
            v_endTM = f[ID][d + " " + ALG][:][:, 2]
            #if ALG == "Movie":
            v_endTM = v_stTM + WIDNT  # change to 5
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
            
            for idx_v, annT in enumerate(v_stTM):  # correct start-end times
                if v_endTM[idx_v] <= v_stTM[idx_v]:
                    print("ERROR", file, ID, v_stTM[idx_v], v_endTM[idx_v], ALG, d)
                    v_endTM[idx_v] = v_stTM[idx_v] + WIDNT
                         
            vl += [_v]
            u_st += [v_stTM]
            u_end += [v_endTM]
    return vl, u_st, u_end

ALG = ["Movie", "Random", "EDA"]
DIM = ["Arousal", "Valence"]

fDIC = {}
EDAfDIC = {}
fstats_DIC = {}
fDIC["Dimension"] = []
EDAfDIC["Dimension"] = []
fstats_DIC["Dimension"] = []

EDAfDIC["DTW"], EDAfDIC["RMSE"], EDAfDIC["Spearman C"], EDAfDIC["Krippendorff Interval"], EDAfDIC["p value"] = [], [], [], [], []


#EDAfDIC["Krippendorff Nominal"], EDAfDIC["Krippendorff Ordinal"], EDAfDIC["Krippendorff Interval"] = [], [], []
EDA_fDIC = {} 
#EDA_fDIC["Dimension"] = {}
#EDA_fDIC["RMSE"], EDA_fDIC["DTW"], EDA_fDIC["p value"], EDA_fDIC["Pearson C"], EDA_fDIC["Spearman C"] = [], [], [], [], []

for d in DIM:
    
    _MOVIEACQ = ["After_The_Rain", "Elephant_s_Dream", "Tears_of_Steel"]    
    #_MOVIEACQ = ["Elephant_s_Dream"]
    for MOVIEACQ in _MOVIEACQ: 
        alg_ann, alg_t = [], []
        for lg in ALG:     
            u_ann, u_GT, u_t = [], [], []        
            
            vl, st, end = UNC_getSTAT(lg, d, MOVIEACQ)  # get volunteers self-report
            valAnn = np.loadtxt("../annotations/" + MOVIEACQ + "_" + d + ".txt",  skiprows=1)
            _tL = valAnn[:, 0].astype(np.int)  # ground truth time
            _labels = valAnn[:, 1]  # ground truth labels  # WRONG BY USER 
            
            _labels = np.array(unNorm(_labels))
            
            emtAnnt, emtAnnV = [], []
            for user in range(len(vl)):
                _d, _a = [], []
                for annIDX in range(len(vl[user])):
                    _d += [np.arange(st[user][annIDX], end[user][annIDX]+1).astype(np.int)]
                    _a += [[vl[user][annIDX]]*len(_d[-1])]
                emtAnnt += [_d]
                emtAnnV += [_a]        
             
            # find simultaneuous time between (emtAnnt, all_users_x)
            for eda_tIDX in range(len(_tL)):  # eda dots time
                # find all annotations for that instant
                idx_ann = []
                for userIDX in range(len(emtAnnt)):  # iterate over user
                    for annIDX in range(len(emtAnnt[userIDX])):  # annotation
                        if (_tL[eda_tIDX] >= emtAnnt[userIDX][annIDX][0]) and (_tL[eda_tIDX] <= (emtAnnt[userIDX][annIDX][-1])):  # simultaneous ann
                            idx_ann += [emtAnnV[userIDX][annIDX][0]]  # add ann
                if len(idx_ann) > 0:
                    u_ann += [np.mean(idx_ann)]  # users avg annotation
                    u_GT += [_labels[eda_tIDX]]
                    u_t += [_tL[eda_tIDX]] 
                else:
                    u_ann += [np.nan]  # users avg annotation
                    u_GT += [np.nan]
                    u_t += [np.nan]
            alg_ann += [u_ann]
            alg_t += [u_t]
       
       # Remove nans
       
        nanIDX = np.argwhere(np.isnan(alg_ann[0]))
        rem = np.arange(len(alg_ann[0]))
        mtn = np.delete(rem, nanIDX)
        
        alg_ann[0] = np.array(alg_ann[0])[mtn] 
        alg_ann[1] = np.array(alg_ann[1])[mtn] 
        alg_ann[2] = np.array(alg_ann[2])[mtn]
        alg_t[0] = np.array(alg_t[0])[mtn] 
        alg_t[1] = np.array(alg_t[1])[mtn] 
        alg_t[2] = np.array(alg_t[2])[mtn]
        print("0", len(alg_ann[0]), len(alg_ann[0]), len(alg_ann[0]))
     
        nanIDX = np.argwhere(np.isnan(alg_ann[1]))
        rem = np.arange(len(alg_ann[1]))
        mtn = np.delete(rem, nanIDX)
        
        alg_ann[0] = np.array(alg_ann[0])[mtn] 
        alg_ann[1] = np.array(alg_ann[1])[mtn] 
        alg_ann[2] = np.array(alg_ann[2])[mtn]
        alg_t[0] = np.array(alg_t[0])[mtn] 
        alg_t[1] = np.array(alg_t[1])[mtn] 
        alg_t[2] = np.array(alg_t[2])[mtn]

        nanIDX = np.argwhere(np.isnan(alg_ann[2]))
        rem = np.arange(len(alg_ann[1]))
        mtn = np.delete(rem, nanIDX)
        
        alg_ann[0] = np.array(alg_ann[0])[mtn] 
        alg_ann[1] = np.array(alg_ann[1])[mtn] 
        alg_ann[2] = np.array(alg_ann[2])[mtn]
        alg_t[0] = np.array(alg_t[0])[mtn] 
        alg_t[1] = np.array(alg_t[1])[mtn] 
        alg_t[2] = np.array(alg_t[2])[mtn]

        my_dpi = 300
        plt.figure(figsize=(1800/my_dpi, 1200/my_dpi), dpi=300)
        if lg == "Movie":
            lg = "Scene"
        plt.title(d + " (" + MOVIEACQ.replace("_", " ") + ")")
        #plt.plot(alg_t[0], alg_ann[0], label="Scene", c="b")
        #plt.plot(alg_t[1], alg_ann[1], label=ALG[1], c="r")
        #plt.plot(alg_t[2], alg_ann[2], label=ALG[2], c="g")

        plt.plot(alg_t[0], alg_ann[0], "v", c="b", label="Scene")
        plt.plot(alg_t[1], alg_ann[1],  "v", c="r", label=ALG[1])
        plt.plot(alg_t[2], alg_ann[2], "v", c="g", label=ALG[2])

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (a.u.)")

        plt.legend()
        plt.ylim(1, 5)
        plt.savefig("../PLOTS/" + MOVIEACQ + "/AnnAvgCoeh" + "_" + d + ".pdf", format="pdf", bbox_inches='tight', dpi=300)
        
        alg_ann[0] = stdNorm(alg_ann[0])
        alg_ann[1] = stdNorm(alg_ann[1])
        alg_ann[2] = stdNorm(alg_ann[2])
        
        comb = combinations(np.arange(len(alg_ann)), 2)
        for i in list(comb):    
            if d + " " + ALG[i[0]] + "-" + ALG[i[1]] not in list(EDA_fDIC.keys()):
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]] = {}
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["RMSE"] = []
                #EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Pearson C"] = []
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Spearman C"] = []
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["DTW"] = []
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["p value"] = []
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Krippendorff Interval"] = []
            
            EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["RMSE"] += [np.round(mean_squared_error(alg_ann[i[0]], alg_ann[i[1]], squared=False), 2)] 
            #EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Pearson C"] += [np.round(np.nan_to_num(tools.pearson_correlation(alg_ann[i[0]], alg_ann[i[1]])[0]), 2)]
                
            if shapiro(alg_ann[i[0]])[0] < 0.05 or shapiro(alg_ann[i[1]])[0] < 0.05:
                print(shapiro(alg_ann[i[0]])[0], shapiro(alg_ann[i[1]])[0], "Wilcoxon signed-rank")                    
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["p value"] += [np.round(np.nan_to_num(wilcoxon(alg_ann[i[0]], alg_ann[i[1]])[1]), 4)]
            else:
                print(shapiro(alg_ann[i[0]])[0], shapiro(alg_ann[i[1]])[0], "weltch t test")        
                EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["p value"] += [np.round(np.nan_to_num(ttest_ind(alg_ann[i[0]], alg_ann[i[1]], equal_var=False)[1]), 4)]  # equal population means 

            EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Spearman C"] += [np.round(np.nan_to_num(spearmanr(alg_ann[i[0]], alg_ann[i[1]])[0]), 2)]
            EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["DTW"] += [np.round(np.nan_to_num(dtw.dtw(stdNorm(alg_ann[i[0]]), stdNorm(alg_ann[i[1]]))[0]), 2)]
            
            dt = np.array([alg_ann[i[0]], alg_ann[i[1]]])        
            EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]]["Krippendorff Interval"] += [np.round(krippendorff.alpha(reliability_data=dt, level_of_measurement='interval'), 2)]
    
    comb = combinations(np.arange(len(alg_ann)), 2)
    for i in list(comb): 
        EDAfDIC["Dimension"] += [d + " " + ALG[i[0]] + "-" + ALG[i[1]]]        
        for k in EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]].keys():
            EDAfDIC[k] += [str(np.round(np.mean(EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]][k]), 2)) + " +- " + str(np.round(np.std(EDA_fDIC[d + " " + ALG[i[0]] + "-" + ALG[i[1]]][k]), 2))]

print(pd.DataFrame(EDAfDIC).to_latex(index=False))


