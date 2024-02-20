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
        y = eda.get_filt_eda(y, FS, 10)
        y, _ = tools.smoother(signal=y, kernel='bartlett', size=int(20 * FS), mirror=True)  
    else:
        x = f[ID]['EDA'][:, -1]*0.001
        if f[ID]['EDA'].shape[1] == 14:  # RIOT
            y = f[ID]['EDA'][:, 5].ravel()  # EDA on A0
            y = clean_dataset(np.array(y))
            y = eda.get_filt_eda(y, FS, 10)
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
    #y = tools.filter_signal(signal=y, ftype="FIR", band='lowpass', order=3, frequency=2, sampling_rate=FS)[0]
    
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
        
        # normalization so area is 1
        _all_users_y[i] =  normEDA(_all_users_y[i])       
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
     
    _all_users_y = np.array(_all_users_y)[usersTRmt]
    _all_users_x = np.array(_all_users_x)[usersTRmt]    
    
    avg = np.mean(_all_users_y, axis=0)

    return _all_users_x, _all_users_y, np.array(all_u_FS), avg


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
    return (sig - np.mean(sig))/(np.std(sig))

def UNC_getSTAT(ALG, d, MOVIEACQ):
    print("ALG: ", ALG, d)
    vl, u_st, u_end = [], [], []
    filesDIR = glob.glob("../DATA/" +MOVIEACQ+"/*.hdf5")
    for file in filesDIR:
        print(file)
        f = h5py.File(file, 'r')
        for ID in f.keys():
            dC = 0
            if ID == "LOGs" or ID == "Flags":  # not an ID
                continue

            v_stTM = f[ID][d + " " + ALG][:][:, 1]
            v_endTM = f[ID][d + " " + ALG][:][:, 2]
            WDIT = 60
            v_endTM = v_stTM + WDIT  # change to 5
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
                    v_endTM[idx_v] = v_stTM[idx_v] + WDIT
                         
            vl += [_v]
            u_st += [v_stTM]
            u_end += [v_endTM]
    return vl, u_st, u_end

ALG = ["Movie", "Random", "EDA"]
DIM = ["Arousal", "Valence"]

DIC = {}
fDIC = {}
fDIC["Dimension"]  = []
for d in DIM:
    for lg in ALG:     
        fDIC["Dimension"] += [d + " " + lg] 
        DIC["Num. Ann. Avg"] , DIC["Max Num. Ann."], DIC["Min. Num. Ann."] = [], [], []
        
        _MOVIEACQ = ["After_The_Rain", "Tears_Of_Steel", "Elephant_s_Dream"]
        #_MOVIEACQ = ["Elephant_s_Dream"]
        for MOVIEACQ in _MOVIEACQ:
            vl, st, end = UNC_getSTAT(lg, d, MOVIEACQ)  # get volunteers self-report
            valAnn = np.loadtxt("../annotations/" + MOVIEACQ + "_" + d + ".txt",  skiprows=1)
            _tL = valAnn[:, 0].astype(np.int)  # ground truth time
            _labels = valAnn[:, 1]  # ground truth labels  # WRONG BY USER 
            
            _labels = np.array(unNorm(_labels))
            
            all_users_x, all_users_y, all_u_FS, avg = getusers_EDA(MOVIEACQ)
            
            emtAnnt, emtAnnV = [], []
            for user in range(len(vl)):
                _d, _a = [], []
                for annIDX in range(len(vl[user])):
                    _d += [np.arange(st[user][annIDX], end[user][annIDX]+1).astype(np.int)]
                    _a += [[vl[user][annIDX]]*len(_d[-1])]
                emtAnnt += [_d]
                emtAnnV += [_a]        
             
            # find simultaneuous time between (emtAnnt, all_users_x)
            u_ann, u_EDA, u_t, N_ann = [], [], [], []
            for eda_tIDX in range(len(all_users_x[0])):  # eda dots time
                # find all annotations for that instant
                idx_ann = []
                for userIDX in range(len(emtAnnt)):  # iterate over user
                    for annIDX in range(len(emtAnnt[userIDX])):  # annotation
                        if (all_users_x[0][eda_tIDX] >= emtAnnt[userIDX][annIDX][0]) and (all_users_x[0][eda_tIDX] <= (emtAnnt[userIDX][annIDX][-1])):  # simultaneous ann
                            idx_ann += [emtAnnV[userIDX][annIDX][0]]  # add ann
                            break
                if len(idx_ann) > 0:
                    u_ann += [np.mean(idx_ann)]  # users avg annotation
                    N_ann += [len(idx_ann)]  # users avg annotation
                    u_EDA += [avg[eda_tIDX]]
                    u_t += [all_users_x[0][eda_tIDX]] 
            
            my_dpi = 300
            plt.figure(figsize=(1800/my_dpi, 1200/my_dpi), dpi=300)
  
            plt.title(lg + " " + d + " (" + MOVIEACQ.replace("_", " ") + ")")
            for user in range(len(vl)):    
                for j in range(len(emtAnnt[user])):  # annotaions
                    lst_d = emtAnnV[user][j] 
                    if not j and not user:
                        plt.scatter(emtAnnt[user][j], lst_d, color="k", alpha=0.1,label="Self-Report")
                    else:
                        plt.scatter(emtAnnt[user][j], lst_d, color="k", alpha=0.1)
            plt.plot(all_users_x[0], avg, label="EDA")
            plt.plot(u_t, u_EDA, "v", label="Sync. EDA")
            plt.plot(u_t, N_ann, ".", label="Number of Annotations")
            #plt.plot(u_t, u_ann, label="Self-Report Interpolation")
            plt.legend(bbox_to_anchor=[0.13, 1.14])
            #plt.show()
            plt.savefig("../PLOTS/" + MOVIEACQ + "/NumAnnByTm" + "_" +  lg + " " + d + ".pdf", format="pdf", bbox_inches='tight', dpi=300)
            plt.close('all')
        
            #DIC["Num. Ind."] += [len(vl)]
            DIC["Num. Ann. Avg"] += [np.mean(N_ann)]
            DIC["Max Num. Ann."] += [np.max(N_ann)]
            DIC["Min. Num. Ann."] += [np.min(N_ann)]
        
        for k in DIC.keys():
            try:
                fDIC[k]
            except:
                fDIC[k] = []
            fDIC[k] += [str(np.round(np.mean(DIC[k]), 2)) + " +- " + str(np.round(np.std(DIC[k]), 2))]
            #fDIC[k] += [str(np.round(np.mean(DIC[k]), 2))]

print(pd.DataFrame(fDIC).to_latex(index=False))



