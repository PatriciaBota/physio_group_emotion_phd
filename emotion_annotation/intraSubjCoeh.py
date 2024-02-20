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
import scipy.stats as stats

def multirater_kfree(n_ij, n, k):
    '''
    Computes Randolph's free marginal multirater kappa for assessing the 
    reliability of agreement between annotators.
    
    Args:
        n_ij: An N x k array of ratings, where n_ij[i][j] annotators 
              assigned case i to category j.
        n:    Number of raters.
        k:    Number of categories.
    Returns:
        Percentage of overall agreement and free-marginal kappa
    
    See also:
        http://justusrandolph.net/kappa/
    '''
    N = len(n_ij)
    
    P_e = 1./k
    P_O = (
        1./(N*n*(n-1))
        * 
        (sum(n_ij[i][j]**2 for i in range(N) for j in range(k)) - N*n)
    )
    
    kfree = (P_O - P_e)/(1 - P_e)
    
    return P_O, kfree


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
    sig = np.append(sig, [0])
    #sig = minmax_scale(sig, (0, 1))
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


def UNC_getSTAT(ALG, d, MOVIEACQ, file):
    f = h5py.File(file, 'r')
    for ID in f.keys():
        dC = 0
        if ID == "LOGs" or ID == "Flags":  # not an ID
            continue

        v_stTM = f[ID][d + " " + ALG][:][:, 1]
        v_endTM = f[ID][d + " " + ALG][:][:, 2]
        #if ALG == "Movie":
        DWITH = 80
        v_endTM = v_stTM + DWITH  # 5 in old
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
                v_endTM[idx_v] = v_stTM[idx_v] + DWITH
                     
    return _v, v_stTM, v_endTM



def getTM_idx(data, TH):
    for _i, t in enumerate(data):
        if t >= TH:
            return _i


def stdNorm(sig):
    return (sig - np.mean(sig))/(np.std(sig))


def contavgGT(MOVIEACQ, d):
    valAnn = np.loadtxt("../annotations/" + MOVIEACQ + "_" + d + ".txt",  skiprows=1)
    _tL = valAnn[:, 0]  # ground truth time
    _labels = valAnn[:, 1]  # ground truth labels  # WRONG BY USER 
    _labels = np.array(unNorm(_labels))
    return _tL, _labels


ALG = ["Movie", "Random", "EDA"]
DIM = ["Arousal", "Valence"]


EDAfDIC = {}
EDAfDIC["Dimension"] = []
for d in DIM:
    #_MOVIEACQ = ["After_The_Rain", "Chatter", "Big_Buck_Bunny", "Lesson_Learned", "Elephant_s_Dream"]
    _MOVIEACQ = ["After_The_Rain", "Elephant_s_Dream", "Tears_of_Steel"]
    
    EDA_fDIC = {}         
    EDA_fDIC["Eval. Error"], EDA_fDIC["STD Dev"], EDA_fDIC["d Opt."],EDA_fDIC["d Max."] = [], [], [], []


    for MOVIEACQ in _MOVIEACQ:   
        filesDIR = glob.glob("../DATA/" + MOVIEACQ + "/*.hdf5")
        
        _tL, _labels = contavgGT(MOVIEACQ, d)

        dstd, d_op, d_max, u_ti, s_ann_t, s_ann_v = [], [], [], [], [], []
        for file in filesDIR:  # iterate over users
            u_ann, u_st, u_end = [], [], []        
            for lg in ALG:    
                
                vl, st, end = UNC_getSTAT(lg, d, MOVIEACQ, file)  # get volunteers self-report
                u_ann += [vl]  # users annotation for 3 methods
                u_st += [st]
                u_end += [end]

            emtAnnt_M, emtAnnV_M = [], []
            user = 0
            _d, _a = [], []
            for annIDX in range(len(u_ann[user])):
                _d += [np.arange(u_st[user][annIDX], u_end[user][annIDX]+1).astype(np.int)]
                _a += [[u_ann[user][annIDX]]*len(_d[-1])]
            emtAnnt_M = _d
            emtAnnV_M = _a  
            
            emtAnnt_R, emtAnnV_R = [], []
            user = 1
            _d, _a = [], []
            for annIDX in range(len(u_ann[user])):
                _d += [np.arange(u_st[user][annIDX], u_end[user][annIDX]+1).astype(np.int)]
                _a += [[u_ann[user][annIDX]]*len(_d[-1])]
            emtAnnt_R = _d
            emtAnnV_R = _a  
   
            emtAnnt_E, emtAnnV_E = [], []
            user = 2
                
            _d, _a = [], []
            for annIDX in range(len(u_ann[user])):
                _d += [np.arange(u_st[user][annIDX], u_end[user][annIDX]+1).astype(np.int)]
                _a += [[u_ann[user][annIDX]]*len(_d[-1])]
            emtAnnt_E = _d
            emtAnnV_E = _a  

            for eda_tIDX in range(len(_tL)):  # eda dots time
                s_ann, s_ann_s, s_ann_e = [], [], []  # REMOVE for intra-subj
                for i in range(3):  # 3 algorithms
                    # see if any moment between users ann is simultaneous
                    for ann1 in range(len(u_st[i])):  # user 0 ann
                        if (_tL[eda_tIDX] >= u_st[i][ann1]) and (_tL[eda_tIDX] <= u_end[i][ann1]):  # simultaneous ann
                            s_ann += [u_ann[i][ann1]]
                            #s_ann_t += [np.arange(u_st[i][ann1], u_end[i][ann1], 1)]
                            #s_ann_v += [[u_ann[i][ann1]]*len(s_ann_t[-1])]
                            s_ann_s += [u_st[i][ann1]]
                            s_ann_e += [u_end[i][ann1]]
            
                if len(s_ann) > 1:
                    sr = np.mean(s_ann)
                    dstd += [np.sqrt(np.sum((s_ann-sr)**2)/(len(s_ann)-1))]
                    if len(s_ann) % 2 == 0: # even
                        lk = len(s_ann)
                    else:
                        lk = len(s_ann) + 1
                    d_op += [0.5*np.sqrt(lk/(lk-1))]
                    d_max += [2*np.sqrt(lk/(lk-1))]
                    u_ti += [_tL[eda_tIDX]]
                                
 
           # my_dpi = 300
           # plt.figure(figsize=(1800/my_dpi, 1200/my_dpi), dpi=300)
           # #if lg == "Movie":
           # #    lg = "Scene"

           # plt.title(d + " (" + MOVIEACQ.replace("_", " ") + ")")
           # plt.plot(u_ti, dstd, "*", label="Standard deviation", c="k")   
           # plt.axhline(np.mean(d_op), xmin=0, xmax=_tL[-1], ls = "-.", label="Upper bound", c="r")
           # plt.axhline(np.mean(dstd), xmin=0, xmax=_tL[-1], label="STD Average", c="b")
           # plt.axhline(np.mean(d_max), xmin=0, xmax=_tL[-1], ls="--", label="Maximum lower bound", c="orange")
           # 
           # for _v in range(len(emtAnnt_E)):
           #     if not _v:
           #         plt.scatter(emtAnnt_E[_v], emtAnnV_E[_v], label="EDA")
           #     else:
           #         plt.scatter(emtAnnt_E[_v], emtAnnV_E[_v])
           # for _v in range(len(emtAnnt_R)):
           # 
           #     if not _v:
           #         plt.scatter(emtAnnt_R[_v], emtAnnV_R[_v], label="Random")
           #     else:
           #         plt.scatter(emtAnnt_R[_v], emtAnnV_R[_v])
           # for _v in range(len(emtAnnt_M)):
           # 
           #     if not _v:
           #         plt.scatter(emtAnnt_M[_v], emtAnnV_M[_v], label="Scene")
           #     else:
           #         plt.scatter(emtAnnt_M[_v], emtAnnV_M[_v])

           # 
           # plt.legend(bbox_to_anchor=[0.13, 1.14])

           # plt.xlabel("Time (s)")
           # plt.ylabel("STD (a.u.)")
            #plt.show()           
            # plt.savefig("../PLOTS/" + MOVIEACQ + "/stdSubjAgreem" + "_" +  lg + " " + d + ".pdf", format="pdf", bbox_inches='tight', dpi=300)
           # plt.close('all')

            #if len(s_ann) < 1:
            #    continue
            
            EDA_fDIC["STD Dev"] += [np.round(np.nan_to_num(np.mean(dstd)), 2)]
            if (np.mean(dstd) - np.mean(d_op)) < 0:
                EDA_fDIC["Eval. Error"] += [0]
            else:
                EDA_fDIC["Eval. Error"] += [np.round(np.nan_to_num(np.mean(dstd) - np.mean(d_op)), 2)]

            EDA_fDIC["d Opt."] += [np.round(np.mean(d_op), 2)]
            EDA_fDIC["d Max."] += [np.round(np.mean(d_max), 2)]
    EDAfDIC["Dimension"] += [d]
    for k in EDA_fDIC.keys():
        if k not in EDAfDIC.keys():
            EDAfDIC[k] = []
        EDAfDIC[k] += [str(np.round(np.mean(EDA_fDIC[k]), 2)) + " +- " + str(np.round(np.std(EDA_fDIC[k]), 2))]

print("User's annotation correlation to GT")
print(pd.DataFrame(EDAfDIC).to_latex(index=False))


            
            


