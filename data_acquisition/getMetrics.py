import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale

NUMDEV = "20dev"
C = "2021530_31724_DATA"
samp_rate = 60
DIR = "DATA/" + NUMDEV + "/" + str(samp_rate) + "/" + C   # estou diretamente  a emitir

print("DIR", DIR)
f = h5py.File(DIR + ".hdf5", 'r')

lis_taxa, lis_ID, lis_avg_dtp, lis_std_dtp = [], [], [], []
time_sorted, loss = [], []  # check if time is received: 0 - out of order; 1 - in order
lis_mean1, lis_std1, lis_max1, lis_min1 = [], [], [], [] 
realSR, minSort, maxSort = [], [], []
outSortP = []
CorrectSP, OutOrdP, SupTSP, lis_med1, DIFSP = [], [], [], [], []
my_dpi = 100
completeAcqTime, lossPUns = [], []
lis_std, lis_max, lis_min, lis_med, lis_mean = [], [], [], [], []
lis_stdS, lis_maxS, lis_minS, lis_medS, lis_meanS = [], [], [], [], []
PNstd, PNmax, PNmin, PNmed, PNmean, PNNoErrPerc, resetC = [], [], [], [], [], [], []
meanresetT, maxresetT, minresetT, medresetT, stdresetT = [], [], [], [], []

def autolabel(rects):  # add label with percentage
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 2)
        #if height == 0:
        #    continue
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points", rotation=90,
                    ha='center', va='bottom')
maxtm = []
for ID in list(f.keys()):  # iterate over devices
    print("ID", ID)
    samp_rate = int(f[ID].attrs['sampling rate'])  # get sampling rate-ATTENTION: confirm it its right/same as said in file name

    samp_period = 1/samp_rate
    if samp_rate == 70:
        samp_period = 0.014
    if samp_rate == 60:
        samp_period = 0.016

    time = f[ID]['EDA'][:, -1]*0.001   # get time in s
    dif = np.round(np.diff(time), 3)  # get 1st der

    packetNbm = f[ID]['EDA'][:, -2]

    timeSorted = np.sort(time)   # sorted time
    
    intervalo_tempo = 0
    intSR, ntimeSorted, packetNbm, difpacketNbm = [], [], [], []
    for acq in correctAcq:
        #intSR += [round(len(timeSorted[acq[0]:acq[-1]])/_dt, 3)]

        ntimeSorted += np.sort(time[acq]).tolist()   # sorted time
    maxtm += [ntimeSorted[-1]]
    print(ID, maxtm[-1])

ALLDEVACQ = np.min(maxtm)

for ID in list(f.keys()):  # iterate over devices
    print("ID", ID)
    samp_rate = int(f[ID].attrs['sampling rate'])  # get sampling rate-ATTENTION: confirm it its right/same as said in file name

    samp_period = 1/samp_rate
    if samp_rate == 70:
        samp_period = 0.014
    if samp_rate == 60:
        samp_period = 0.016

    print("Sampling Period", samp_period)
    print("Sampling Rate", samp_rate)

    time = f[ID]['EDA'][:, -1]*0.001   # get time in s
    dif = np.round(np.diff(time), 3)  # get 1st der
    plt.figure()
    plt.plot(np.arange(len(time)), time, ".")
    plt.savefig("Plots/" + NUMDEV + "/" + str(samp_rate) + "/" + str(C) + "_DEV" + str(ID) + "_time_" + str(samp_period)+"_" + NUMDEV + ".pdf", format="pdf", bbox_inches='tight')
    #plt.show()
    plt.close('all') 
    packetNbm = f[ID]['EDA'][:, -2]
    correctAcq, _temp = [], []
    for i, d in enumerate(list(np.diff(packetNbm))):
        if not i:  
            _temp = [i]
        if d != 1:
            if len(_temp) > 0:
                correctAcq += [_temp] 
                _temp = []                
        else:
            if len(_temp) == 0:
                _temp = [i]                
            _temp += [i+1]
    print("Numb acq", len(correctAcq))

    timeSorted = np.sort(time)   # sorted time
    
    intervalo_tempo = 0
    intSR, ntimeSorted, packetNbm, difpacketNbm, ntime = [], [], [], [], []
    for acq in correctAcq:
        #intSR += [round(len(timeSorted[acq[0]:acq[-1]])/_dt, 3)]

        ntimeSorted += np.sort(time[acq]).tolist()   # sorted time
        ntime += time[acq].tolist()
        intervalo_tempo = ntimeSorted[-1] - ntimeSorted[0]  # get acqusition duration

        packetNbm = f[ID]['EDA'][acq][:, -2]   # get packet number
        packetNbm += packetNbm[np.argsort(time[acq])].astype(int).tolist() # sort packet number using time
    
        difpacketNbm += list(np.diff(packetNbm))  # dif packet number, ideal should be 0 and - 2**16-1
        if ALLDEVACQ in acq:
            _idx = acq.index(ALLDEVACQ)
            _dt = timeSorted[acq[:_idx]] - timeSorted[acq[0]]
        else:   
            _dt = timeSorted[acq[-1]] - timeSorted[acq[0]]
        intervalo_tempo += _dt   # get acqusition duration
    timeSorted = np.array(ntimeSorted)
    difSorted = np.round(np.diff(timeSorted), 3)  # get 1st der
    
    ## CUT every variable till ALLDEVACQ
    for i_, d in enumerate(timeSorted):
        if d >= ALLDEVACQ:
            cut_time = i_
            break
    timeSorted = timeSorted[:cut_time]
    ntime = time[:cut_time]
    packetNbm = packetNbm[:cut_time]
    difpacketNbm = difpacketNbm[:cut_time-1]
    dif = dif[:cut_time-1]

    #for acq in correctAcq:
    #    print("1: ",   np.unique(np.diff(packetNbm[acq])))  # only 1 are shown 
        
    expected = round(intervalo_tempo*samp_rate, 3)  # get expected number of points for acquisition

    #realSR += [round(len(timeSorted)/intervalo_tempo, 3)]
    #realSR += [round(np.mean(intSR), 3)]
    
    #print("Real SR: ", realSR[-1])
    taxa = round((len(timeSorted)/expected)*100, 3)  # % of acquired data
    #dev_loss = round((expected - len(timeSorted))/expected, 3)*100  
    
    #lossPUns += [round(dev_loss + np.sign(dif).tolist().count(-1)/expected*100, 3)]
    
    #print("Device loss (%):", dev_loss)
    #print("Device loss + unsorted (%):", lossPUns[-1])
    dtime = timeSorted
    
    # save data on lists
    lis_ID += [int(ID)]  
    #loss += [dev_loss]
    timems = time*1000
    results, edges = np.histogram(dtime, bins=np.arange(int(np.max(dtime))))  # create histogram - np.arange(np.round(np.max(dtime)))
    
    #results = results/results.sum()
    results = minmax_scale(results) * 100
    binWidth = edges[1] - edges[0]  # bins width
    
    # store histogram data
    lis_avg_dtp += [round(np.mean(results), 3)]
    lis_std_dtp += [round(np.std(results), 3)]

    # data per time acquisition
    plt.figure()  # create figure
    plt.tight_layout()
    b = plt.bar(edges[:-1], results, binWidth)
    plt.xlabel("Time (s)")
    plt.ylabel("Data (%)")
    #plt.show()  
    plt.savefig("Plots/" + NUMDEV + "/" + str(samp_rate) + "/" + str(C) + "_DEV" + str(ID) + "_DatapTm_SR" + str(samp_period)+"_" + NUMDEV + ".pdf", format="pdf", bbox_inches='tight')
    #######
    plt.close('all') 

    time_sorted += ["No" if -1 in d else "Yes" for d in [np.unique(np.sign(dif))]]  # no data out of sorts
    
    minSort += [round(np.min(np.unique(timeSorted - ntime)), 3)]
    maxSort += [round(np.max(np.unique(timeSorted - ntime)), 3)]

    plt.figure()
    plt.tight_layout()
    
    results, edges = np.histogram(difSorted, bins=np.unique(difSorted))  # create histogram
    binWidth = edges[1] - edges[0]  # bins width
    results = results/results.sum()*100
    edges = edges[:-1]  # comment ?
    
    b = plt.bar(edges, results, binWidth)  # bar plot
    plt.xticks(edges, rotation='vertical')
    autolabel(b)
    plt.ylabel("Percentage")
    plt.xlabel("Time Diff")
    # plt.show()  
    plt.savefig("Plots/" + NUMDEV + "/" + str(samp_rate) + "/" + str(C) + "_DEV" + str(ID) + "_SRHist" + str(samp_period)+"_" + NUMDEV + ".pdf", format="pdf", bbox_inches='tight')
    plt.close('all') 
    
    # count samples in out of order
    OutOrdP += [round((np.sign(dif).tolist().count(-1)/len(dif))*100, 3)]
    print("Out of order (%)", OutOrdP[-1])    

    ####
    # Sampling period statistics
    # count samples in correct SR
    try:
        CorrectSP += [round(results[np.where(edges == samp_period)[0][0]], 3)]
        print("Equal to SP (%)", CorrectSP[-1])
        print("Equal to SP2 (%)", round(len(difSorted[np.where(difSorted == samp_period)[0]])/len(difSorted)*100, 3))     
    except Exception as e: 
        print(e, "not in SP samples")
        CorrectSP += [0.000]

    SupTSP += [round(np.sum(results[np.where(edges > samp_period)[0]]), 3)]
    print("Sup to SP (%)", SupTSP[-1])

    DIFSP += [round(np.sum(results[np.where(edges != samp_period)[0]]), 3)]
    print("Dif  SP (%)", DIFSP[-1])
    print("Dif  SP2 (%)", round(len(difSorted[np.where(difSorted != samp_period)[0]])/len(difSorted)*100, 3))

    difSorted[np.where(difSorted != samp_period)[0]]
    
    print("Check SP stats", DIFSP [-1] + CorrectSP[-1])
    
    # falhas no SP
    lis_min1 += [np.round(np.min(difSorted[np.where(difSorted != samp_period)[0]]), 3)]
    lis_max1 += [np.round(np.max(difSorted[np.where(difSorted != samp_period)[0]]), 3)]
    lis_mean1 += [np.round(np.mean(difSorted[np.where(difSorted != samp_period)[0]]), 3)]
    lis_med1 += [np.round(np.median(difSorted[np.where(difSorted != samp_period)[0]]), 3)]
    lis_std1 += [np.round(np.std(difSorted[np.where(difSorted != samp_period)[0]]), 3)]
    
    lis_std += [np.round(np.std(dif), 3)]
    lis_max += [np.round(np.max(dif), 3)]
    lis_min += [np.round(np.min(dif), 3)]
    lis_med += [np.round(np.median(dif), 3)]
    lis_mean += [np.round(np.mean(dif), 3)]
    
    difSortedCA = []
    for acq in correctAcq:
        difSortedCA += np.round(np.diff(timeSorted[acq]), 3).tolist()  # get 1st der
    
    lis_stdS += [np.round(np.std(difSortedCA), 3)]
    lis_maxS += [np.round(np.max(difSortedCA), 3)]
    lis_minS += [np.round(np.min(difSortedCA), 3)]
    lis_medS += [np.round(np.median(difSortedCA), 3)]
    lis_meanS += [np.round(np.mean(difSortedCA), 3)]
    
    completeAcqTime += [dtime[-1]]
    print("Acquisition Time: ", completeAcqTime[-1])

    unqdifpacketNbm = list(np.unique(difpacketNbm))
    unqdifpacketNbm.remove(1)
    try:
        unqdifpacketNbm.remove(-2**16 +1)
    except:
        print(" ")
    
    PNstd += [np.round(np.std(unqdifpacketNbm), 3)]
    PNmax += [np.round(np.max(unqdifpacketNbm), 3)]
    PNmin += [np.round(np.min(unqdifpacketNbm), 3)]
    PNmed += [np.round(np.median(unqdifpacketNbm), 3)]
    PNmean += [np.round(np.mean(unqdifpacketNbm), 3)]

    counterCorrPN = 100 - ((difpacketNbm.count(1) + difpacketNbm.count(-2**16+1))/len(difpacketNbm)* 100)
    PNNoErrPerc += [np.round(counterCorrPN, 3)]
    print("PN error: ", PNNoErrPerc[-1])
    
    # plot temporal PN
    plt.figure()
    plt.tight_layout()
    #plt.plot(timeSorted, packetNbm, "o")
    plt.plot(np.arange(len(packetNbm)), packetNbm, ".")    
    plt.xlabel("Time (s)")
    plt.ylabel("Packet Number (a.u.)")
    # plt.show()  
    plt.savefig("Plots/" + NUMDEV + "/" + str(samp_rate) + "/" + str(C) + "_DEV" + str(ID) + "_packetNBPTime" + str(samp_period)+"_" + NUMDEV + ".pdf", format="pdf", bbox_inches='tight')
    plt.close('all') 

    plt.figure()
    plt.tight_layout()
    #plt.plot(timeSorted, packetNbm, "o")
    plt.plot(np.arange(len(np.diff(packetNbm))), np.diff(packetNbm), ".")    
    plt.xlabel("Time (s)")
    plt.ylabel("Packet Number (a.u.)")
    # plt.show()  
    plt.savefig("Plots/" + NUMDEV + "/" + str(samp_rate) + "/" + str(C) + "_DEV" + str(ID) + "_DIFFpacketNBPTime" + str(samp_period)+"_" + NUMDEV + ".pdf", format="pdf", bbox_inches='tight')
    plt.close('all') 

    #try:
    #    difpacketNbm.remove(-2**16 +1)  # remove natural reset of variable overflow
    #except:
    #    print("")
            
    resetIDX, c_resetC = [], 0
    for i, d in enumerate(difpacketNbm):
        if d <= -5:
            if d <= -65500:
                continue  ## ignore - overflow but lost data
            resetIDX += [i] 
            c_resetC += 1
    resetC += [c_resetC]
    resetIDX = np.array(resetIDX)
     
    timeSorted = np.array(timeSorted)
    if len(resetIDX) > 0:
        print("resetIDX", timeSorted[resetIDX+1])
        
        resetT = timeSorted[np.array(resetIDX)+1] - timeSorted[np.array(resetIDX)]
        
        stdresetT += [np.round(np.std(resetT), 3)]
        maxresetT += [np.round(np.max(resetT), 3)]
        minresetT += [np.round(np.min(resetT), 3)]
        medresetT += [np.round(np.median(resetT), 3)]
        meanresetT += [np.round(np.mean(resetT), 3)]
        s = 0
        _realSR = []
        for e in resetIDX:
            e += 1
            _realSR += [round(len(timeSorted[s:e])/(timeSorted[e]-timeSorted[s]), 3)]
            s = e
        realSR += [round(np.mean(_realSR), 3)]  
    else:
        print("resetIDX", resetIDX)
        
        stdresetT += [0.000]
        maxresetT += [0.000]
        minresetT += [0.000]
        medresetT += [0.000]
        meanresetT += [0.000]
        realSR += [round(len(timeSorted)/intervalo_tempo, 3)]
        

#  get average +- std of all devices
#print("loss:", round(np.mean(loss), 3), "+-", round(np.std(loss), 3))

DIR = DIR.split("/")[-1]
dict = {'DIR': DIR, 'Expected Samples/s': samp_rate, 'Obtained Samples/s': realSR, 'ID': lis_ID, "Acq Time (s)": completeAcqTime, "SP Out of Order (%)": OutOrdP , "Min out of order (s)": minSort, "Max out of order (s)": maxSort, "On SP (%)": CorrectSP, "Min Sorted DER (s)": lis_minS, "Mean Sorted DER (s)": lis_meanS, "Med Sorted DER": lis_medS, "STD Sorted DER (s)": lis_stdS, "Max Sorted DER (s)": lis_maxS, "!= SP (%) ": DIFSP, "!= SP Sorted DER Mean (s)": lis_mean1, "!= SP Sorted DER Med (s)": lis_med1, '!= SP Sorted DER STD (s)': lis_std1, '!= SP Sorted DER Max (s)': lis_max1, '!= SP Sorted DER Min (s)': lis_min1, 'Time Hist Data Points (s)': lis_avg_dtp, 'Time Hist Data Points std (s)': lis_std_dtp, "PN Loss (%)": PNNoErrPerc, "PN mean (N.S)": PNmean, "PN med (N.S)": PNmed, "PN std (N.S)": PNstd,"PN max (N.S)": PNmax, "PN min (N.S)": PNmin, "Reset counter (a.u.)": resetC, "Mean Reset Tm (s)": meanresetT, "Med Reset Tm (s)": medresetT, "STD Reset Tm (s)": stdresetT, "Max Reset Tm (s)": maxresetT, "Min Reset Tm (s)": minresetT}  # create dict w/ data
df = pd.DataFrame(dict)  # convert to dataframe
df.to_csv("Plots/" + NUMDEV + "/" + str(samp_rate) + "/RESULTS/" + str(C) + "_DEV" + str(ID) + str(samp_period)+"_" + NUMDEV + ".csv", index=False)  # save data locally

