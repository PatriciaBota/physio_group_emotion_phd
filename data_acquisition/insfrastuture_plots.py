import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
import datetime
import os
import glob
#NUMDEV = NUMDEV.replace("/", "_")
plt.rcParams.update({'font.size': 20})


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

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

def add_label(parts, label):
    import matplotlib.patches as mpatches    
    color = parts["bodies"][0].get_facecolor().flatten()
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = parts[partname]
        vp.set_edgecolor(color)
        vp.set_linewidth(1)
    _labels.append((mpatches.Patch(color=color), label))


def getLoss(NUMDEV, samp_rate, C):
    _fLOSS = []
    d =  "DATA/" + NUMDEV + "/" + str(samp_rate) + "/" + C + ".hdf5"   # estou diretamente  a emitir

    f = h5py.File(d, 'r')

    PNNoErrPerc, lis_max1, maxtm, mintm, maxWT = [], [], [], [], []
    for ID in list(f.keys()):  # iterate over devices
        time = np.array(f[ID]['EDA'][:, -1]*0.001)   # get time in s
        timeSorted = np.sort(time)   # sorted time

        maxtm += [timeSorted[-1]]
        mintm += [timeSorted[0]]
        print(ID, mintm[-1], maxtm[-1])

        ENDACQ = np.min(maxtm)
        STARTACQ = np.max(mintm)

        print("S-E ACQ", STARTACQ,  ENDACQ)  # start and end time of acquisition (time when ALL devices were acquiring data)

    for ID in list(f.keys()):  # iterate over devices
        print("ID", ID)

        samp_rate = int(f[ID].attrs['sampling rate'])  # get sampling rate  ATTENTION: confirm it its right/same as said in file name

        samp_period = 1/samp_rate
        if samp_rate == 70:
            samp_period = 0.014
        if samp_rate == 60:
            samp_period = 0.016

        print("Sampling Period", samp_period)
        print("Sampling Rate", samp_rate)

        time = np.array(f[ID]['EDA'][:, -1]*0.001)   # get time in s
        dif = np.round(np.diff(time), 3)  # get time 1st derivative    
        timeSorted = np.sort(time)   # sorted time  # ATTENTION: doesn't work if device was turned off and on (time restarts) 
        difSorted = np.round(np.diff(timeSorted), 3)  # get 1st der
        
        packetNbm = f[ID]['EDA'][:, -2]  # package number
        packetNbm = packetNbm[np.argsort(time)].astype(int)  # sort packet number using time
        difpacketNbm = np.diff(packetNbm)

        # cut by start, end acq time
        for i_, d in enumerate(timeSorted):  # find index of cut time (with minimum acquisition time from all devices)
            if d >= ENDACQ:
                cut_time = i_
                break
        for i_, d in enumerate(timeSorted):  # find index of cut time (with minimum acquisition time from all devices)
            if d >= STARTACQ:
                start_time = i_
                break

        # cut variables to minimum time when all devices are acquiring data
        timeSorted = timeSorted[start_time:cut_time]
        difSorted = difSorted[start_time:cut_time-1]
        
        packetNbm = packetNbm[start_time:cut_time]
        difpacketNbm = difpacketNbm[start_time:cut_time-1]  # TODO: UNCERTAIN -1 or +1

        # loss
        difpacketNbm = difpacketNbm.tolist()

        lost_data_riot = 0 #conta o numero de pacotes perdidos

        for g in range(1, len(packetNbm)):
            if not((packetNbm[g]-packetNbm[g-1]) == 1 or \
                    (packetNbm[g]-packetNbm[g-1]) == -65535) and (packetNbm[g]-packetNbm[g-1]) != 0:
                if packetNbm[g]-packetNbm[g-1]>1:
                    lost_data_riot += (packetNbm[g]-packetNbm[g-1])-1
                elif packetNbm[g]-packetNbm[g-1]<0:
                    lost_data_riot += (packetNbm[g]-0) + (2**16-1 - packetNbm[g-1])

        counterCorrPN = lost_data_riot*100/(lost_data_riot+len(packetNbm))
        print("PN error (%):", np.round(counterCorrPN, 3))
        
        _fLOSS += [counterCorrPN]
    return _fLOSS


labels = ["40 (3 Ant)", "16 (3 Ant)", "40 (1 Ant)", "16 (1 Ant)", "16 (Multi)", "10 (Multi)"]

_labels = []
my_dpi = 300
fig, axes = plt.subplots(figsize=(3600 / my_dpi, 2200 / my_dpi), dpi=my_dpi)
plt.tight_layout()

# [1 dev, 10 dev, 20 dev]
#p = plt.violinplot([0, 0, 83], showmeans=True, showextrema=True)
#add_label(p, "16 (3 Ant)")

p = plt.violinplot([getLoss("1dev", 60, "2021525_2141_DATA"), getLoss("10dev", 60, "2021613_81726_DATA"), getLoss("20dev", 60, "2021526_131724_DATA")], showmeans=True, showextrema=True)
add_label(p, "16 (3 Antennas)")
p = plt.violinplot([getLoss("1dev", 25, "2021525_111742_DATA"), getLoss("10dev", 25, "2021527_11724_DATA"), getLoss("20dev", 25, "2021529_181732_DATA")], showmeans=True, showextrema=True)
add_label(p, "40 (3 Antennas)")
p = plt.violinplot([[-1], getLoss("10dev", 60, "2021528_141722_DATA"), [-1]], showmeans=True, showextrema=True)
add_label(p, "16 (1 Antenna)")
p = plt.violinplot([[-1], [-1], getLoss("20dev", 25, "2021528_231722_DATA")], showmeans=True, showextrema=True) # added last
add_label(p, "40 (1 Antenna)") # added last
p = plt.violinplot([[-1]*20, [-1]*20, getLoss("20dev", 60, "2021530_31724_DATA")], showmeans=True, showextrema=True)
add_label(p, "16 (Bridge)")
p = plt.violinplot([[-1]*10, getLoss("10dev", 100, "2021530_191732_DATA"), [-1]*10], showmeans=True, showextrema=True)
add_label(p, "10 (Bridge)")

#p = plt.violinplot([getLoss("1dev", 25, "2021525_111742_DATA"), getLoss("10dev", 25, "2021527_11724_DATA"), getLoss("20dev", 25, "2021529_181732_DATA")], showmeans=True, showextrema=True)
#add_label(p, "40 SP")
#p = plt.violinplot([getLoss("1dev", 60, "2021525_2141_DATA"), getLoss("10dev", 60, "2021613_81726_DATA"), getLoss("20dev", 60, "2021526_131724_DATA")], showmeans=True, showextrema=True)
#add_label(p, "16 SP")
#p = plt.violinplot([getLoss("1dev", 200, "2021419_111732_DATA"), getLoss("10dev", 200, "2021526_61731_DATA"), [-1]], showmeans=True, showextrema=True)
#add_label(p, "5 SP")


#p = plt.violinplot([getLoss("10dev", 25, "202164_21724_DATA"), getLoss("20dev", 25, "2021612_141756_DATA")], showmeans=True, showextrema=True)
#add_label(p, "Default Firmware 40 SP")
#p = plt.violinplot([getLoss("10dev", 25, "2021527_11724_DATA"), getLoss("20dev", 25, "2021529_181732_DATA")], showmeans=True, showextrema=True)
#add_label(p, "EmotiphAI Firmware 40 SP")
#p = plt.violinplot([getLoss("10dev", 60, "202163_91742_DATA"), [-1]], showmeans=True, showextrema=True)
#add_label(p, "Default Firmware 16 SP")
#p = plt.violinplot([getLoss("10dev", 60, "2021613_81726_DATA"), getLoss("20dev", 60, "2021526_131724_DATA")], showmeans=True, showextrema=True)
#add_label(p, "EmotiphAI Firmware 16 SP")

plt.ylim(0, 100)

#labels = ['A', 'B', 'C', 'D']
set_axis_style(axes, [1, 10, 20])
#axes.set_xticklabels(["A", "B"], ha="center") 
plt.xlabel("Number of Devices")
plt.ylabel("Loss (%)") 
plt.legend(*zip(*_labels), bbox_to_anchor=(1.4, 1))
plt.savefig("INFRA_LOSSpdev.pdf", format="pdf", bbox_inches='tight')




