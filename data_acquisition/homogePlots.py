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





labels = ["40 (3 Ant)", "16 (3 Ant)", "40 (1 Ant)", "16 (1 Ant)", "16 (Multi)", "10 (Multi)"]

_labels = []
my_dpi = 300
fig, axes = plt.subplots(figsize=(3600 / my_dpi, 2200 / my_dpi), dpi=my_dpi)
plt.tight_layout()

# [1 dev, 10 dev, 20 dev]
#p = plt.violinplot([0, 0, 83], showmeans=True, showextrema=True)
#add_label(p, "16 (3 Ant)")

#p = plt.violinplot([getLoss("1dev", 60, "2021525_2141_DATA"), getLoss("10dev", 60, "2021613_81726_DATA"), getLoss("20dev", 60, "2021526_131724_DATA")], showmeans=True, showextrema=True)
#add_label(p, "16 (3 Antennas)")
#p = plt.violinplot([[98.353], [90.405, 90.405+9.369,90.405-9.369], [85.23, 85.23+26.624, 85.23-26.624]], showmeans=True, showextrema=True)
p = plt.violinplot([[98.353], [90.405, 90.405+9.369,90.405-9.369], [85.23, 85.23+26.624, 85.23-26.624]], showmeans=True, showextrema=True)
add_label(p, "40ms")



plt.ylim(0, 100)

#labels = ['A', 'B', 'C', 'D']
set_axis_style(axes, [1, 10, 20])
#axes.set_xticklabels(["A", "B"], ha="center") 
plt.xlabel("Number of Devices")
plt.ylabel("Time Consistency (%)") 
plt.legend(*zip(*_labels), bbox_to_anchor=(1.4, 1))
plt.savefig("Syst_TimCons.pdf", format="pdf", bbox_inches='tight')
