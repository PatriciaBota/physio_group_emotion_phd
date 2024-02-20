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

# local
import biosppy as bp



#DIR = np.sort(glob.glob("../3_Physio/Raw/*.hdf5"))  # replace with directory where the file is
DIR = ["../3_Physio/Raw/S13_physio_raw_data_M14.hdf5"]  # replace with directory where the file is
for d in DIR:
    f = h5py.File(d, 'r+')  # open file
    print("File: ", f)
    for ID in list(f.keys()):  # iterate over devices
        if ID != "LOGs" and ID != "Flags":  # is a device
            print("ID", ID)
            f[ID].attrs['movie'] = "The Last Pioneer"#f[ID].attrs['movie'][:-4]
    f.close()
