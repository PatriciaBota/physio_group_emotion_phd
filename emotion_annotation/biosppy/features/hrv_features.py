import numpy as np
import math
from .. import utils


def time_domain(nni, sampling_rate=4):

    nn50 = len(np.argwhere(abs(np.diff(nni))>0.05*sampling_rate))
    # pnn50 = nn50 / len(nni)
    sdnn = nni.std()
    rmssd = ((np.diff(nni) ** 2).mean()) ** 0.5

    return rmssd, sdnn, nn50 #, pnn50


def pointecare_feats(nn):
    x1 = np.asarray(nn[:-1])
    x2 = np.asarray(nn[1:])

    # SD1 & SD2 Computation
    sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
    sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

    csi = sd2/sd1

    csv = math.log10(sd1*sd2)

    return sd1, sd2, csi, csv


def signal_hrv(nni, sampling_rate=1000.):
    """ Compute BVP characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    ons : list
        Signal onsets.

    hr: list
        Bvp heart rate.
    """

    # ensure numpy array
    args, names = [], []

    try:
        rms_sd, sdnn, nn50 = time_domain(nni, sampling_rate)
        mean_nn = nni.mean()
        var = nni.var()
    except:
        rms_sd, sdnn, nn50, mean_nn, var = None, None, None, None, None
    args += [rms_sd]
    names += ['rms_sd']
    args += [sdnn]
    names += ['sdnn']
    args += [nn50]
    names += ['nn50']
    args += [mean_nn]
    names += ['mean_nn']
    args += [var]
    names += ['var']

    try:
        sd1, sd2, csi, csv = pointecare_feats(nni)
    except:
        sd1, sd2, csi, csv = None, None, None, None
    
    args += [sd1]
    names += ['sd1']
    args += [sd2]
    names += ['sd2']
    args += [csi]
    names += ['csi']
    args += [csv]
    names += ['csv']

    return utils.ReturnTuple(tuple(args), tuple(names))
