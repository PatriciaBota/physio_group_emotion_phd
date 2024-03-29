import numpy as np
import pyhrv
from .. import utils
from .. import eda
import json


def eda_features(signal=None, TH=0.08, sampling_rate=1000.):
    """Compute EDA characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    onsets : list
        Signal EDR events onsets.

    pks : list
        Signal EDR events peaks.

    amps : list
        Signal EDR events Amplitudes.

    phasic_rate : list
        Signal EDR events rate in 60s.

    rise_ts : list
        Rise times, i.e. onset-peak time difference.

    half_rise : list
        Half Rise times, i.e. time between onset and 50% amplitude.

    half_rec : list
        Half Recovery times, i.e. time between peak and 63% amplitude.

    six_rise : list
        63 % rise times, i.e. time between onset and 63% amplitude.

    six_rec : list
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """
    # ensure numpy
    signal = np.array(signal)
    args, names = [], []

    # onsets, pks, amps
    onsets, pks, amps, _ = eda.get_eda_param(signal, min_amplitude=TH, sampling_rate=sampling_rate)
    args += [onsets]
    names += ['onsets']
    args += [pks]
    names += ['pks']
    args += [amps]
    names += ['amps']

    # phasic_rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(pks))
    except:
        phasic_rate = None
    args += [phasic_rate]
    names += ['phasic_rate']

    # rise_ts
    try:
        rise_ts = np.array(pks) - np.array(onsets)
    except:
        rise_ts = None
    args += [rise_ts]
    names += ['rise_ts']

    # half, six, half_rise, half_rec, six_rec
    #_, _, half_rise, half_rec, six_rise, six_rec = eda.edr_times(signal, onsets, pks)
    half_rec, six_rec, _, _ = eda.Mar_edr_times(signal, onsets, pks)
        
    #args += [half_rise]
    #names += ['half_rise']
    args += [half_rec]
    names += ['half_rec']
    #args += [six_rise]
    #names += ['six_rise']
    args += [six_rec]
    names += ['six_rec']

    args = np.nan_to_num(args)

    return utils.ReturnTuple(tuple(args), tuple(names))
