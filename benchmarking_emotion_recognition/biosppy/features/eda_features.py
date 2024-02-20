import numpy as np
import pyhrv
from .. import utils
from .. import eda
import json
from . import resp_features

def eda_features(_eda, edr=None, sampling_rate=1000., min_amplitude=0):
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
    dict = json.load(open('eda_features_log.json'))
    args, names = [], []

    # # # get EDR signal
    if edr is not None:
        try:
            edr = eda.get_scr(_eda, sampling_rate)
        except:
            edr = []

    # onsets, pks, amps
    onsets, pks, amps, _ = eda.get_eda_param(edr, min_amplitude)
    if dict['onsets']['use'] == 'yes':
        args += [onsets]
        names += ['onsets']
    if dict['pks']['use'] == 'yes':
        args += [pks]
        names += ['pks']
    if dict['amps']['use'] == 'yes':
        args += [amps]
        names += ['amps']

    # phasic_rate
    if dict['phasic_rate']['use'] == 'yes':
        try:
            phasic_rate = sampling_rate * (60. / np.diff(pks))
        except:
            phasic_rate = 0.0
        args += [phasic_rate]
        names += ['phasic_rate']

    # half, six, half_rise, half_rec, six_rec
    # eda_onsets, eda_pks, eda_amp, _ = eda.get_eda_param(_eda, min_amplitude)
    _, _, half_rise, half_rec, six_rise, six_rec = eda.edr_times(_eda, onsets, pks)

    if dict['rise_ts']['use'] == 'yes':
        # rise_ts
        try:
            rise_ts = np.array(pks) - np.array(onsets)
        except:
            rise_ts = 0.0
        args += [rise_ts]
        names += ['rise_ts']

    if dict['half_rise']['use'] == 'yes':
        args += [half_rise]
        names += ['half_rise']
    if dict['half_rec']['use'] == 'yes':
        args += [half_rec]
        names += ['half_rec']
    if dict['six_rise']['use'] == 'yes':
        args += [six_rise]
        names += ['six_rise']
    if dict['six_rec']['use'] == 'yes':
        args += [six_rec]
        names += ['six_rec']
    if dict['onPkVol']['use'] == 'yes':
        args += [resp_features.calc_inhale(_eda, onsets)[0]]
        names += ['onPkVol']
    if dict['pkOnVol']['use'] == 'yes':
        args += [resp_features.calc_exhale(_eda, onsets)[0]]
        names += ['pkOnVol']
    if dict['EDRVolRatio']['use'] == 'yes':
        args += [resp_features.calc_inhExhRatio(resp_features.calc_inhale(_eda, onsets)[0], resp_features.calc_exhale(_eda, onsets)[0])[0]]
        names += ['EDRVolRatio']

    return utils.ReturnTuple(tuple(args), tuple(names))
