import numpy as np
from .. import utils
from .. import tools as st
import json
#from scipy import signal
#from scipy import fftpack as fft
from scipy.stats import stats
import tsfel

def mfcc(spectrum, f, num_coeffs=26):
    sp = np.square(spectrum) / len(f)

    mel_freqs = np.linspace(0, 2595 * np.log10(1 + f[-1] / 700.), num_coeffs + 2)
    freqs = 700 * (np.power(10, (mel_freqs / 2595)) - 1)

    indexes = np.array(freqs * len(f) / freqs[-1], np.int32)
    filter_results = np.empty(0)

    for i in range(0, num_coeffs):
        bin1 = indexes[i]
        bin2 = indexes[i + 2]
        peak = indexes[i + 1]

        if bin1 != bin2 != peak:
            a = sp[bin1:bin2]
            b = mfcc_filter(bin2 - bin1, peak - bin1)
            filter_results = np.append(filter_results, np.dot(a, b))
        else:
            filter_results = np.append(filter_results, [0])
    out = fft.dct(filter_results, type=2, axis=0, norm='ortho')

    return out


def mfcc2(spectrum, rate, num_coeffs=26):
    sp = np.square(spectrum) / len(spectrum)
    f = np.linspace(0, rate/2, len(spectrum))
    mel_freqs = np.linspace(0, 2595 * np.log10(1 + f[-1] / 700.), num_coeffs + 2)
    freqs = 700 * (np.power(10, (mel_freqs / 2595)) - 1)

    indexes = np.array(freqs * (len(f)-1) / freqs[-1], np.int32)
    filter_results = np.empty(0)

    for i in range(0, num_coeffs):
        bin1 = indexes[i]
        bin2 = indexes[i + 2]
        peak = indexes[i + 1]

        if bin1 != bin2 != peak:
            a = sp[bin1:bin2+1]
            b = mfcc_filter(bin1, peak, bin2)
            value = np.dot(a, b)
            if value != 0:
                filter_results = np.append(filter_results, np.log(np.dot(a, b)))
            else:
                filter_results = np.append(filter_results, np.log(1e-308))
        else:
            filter_results = np.append(filter_results, [1e-308])
    out = fft.dct(filter_results, type=2, axis=0, norm='ortho')

    return out


def mfcc_filter(bin1, peak, bin2):
    if bin2 == bin1:
        return np.ones(1)
    if peak == bin1:
        return np.linspace(1, 0, bin2-bin1+1)
    out = np.linspace(0, 1, peak-bin1+1)
    out = np.append(out, np.linspace(1, 0, bin2-peak+1)[1::])
    return out


def signal_mfcc(t, _signal, FS):
    """Compute spectral metrics describing the signal.
        Parameters
        ----------
        signal : array
            Input signal.
        FS : float
            Sampling frequency

        Returns
        -------


        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        """
    # check inputs
    if _signal is None or np.array(_signal) == []:
        print("Signal is empty.")

    # ensure numpy
    _signal = np.array(_signal)

    #spectrum = np.fft.fft(_signal)
    args, names = [], []
    
    try:
        #_mfcc = mfcc(spectrum, 8000)
        _mfcc = tsfel.feature_extraction.features.mfcc(_signal, FS)
    except:
        _mfcc = [None]
    
    #_mfcc = mfcc2(spectrum, 8000)
        
    #args += [_mfcc.tolist()]
    #names += ['mfcc']

    # kurtosis
    try:
        kurtosis = stats.kurtosis(_mfcc, bias=False)
    except:
        kurtosis = None
    args += [kurtosis]
    names += ['kurtosis']

    # skweness
    try:
        skewness = stats.skew(_mfcc, bias=False)
    except:
        skewness = None
    args += [skewness]
    names += ['skewness']

    args = np.nan_to_num(args)
    return utils.ReturnTuple(tuple(args), tuple(names))
