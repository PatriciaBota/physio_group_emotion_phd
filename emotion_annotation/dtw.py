# Imports
import numba
import numpy as np

@numba.jit
def data_std(vector):
    """
    vector:
    :return:
    """
    vmean = np.mean(vector)
    vstd = np.std(vector)
    vnorm = [0]*len(vector)
    for v in range(len(vector)):
        if not vstd:
            vnorm[v] = (vector[v] - vmean)
        else:
            vnorm[v] = (vector[v] - vmean)/vstd

    return np.array(vnorm)

@numba.jit
def dtw(x, y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = (x[i] - y[j]) ** 2
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    path = _traceback(D0)
    if not np.sum(D1.shape):
        d = D1[-1, -1]
    else:
        d = D1[-1, -1]/np.sum(D1.shape)
    return d, C, D1, path


@numba.jit
def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


@numba.jit
def new_matrix_dtw_dist(signal, templates):
    lx_s = len(signal)
    lx_c = len(templates)
    dist_matrix_custom = np.zeros((lx_s, lx_c))
    for i in range(lx_s):
        for j in range(lx_c):
            # try:
            #     dist_matrix_custom[i, j], C, D1, path = dtw(data_std(signal[i]), data_std(templates[j]))
            # except ZeroDivisionError:
            #     dist_matrix_custom[i, j] = 0
            dist_matrix_custom[i, j], C, D1, path = dtw(data_std(signal[i]), data_std(templates[j]))
    return np.nan_to_num(np.array(dist_matrix_custom))


@numba.jit
def new_matrix_pcorr_dist(signal, templates):
    lx_s = len(signal)
    lx_c = len(templates)
    dist_matrix_custom = np.zeros((lx_s, lx_c))
    for i in range(lx_s):
        for j in range(lx_c):
            dist_matrix_custom[i, j] = bs.pearson_correlation(signal[i], templates[j])[0]
    return np.nan_to_num(np.array(dist_matrix_custom))


@numba.jit
def new_matrix_eucd_dist(signal, templates):
    lx_s = len(signal)
    lx_c = len(templates)
    dist_matrix_custom = np.zeros((lx_s, lx_c))
    for i in range(lx_s):
        for j in range(lx_c):
            dist_matrix_custom[i, j] = np.sqrt(np.sum((signal[i] - templates[j]) ** 2))
    return np.nan_to_num(np.array(dist_matrix_custom))

from scipy.stats import multivariate_normal


@numba.jit
def new_matrix_prob_dist(signal, templates_m, templ_cov):
    lx_s = len(signal)
    lx_c = len(templates_m)
    dist_matrix_custom = np.zeros((lx_s, lx_c))
    for i in range(lx_s):
        for j in range(lx_c):
            dist_matrix_custom[i, j] = multivariate_normal.pdf(signal[i], mean=templates_m[j], cov=templ_cov[j], allow_singular='True')
    return np.nan_to_num(np.array(dist_matrix_custom))
