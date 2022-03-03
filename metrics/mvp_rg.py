import math
import numpy as np


def transform(src, r, t):
    r = r.transpose(0, 2, 1)
    res = src @ r + t[:, None, :]
    return res


def RMSE(src, r1, r2, t1, t2):
    '''
    calculate rmse.
    :param src: shape=(B, n, 3), point clouds
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :param t1: shape=(B, 3), pred
    :param t2: shape=(B, 3), gt
    :return:
    '''
    tgt1 = transform(src, r1, t1)
    tgt2 = transform(src, r2, t2)
    rmse = np.sum(np.sqrt(np.sum((tgt1 - tgt2) ** 2, axis=-1)), axis=-1) / src.shape[1]
    return rmse
