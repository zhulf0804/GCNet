import math
import numpy as np


def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def Error_t(t1, t2):
    '''
    calculate translation mse error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    assert t1.shape == t2.shape
    error_t = np.sqrt(np.sum((t1 - t2) ** 2, axis=1))
    return error_t
