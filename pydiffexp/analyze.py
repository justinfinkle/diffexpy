import numpy as np
import pandas as pd
import scipy.special as special

"""
This module is meant to facilitate the analysis of gene trajectories
"""


def pairwise_corr(x, y, axis=0):
    """
    Memory efficient vector calculation of features
    Adapted from https://stackoverflow.com/questions/33650188/efficient-pairwise-correlation-for-two-matrices-of-features

    :param x: DataFrame
    :param y: DataFrame, must share one dimension of x
    :param axis: int; 0 or 1. Axis along which to correlate
    :return:
    """
    if axis == 1:
        x = x.T
        y = y.T
    n_rows = y.shape[0]
    sx = x.sum(0)
    sy = y.sum(0)
    p1 = n_rows * np.dot(y.T, x)
    p2 = pd.DataFrame(sx).dot(pd.DataFrame(sy).T).T
    p3 = n_rows * ((y ** 2).sum(0)) - (sy ** 2)
    p4 = n_rows * ((x ** 2).sum(0)) - (sx ** 2)

    # pearsonr
    r = (p1 - p2) / np.sqrt(pd.DataFrame(p4).dot(pd.DataFrame(p3).T).T)

    # This part is directly from the scipy pearsonr function to calculate p-values
    # Code was linearized for an array

    n = len(x)
    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r[r > 1] = 1.0
    r[r < -1] = -1.0
    df = n - 2

    t_squared = r ** 2 * (df / ((1.0 - r) * (1.0 + r)))

    # This function is directly from scipy.stats
    def _betai(x):
        x = np.asarray(x)
        x = np.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
        return special.betainc(0.5 * df, 0.5, x)

    prob = (df/(df+t_squared)).applymap(_betai)

    # If the correlation is 1 then the p is 0
    prob[np.array(np.abs(r) == 1)] = 0

    return r, prob


def to_idx_cluster(x):
    """
    Make list-like into indexable string
    :param x: list-like
    :return:
    """

    return str(tuple(x))
