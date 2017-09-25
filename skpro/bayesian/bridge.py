import numpy as np


def ECDF(a):
    """
    Return the Empirical CDF of an array as x-y mapping

    :param a: Sample
    :return:
    """
    xs = np.sort(a)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys


def ecdf(a):
    """
    Return the Empirical CDF of an array as function
    :param a:
    :return:
    """

    xs, ys = ECDF(a)

    def cdf_hat(x):
        index = np.searchsorted(xs, x)
        index = len(ys) - 1 if index >= len(ys) else index
        return ys[index]
    
    return cdf_hat