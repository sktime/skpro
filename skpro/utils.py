import numpy as np


def ecdf(a, return_func=False):
    """

    Parameters
    ----------
    a: array
        Input array representing a sample
    return_func: bool
        If true, a python function that represents the ecdf is returned

    Returns
    -------
    mixed   Empirical CDF of the input sample
    """
    xs = np.sort(np.array(a))
    ys = np.arange(1, len(xs)+1)/float(len(xs))

    if not return_func:
        return xs, ys

    def func(x):
        index = np.searchsorted(xs, x)
        index = len(ys) - 1 if index >= len(ys) else index
        return ys[index]

    return func

