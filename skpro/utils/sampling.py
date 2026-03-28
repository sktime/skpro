# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Sampling utility functions for bootstrap and bagging estimators."""

__author__ = ["fkiraly"]

import numpy as np


def _random_ss_ix(ix, size, replace=True, random_state=None):
    """Randomly uniformly sample indices from a list of indices.

    Parameters
    ----------
    ix : array-like
        list of indices to sample from
    size : int
        number of indices to sample
    replace : bool, default=True
        whether to sample with replacement
    random_state : int, RandomState instance or None, optional (default=None)
        If RandomState instance, used as the random number generator.
        If None, a new RandomState instance is created.

    Returns
    -------
    ixs : array-like
        sampled indices, same type as ``ix``
    """
    if random_state is None:
        random_state = np.random.RandomState()

    a = range(len(ix))
    ixs = ix[random_state.choice(a, size=size, replace=replace)]
    return ixs
