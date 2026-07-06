# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Sampling utility functions for bootstrap and bagging estimators."""

__author__ = ["fkiraly"]

from skpro.utils.random_state import check_random_state


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
        If int, used as the seed.
        If None, the global random state is used.

    Returns
    -------
    ixs : array-like
        sampled indices, same type as ``ix``
    """
    random_state = check_random_state(random_state)

    a = range(len(ix))
    ixs = ix[random_state.choice(a, size=size, replace=replace)]
    return ixs
