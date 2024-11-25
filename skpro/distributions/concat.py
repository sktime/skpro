# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Concat operation."""

__author__ = ["SaiRevanth25"]

from skpro.distributions._concat import ConcatDistr


def concat(objs, axis=0):
    """
    Concatenate a list of distributions into a ConcatDistr.

    Parameters
    ----------
    objs : list
        List of distribution-like objects to concatenate.
    axis : {0/'index', 1/'columns'}, default 0
        Axis to concatenate along.

    Returns
    -------
    ConcatDistr
        An object representing the concatenation of the given distributions.

    Examples
    --------
    >>> import skpro.distributions as skpro
    >>> d1 = Normal(mu=[[1, 2], [3, 4]], sigma=1)
    >>> d2 = Normal(mu=0, sigma = [[2, 42]])
    >>> skpro.concat([d1,d2]).mean()
            0	1
        0	1	2
        1	3	4
        2	0	0
    >>> skpro.concat([d1,d2]).var()

            0	1
        0	1	1
        1	1	1
        2	4	1764
    >>> d3 = Gamma(alpha=[[5, 2]], beta=4)
    >>> d4 = Laplace(mu= [5,7], scale=[2,8])
    >>> skpro.concat([d2,d3,d4]).pdf(x=1)
                    0	 1
        Normal	4.0000	1764.000
        Gamma	0.3125	0.125
        Laplace	8.0000	128.000
    """
    if not isinstance(objs, list):
        raise ValueError("`objs` must be a list of distribution-like objects.")
    if axis not in [0, 1, "index", "columns"]:
        raise ValueError("`axis` must be one of {0, 1, 'index', 'columns'}.")

    return ConcatDistr(objs, axis=axis)
