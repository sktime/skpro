# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd


def empirical_from_discrete(dist, index=None, columns=None):
    """Convert a list of scipy discrete distributions to an skpro Empirical object.

    Parameters
    ----------
    dist : list of rv_discrete
        List of scipy discrete distributions, instances of rv_discrete.
    index : pd.Index or coercible, optional
        Index of the resulting empirical distribution.
        Must be the same length as dist.
    columns : pd.Index or coercible, optional
        Columns of the resulting empirical distribution.
        Must be of length 1.
    """
    from skpro.distributions.empirical import Empirical

    if index is None:
        index = pd.RangeIndex(len(dist))

    xks = [d.xk for d in dist]
    pks = [d.pk for d in dist]

    lens = [len(xk) for xk in xks]
    idxs_inst = [np.repeat(index[i], leni) for i, leni in enumerate(lens)]
    idx_inst_flat = np.concatenate(idxs_inst)
    idx_spl = [np.arange(leni) for leni in lens]
    idx_spl_flat = np.concatenate(idx_spl)

    idx_mult = pd.MultiIndex.from_arrays([idx_spl_flat, idx_inst_flat])

    spl = pd.DataFrame(np.concatenate(xks), index=idx_mult, columns=columns)
    weights = pd.Series(np.concatenate(pks), index=idx_mult)

    emp = Empirical(
        spl=spl, weights=weights, time_indep=True, index=index, columns=columns
    )
    return emp
