# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd


def empirical_from_rvdf(dist, index=None, columns=None):
    """Convert a statsmodels rv_discrte_float to an skpro Empirical object.

    Parameters
    ----------
    dist : rv_discrte_float object
        Instance of rv_discrete.
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

    xk = dist.xk
    pk = dist.pk

    xks = [xk[i] for i in range(len(xk))]
    pks = [pk[i] for i in range(len(pk))]

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
