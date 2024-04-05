# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Empirical quantile parametrized distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical


class QPD_Empirical(Empirical):
    """Empirical quantile parametrized distribution.

    This distribution is parameterized by a set of quantile points.

    todo: add docstr

    Parameters
    ----------
    quantiles : pd.DataFrame with pd.MultiIndex
        quantile points
        first (lowest) index must be float, corresponding to quantile
        further indices are sample indices
    time_indep : bool, optional, default=True
        if True, ``sample`` will sample individual instance indices independently
        if False, ``sample`` will sample entire instances from ``spl``
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> import pandas as pd
    >>> from skpro.distributions import QPD_Empirical

    >>> spl_idx = pd.MultiIndex.from_product(
    ...     [[0.2, 0.5, 0.8], [0, 1, 2]], names=["alpha", "sample"]
    ... )
    >>> spl = pd.DataFrame(
    ...     [[0, 1], [2, 3], [4, 5], [1, 2], [4, 5], [7, 8], [2, 3], [6, 7], [10, 11]],
    ...     index=spl_idx,
    ...     columns=["a", "b"],
    ... )
    >>> dist = QPD_Empirical(spl)
    >>> empirical_sample = dist.sample(3)
    """

    _tags = {
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "cdf", "ppf"],
        "distr:measuretype": "discrete",
    }

    def __init__(self, quantiles, time_indep=True, index=None, columns=None):
        self.quantiles = quantiles
        self.time_indep = time_indep
        self.index = index
        self.columns = columns

        super().__init__(
            spl=quantiles,
            weights=self._get_empirical_weighted_sample(quantiles),
            time_indep=time_indep,
            index=index,
            columns=columns,
        )

    def _get_empirical_weighted_sample(self, empirical_spl):
        """Compute quantile weights for empirical distribution."""
        alphas = empirical_spl.index.get_level_values(0).unique()
        alpha_sorted = sorted(alphas)

        # obtain alpha weights for empirical distr such that we take the nearest
        # available quantile prob (the cum weights match the chosen quantile prob)
        alpha_np = np.array(alpha_sorted)
        alpha_diff = np.diff(alpha_np)
        alpha_diff2 = np.repeat(alpha_diff, 2) / 2
        weight_double = np.concatenate([[alpha_np[0]], alpha_diff2, [1 - alpha_np[-1]]])
        weight_double2 = weight_double.reshape(-1, 2)
        weights = weight_double2.sum(axis=1)

        # obtain weights per empirical sample
        empirical_spl_weights = pd.Series(index=empirical_spl.index)
        for i, a in enumerate(alpha_sorted):
            empirical_spl_weights.loc[a] = weights[i]

        return empirical_spl_weights

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # params1 is a DataFrame with simple row multiindex
        spl_idx = pd.MultiIndex.from_product(
            [[0.2, 0.5, 0.8], [0, 1, 2]], names=["alpha", "sample"]
        )
        spl = pd.DataFrame(
            [[0, 1], [2, 3], [4, 5], [1, 2], [4, 5], [7, 8], [2, 3], [6, 7], [10, 11]],
            index=spl_idx,
            columns=["a", "b"],
        )
        params1 = {
            "quantiles": spl,
            "time_indep": True,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }

        params2 = {
            "quantiles": spl,
            "time_indep": False,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
