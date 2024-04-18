# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Empirical quantile parametrized distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical


class QPD_Empirical(Empirical):
    r"""Empirical quantile parametrized distribution.

    This distribution is parameterized by a set of quantile points and quantiles,
    quantiles :math:`q_1, q_2, \dots, q_N`
    at quantile points :math:`p_1, p_2, \dots, p_N`,
    with :math:`0 \le p_1 < p_2 < \dots < p_N \le 1`.

    It represents a distribution with piecewise constant CDF and quantile function,
    the unique distribution satisfying:

    * the support is :math:`[q_1, q_N]`
    * for any quantile point :math:`p \in [p_1, p_N]`, it holds that
      :math:`\mbox{ppf}(p)` = :math:`\mbox{ppf}(p_i)`,
      where :math:`i` is the index minimizing :math:`|p_i - p|`,
      in all cases where this minimizer is unique.

    In vernacular terms, the quantile function agrees with the quantiles prescribed by
    :math:`q_i` at the quantile points :math:`p_i`, and for other quantile points
    agrees with the value at the nearest quantile point.

    In explicit terms, the distribution is an empirical distribution (sum-of-diracs),
    supported at the quantiles :math:`q_1, q_2, \dots, q_N`,
    with weights :math:`w_1, w_2, \dots, w_N`
    such that :math:`w_i = (p_{i+1} - p_{i-1})/2` for :math:`1 = 1, \dots, N`,
    where we define :math:`p_0 = -p_1` and :math:`p_{N+1} = 2 - p_N`.

    Formally, the distribution is parametrized by the quantiles :math:`q_i`
    and the quantile points :math:`p_i`, not by the quantiles and weights :math:`w_i`,
    so it is distinct from the empirical distribution (``skpro`` ``Empirical``),
    as a parameterized distribution,
    by being quantile parameterized and not sample parameterized.

    However, it is equivalent, as an unparameterized distribution,
    to an ``Empirical`` distribution with weights and nodes given as above.

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

    def _iloc(self, rowidx=None, colidx=None):
        index = self.index
        columns = self.columns

        spl_subset = self.quantiles

        if rowidx is not None:
            rowidx_loc = index[rowidx]
            # subset multiindex to rowidx by last level
            spl_subset = self.spl.loc[(slice(None), rowidx_loc), :]
            subs_rowidx = index[rowidx]
        else:
            subs_rowidx = index

        if colidx is not None:
            spl_subset = spl_subset.iloc[:, colidx]
            subs_colidx = columns[colidx]
        else:
            subs_colidx = columns

        return QPD_Empirical(
            spl_subset,
            time_indep=self.time_indep,
            index=subs_rowidx,
            columns=subs_colidx,
        )

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
