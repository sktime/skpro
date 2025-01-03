# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Binomial probability distribution."""

__author__ = ["meraldoantonio"]

import pandas as pd
from scipy.stats import binom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Binomial(_ScipyAdapter):
    r"""Binomial distribution.

    Most methods wrap ``scipy.stats.binom``.
    The Binomial distribution is parameterized by the number of trials :math:`n`
    and the probability of success :math:`p`,
    such that the probability mass function (PMF) is given by:

    .. math:: P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}

    Parameters
    ----------
    n : int or array of int (1D or 2D), must be non-negative
    p : float or array of float (1D or 2D), must be in [0, 1]
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.binomial import Binomial

    >>> d = Binomial(n=[[10, 10], [20, 30], [40, 50]], p=0.5)
    """

    _tags = {
        "capabilities:approx": ["pmf"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, n, p, index=None, columns=None):
        self.n = n
        self.p = p

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return binom

    def _get_scipy_param(self):
        n = self._bc_params["n"]
        p = self._bc_params["p"]

        return [], {"n": n, "p": p}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"n": [[10, 10], [20, 30], [40, 50]], "p": 0.5}
        params2 = {
            "n": 10,
            "p": 0.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"n": 15, "p": 0.7}

        return [params1, params2, params3]
