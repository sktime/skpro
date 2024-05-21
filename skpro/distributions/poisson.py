# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Poisson probability distribution."""

__author__ = ["fkiraly", "malikrafsan"]

import pandas as pd
from scipy.stats import poisson, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Poisson(_ScipyAdapter):
    """Poisson distribution.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions import Poisson

    >>> distr = Poisson(mu=[[1, 1], [2, 3], [4, 5]])
    """

    _tags = {
        "capabilities:approx": ["ppf", "energy"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, index=None, columns=None):
        self.mu = mu

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return poisson

    def _get_scipy_param(self) -> dict:
        mu = self._bc_params["mu"]

        return [mu], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "mu": 0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
