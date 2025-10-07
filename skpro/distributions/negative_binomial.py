# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Negative binomial probability distribution."""

__author__ = ["tingiskhan"]

import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import nbinom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class NegativeBinomial(_ScipyAdapter):
    """Negative binomial distribution.

    Most methods wrap ``scipy.stats.nbinom``.

    Parameters
    ----------
    mu : ArrayLike
        mean of the distribution.
    alpha: ArrayLike
        dispersion of distribution.

    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import NegativeBinomial

    >>> distr = NegativeBinomial(mu=1.0, alpha=1.0)
    """

    _tags = {
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu: ArrayLike, alpha: ArrayLike, index=None, columns=None):
        self.mu = mu
        self.alpha = alpha

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return nbinom

    def _get_scipy_param(self) -> dict:
        mu = self._bc_params["mu"]
        alpha = self._bc_params["alpha"]

        n = alpha
        p = alpha / (alpha + mu)

        return [n, p], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]], "alpha": 2.0}
        params2 = {
            "mu": 1.0,
            "alpha": 2.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
