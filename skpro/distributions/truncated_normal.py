# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Truncated Normal probability distribution."""

__author__ = ["ShreeshaM07"]

import pandas as pd
from scipy.stats import rv_continuous, truncnorm

from skpro.distributions.adapters.scipy import _ScipyAdapter


class TruncatedNormal(_ScipyAdapter):
    """A truncated normal probability distribution.

    Most methods wrap ``scipy.stats.truncnorm``.
    It truncates the normal distribution at
    the abscissa ``l_trunc`` and ``r_trunc``.

    Note: The truncation parameters passed
    is internally shifted to be centred at
    mean and scaled by sigma.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the normal distribution
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution
    l_trunc : float or array of float (1D or 2D)
        Left truncation abscissa.
    r_trunc : float or array of float (1D or 2D)
        Right truncation abscissa.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.truncated_normal import TruncatedNormal

    >>> d = TruncatedNormal(\
            mu=[[0, 1], [2, 3], [4, 5]],\
            sigma= 1,\
            l_trunc= [[-0.1,0.5],[1.5,2.4],[4.1,5]],\
            r_trunc= [[0.8,2],[4,5],[5,7]]\
        )
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, l_trunc, r_trunc, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.l_trunc = l_trunc
        self.r_trunc = r_trunc

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return truncnorm

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]
        l_trunc = self._bc_params["l_trunc"]
        r_trunc = self._bc_params["r_trunc"]

        # shift it to be centred at mu and sigma
        a = (l_trunc - mu) / sigma
        b = (r_trunc - mu) / sigma

        return [], {
            "loc": mu,
            "scale": sigma,
            "a": a,
            "b": b,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {
            "mu": [[0, 1], [2, 3], [4, 5]],
            "sigma": 1,
            "l_trunc": [[-0.1, 0.5], [1.5, 2.4], [4.1, 5]],
            "r_trunc": [[0.8, 2], [4, 5], [5, 7]],
        }
        params2 = {
            "mu": 0,
            "sigma": 1,
            "l_trunc": [-10, -5],
            "r_trunc": [5, 10],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1, "sigma": 2, "l_trunc": -3, "r_trunc": 5}
        return [params1, params2, params3]
