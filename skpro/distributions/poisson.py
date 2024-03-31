# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Poisson probability distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.stats import poisson

from skpro.distributions.base import BaseDistribution


class Poisson(BaseDistribution):
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
    }

    def __init__(self, mu, index=None, columns=None):
        self.mu = mu
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        # important: if only one argument, it is a lenght-1-tuple, deal with this
        self._mu = self._get_bc_params(self.mu)[0]
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns)

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr = self._mu
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        mean_arr = self._mu
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def pmf(self, x):
        """Probability mass function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = poisson.pmf(x.values, d.mu)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pmf(self, x):
        """Logarithmic probability mass function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = poisson.logpmf(x.values, d.mu)
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = poisson.cdf(x.values, d.mu)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        icdf_arr = poisson.ppf(p.values, d.mu)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

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
