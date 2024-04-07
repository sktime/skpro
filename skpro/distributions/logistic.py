# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Logistic probability distribution."""

__author__ = ["malikrafsan"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution

class Logistic(BaseDistribution):
    """Logistic distribution.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the logistic distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.logistic import Logistic

    >>> l = Logistic(mu=[[0, 1], [2, 3], [4, 5]], scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        self._mu, self._scale = self._get_bc_params(self.mu, self.scale)
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
        r"""Return variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the variance :math:`\mathbb{V}[X] = \frac{\mathbb{S}^2 \times \pi^3}{3}`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        var_arr = (self._scale**2 * np.pi**2) / 3
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)
    
    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        numerator = np.exp(-(x.values - d.mu) / d.scale)
        denumerator = d.scale * ((1 + np.exp(-(x.values - d.mu) / d.scale))**2)
        pdf_arr = numerator / denumerator
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        y = -(x.values - d.mu) / d.scale
        lpdf_arr = y - np.log(d.scale) - 2 * np.log(1 + np.exp(y))
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = (1 + np.tanh((x.values - d.mu) / (2 * d.scale))) / 2
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        ppf_arr = d.mu + d.scale * np.log(p.values / (1 - p.values))
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "scale": 1}
        params2 = {
            "mu": 0,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]

