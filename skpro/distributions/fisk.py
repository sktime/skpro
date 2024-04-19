# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-logistic aka Fisk probability distribution."""

__author__ = ["fkiraly"]

import pandas as pd
from scipy.stats import fisk

from skpro.distributions.base import BaseDistribution


class Fisk(BaseDistribution):
    r"""Fisk distribution, aka log-logistic distribution.

    The Fisk distibution is parametrized by a scale parameter :math:`\alpha`
    and a shape parameter :math:`\beta`, such that the cumulative distribution
    function (CDF) is given by:

    .. math:: F(x) = \left(1 + \frac{x}{\alpha}\right)^{-\beta}\right)^{-1}

    Parameters
    ----------
    alpha : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    beta : float or array of float (1D or 2D), must be positive
        shape parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.fisk import Fisk

    >>> d = Fisk(mu=[[0, 1], [2, 3], [4, 5]], scale=1)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, alpha, beta, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        # important: if only one argument, it is a lenght-1-tuple, deal with this
        self._alpha, self._beta = self._get_bc_params(self.alpha, self.beta)
        shape = self._alpha.shape

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
        mean_arr = fisk.mean(scale=self._alpha, c=self._beta)
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
        var_arr = fisk.var(scale=self._alpha, c=self._beta)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = fisk.pdf(x.values, scale=d.alpha, c=d.beta)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = fisk.logpdf(x.values, scale=d.alpha, c=d.beta)
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = fisk.cdf(x.values, scale=d.alpha, c=d.beta)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        icdf_arr = fisk.ppf(x.values, scale=d.alpha, c=d.beta)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"alpha": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
