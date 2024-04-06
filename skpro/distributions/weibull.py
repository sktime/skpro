# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Weibull probability distribution."""

__author__ = ["malikrafsan"]

import numpy as np
import pandas as pd
from scipy.special import gamma

from skpro.distributions.base import BaseDistribution

class Weibull(BaseDistribution):
    """Weibull distribution.

    Parameters
    ----------
    lambda_val : float or array of float (1D or 2D), must be positive
        lambda parameter of the distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.weibull import Weibull

    >>> w = Weibull(lambda_val=[[0, 1], [2, 3], [4, 5]], scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, lambda_val, scale, index=None, columns=None):
        self.lambda_val = lambda_val
        self.scale = scale
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        self._lambda_val, self._scale = self._get_bc_params(self.lambda_val, self.scale)
        shape = self._lambda_val.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns)

    def energy(self, x=None):
        raise NotImplementedError

    def mean(self):
        r"""Return expected value of the distribution.
        
        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\lambda \Gamma (1+\frac{1}{k})`
        
        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr = self._scale * gamma(1 + 1 / self._lambda_val)
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\lambda^2 \left( \Gamma(1+\frac{2}{k}) - \Gamma^2(1+\frac{1}{k}) \right)`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        left_gamma = gamma(1 + 2 / self._lambda_val)
        right_gamma = gamma(1 + 1 / self._lambda_val) ** 2
        var_arr = self._scale ** 2 * (left_gamma - right_gamma)
        return pd.DataFrame(var_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        # if x.values[i] < 0, then pdf_arr[i] = 0
        pdf_arr = (d.lambda_val / d.scale) * (x.values / d.scale) ** (d.lambda_val - 1) * np.exp(- (x.values / d.scale) ** d.lambda_val)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = np.log(d.lambda_val / d.scale) + (d.lambda_val - 1) * np.log(x.values / d.scale) - (x.values / d.scale) ** d.lambda_val
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        # if x.values[i] < 0, then cdf_arr[i] = 0
        cdf_arr = 1 - np.exp(-(x.values / d.scale) ** d.lambda_val)
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        ppf_arr = d.scale * (-np.log(1 - p.values)) ** (1 / d.lambda_val)
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"lambda_val": [[0, 1], [2, 3], [4, 5]], "scale": 1}
        params2 = {
            "lambda_val": 1,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
