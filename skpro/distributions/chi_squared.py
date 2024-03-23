# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Chi-Squared probability distribution."""

__author__ = ["sukjingitsit"]

import numpy as np
import pandas as pd
from scipy.special import chdtriv, gamma, gammainc

from skpro.distributions.base import BaseDistribution


class ChiSquared(BaseDistribution):
    """Chi-Squared distribution (skpro native).

    Parameters
    ----------
    dof : float or array of float (1D or 2D)
        degrees of freedom of the chi-squared distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    Example
    -------
    >>> from skpro.distributions.normal import ChiSquared
    >>> chi = ChiSquared(dof=[[1, 2], [3, 4], [5, 6]])
    """

    _tags = {
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, dof, index=None, columns=None):
        self.dof = dof
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        self._dof = self._get_bc_params(self.dof)[0]
        shape = self._dof.shape
        if index is None:
            index = pd.RangeIndex(shape[0])
        if columns is None:
            columns = pd.RangeIndex(shape[1])
        super().__init__(index=index, columns=columns)

    r"""Energy implementation issues:

    The self-energy is mathematically difficult to calculate due to
    their being no proper closed form. As discussed with fkiraly,
    using E(d.energy(x)) is one possible way, but the question arises
    on how to approximate the integral. The other alternative is to use
    sampling to estimate the self-energy.

    The closed form version for cross-energy can be framed as follows:
    Here, :math:`k=dof`
    :math:`x <= 0, \operatorname{energy}(x) = k + \vert x \vert`
    :math:`x > 0, \operatorname{energy}(x) =
    x*(2*\operatorname{CDF}(k,x)-1)+k-2k*\operatorname{CDF}(k+1,x)`
    where :math:`\operatorname{CDF}(k,x)` represents the CDF of x
    for a chi-square distribution with k degrees of freedom.
    """

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`
        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr = self._dof
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
        sd_arr = 2 * self._dof
        return pd.DataFrame(sd_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = np.exp(-x / 2) * np.power(x, (d.dof - 2) / 2)
        pdf_arr = pdf_arr / (np.power(2, d.dof / 2) * gamma(d.dof / 2))
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = -x / 2 + (d.dof - 2) * np.log(x) / 2
        lpdf_arr = lpdf_arr - (d.dof * np.log(2) / 2 + np.log(gamma(d.dof / 2)))
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        # cdf_arr = chdtr(d.dof, x)
        cdf_arr = gammainc(d.dof / 2, x / 2)
        cdf_arr = cdf_arr / (np.power(2, d.dof / 2) * gamma(d.dof / 2))
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        # Working on maths of native ppf
        d = self.loc[p.index, p.columns]
        icdf_arr = chdtriv(d.dof, p)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"dof": [[1, 2], [3, 4], [5, 6]]}
        params2 = {
            "dof": 10,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
