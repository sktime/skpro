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
        # packaging info
        # --------------
        "authors": "sukjingitsit",
        # estimator tags
        # --------------
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, dof, index=None, columns=None):
        self.dof = dof

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

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._bc_params["dof"]

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        return 2 * self._bc_params["dof"]

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        dof = self._bc_params["dof"]
        pdf_arr = np.exp(-x / 2) * np.power(x, (dof - 2) / 2)
        pdf_arr = pdf_arr / (np.power(2, dof / 2) * gamma(dof / 2))
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        dof = self._bc_params["dof"]
        lpdf_arr = -x / 2 + (dof - 2) * np.log(x) / 2
        lpdf_arr = lpdf_arr - (dof * np.log(2) / 2 + np.log(gamma(dof / 2)))
        return lpdf_arr

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        dof = self._bc_params["dof"]
        cdf_arr = gammainc(dof / 2, x / 2)
        cdf_arr = cdf_arr / (np.power(2, dof / 2) * gamma(dof / 2))
        return cdf_arr

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            ppf values at the given points
        """
        dof = self._bc_params["dof"]
        icdf_arr = chdtriv(dof, p)
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"dof": [[1, 2], [3, 4], [5, 6]]}
        params2 = {
            "dof": 10,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"dof": 3}
        return [params1, params2, params3]
