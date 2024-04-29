# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-logistic aka Fisk probability distribution."""

__author__ = ["fkiraly"]

import pandas as pd
from scipy.stats import fisk

from skpro.distributions.base import BaseDistribution


class Fisk(BaseDistribution):
    r"""Fisk distribution, aka log-logistic distribution.

    The Fisk distribution is parametrized by a scale parameter :math:`\alpha`
    and a shape parameter :math:`\beta`, such that the cumulative distribution
    function (CDF) is given by:

    .. math:: F(x) = 1 - \left(1 + \frac{x}{\alpha}\right)^{-\beta}\right)^{-1}

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
    >>> from skpro.distributions.fisk import FiskScipy as Fisk

    >>> d = Fisk(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, alpha=1, beta=1, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        mean_arr = fisk.mean(scale=alpha, c=beta)
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        var_arr = fisk.var(scale=alpha, c=beta)
        return var_arr

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
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        pdf_arr = fisk.pdf(x, scale=alpha, c=beta)
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
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        lpdf_arr = fisk.logpdf(x, scale=alpha, c=beta)
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
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        cdf_arr = fisk.cdf(x, scale=alpha, c=beta)
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
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        icdf_arr = fisk.ppf(p, scale=alpha, c=beta)
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"alpha": [[1, 1], [2, 3], [4, 5]], "beta": 3}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
