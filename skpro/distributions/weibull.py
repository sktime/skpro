# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Weibull probability distribution."""

__author__ = ["malikrafsan"]

import numpy as np
import pandas as pd
from scipy.special import gamma

from skpro.distributions.base import BaseDistribution


class Weibull(BaseDistribution):
    r"""Weibull distribution.

    The Weibull distribution is parametrized by scale parameter :math:`\lambda`,
    and shape parameter :math:`k`, such that the cdf is given by:

    .. math:: F(x) = 1 - \exp\left(-\left(\frac{x}{\lambda}\right)^k\right)

    The scale parameter :math:`\lambda` is represented by the parameter ``scale``,
    and the shape parameter :math:`k` by the parameter ``k``.

    Parameters
    ----------
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    k : float or array of float (1D or 2D), must be positive
        shape parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.weibull import Weibull

    >>> w = Weibull(scale=[[1, 1], [2, 3], [4, 5]], k=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, scale, k, index=None, columns=None):
        self.scale = scale
        self.k = k

        super().__init__(index=index, columns=columns)

    def _mean(self):
        r"""Return expected value of the distribution.

        For Weibull distribution, expectation is given by,
        :math:`\lambda \Gamma (1+\frac{1}{k})`

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        scale = self._bc_params["scale"]
        k = self._bc_params["k"]
        mean_arr = scale * gamma(1 + 1 / k)
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        For Weibull distribution, variance is given by
        :math:`\lambda^2 \left( \Gamma(1+\frac{2}{k}) - \Gamma^2(1+\frac{1}{k}) \right)`

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        scale = self._bc_params["scale"]
        k = self._bc_params["k"]

        left_gamma = gamma(1 + 2 / k)
        right_gamma = gamma(1 + 1 / k) ** 2
        var_arr = scale**2 * (left_gamma - right_gamma)
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
        k = self._bc_params["k"]
        scale = self._bc_params["scale"]

        pdf_arr = (k / scale) * (x / scale) ** (k - 1) * np.exp(-((x / scale) ** k))
        pdf_arr = pdf_arr * (x >= 0)  # if x < 0, pdf = 0
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
        k = self._bc_params["k"]
        scale = self._bc_params["scale"]

        lpdf_arr = np.log(k / scale) + (k - 1) * np.log(x / scale) - (x / scale) ** k
        lpdf_arr = lpdf_arr + np.log(x >= 0)  # if x < 0, pdf = 0, so log pdf = -inf
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
        k = self._bc_params["k"]
        scale = self._bc_params["scale"]

        cdf_arr = 1 - np.exp(-((x / scale) ** k))
        cdf_arr = cdf_arr * (x >= 0)  # if x < 0, cdf = 0
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
        k = self._bc_params["k"]
        scale = self._bc_params["scale"]

        ppf_arr = scale * (-np.log(1 - p)) ** (1 / k)
        return ppf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"scale": [[1, 1], [2, 3], [4, 5]], "k": 1}
        params2 = {
            "scale": 1,
            "k": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"scale": 2, "k": 3}

        return [params1, params2, params3]
