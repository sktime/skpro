# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Logistic probability distribution."""

__author__ = ["malikrafsan"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Logistic(BaseDistribution):
    r"""Logistic distribution.

    The logistic distribution is parametrized by a mean parameter :math:`\mu`,
    and scale parameter :math:`s`, such that the cdf is given by:

    .. math:: F(x) = \frac{1}{1 + \exp\left(\frac{x - \mu}{s}\right)}

    The scale parameter :math:`s` is represented by the parameter ``scale``,
    and the mean parameter :math:`\mu` by the parameter ``mu``.

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
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._bc_params["mu"]

    def _var(self):
        r"""Return variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the variance :math:`\mathbb{V}[X] = \frac{\mathbb{S}^2 \times \pi^3}{3}`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        scale = self._bc_params["scale"]
        var_arr = (scale**2 * np.pi**2) / 3
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
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        numerator = np.exp(-(x - mu) / scale)
        denominator = scale * ((1 + np.exp(-(x - mu) / scale)) ** 2)
        pdf_arr = numerator / denominator
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
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        y = -(x - mu) / scale
        lpdf_arr = y - np.log(scale) - 2 * np.logaddexp(0, y)
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
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        cdf_arr = (1 + np.tanh((x - mu) / (2 * scale))) / 2
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
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        ppf_arr = mu + scale * np.log(p / (1 - p))
        return ppf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "scale": 1}
        params2 = {
            "mu": 0,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": -2, "scale": 2}

        return [params1, params2, params3]
