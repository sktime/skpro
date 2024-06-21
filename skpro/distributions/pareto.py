# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Pareto probability distribution."""

__author__ = ["sukjingitsit"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Pareto(BaseDistribution):
    r"""Pareto distribution (skpro native).

    The scale is represented by the parameter ``scale``,
    and the Pareto index (or shape parameter) :math:`\alpha`
    by the parameter ``alpha``.

    The CDF can be represented as,
    :math:`F(x) = 1-\left(\frac{\text{scale}}{x}\right)^\alpha
    \text{ if } x>0, 0 \text{ if } x<0`

    Parameters
    ----------
    scale : float or array of float (1D or 2D), must be positive
        scale of the Pareto distribution
    alpha : float or array of float (1D or 2D), must be positive
        shape of the Pareto distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.pareto import Pareto

    >>> n = Pareto(scale=[[1, 1.5], [2, 2.5], [3, 4]], alpha=3)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, scale, alpha, index=None, columns=None):
        self.scale = scale
        self.alpha = alpha

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        scale = self._bc_params["scale"]
        mean = np.where(alpha <= 1, np.infty, scale**alpha / (alpha - 1))
        return mean

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        scale = self._bc_params["scale"]
        var = np.where(
            alpha <= 2, np.infty, scale**2 * alpha / ((alpha - 2) * (alpha - 1) ** 2)
        )
        return var

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
        scale = self._bc_params["scale"]
        pdf_arr = alpha * np.power(scale, alpha)
        pdf_arr /= np.power(x, alpha + 1)
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
        scale = self._bc_params["scale"]
        return np.log(alpha / x) + alpha * np.log(scale / x)

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
        scale = self._bc_params["scale"]
        cdf_arr = np.where(x < scale, 0, 1 - np.power(scale / x, alpha))
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
        scale = self._bc_params["scale"]
        return scale / np.power(1 - p, 1 / alpha)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"scale": [[1, 1.5], [2, 3], [4, 5]], "alpha": 3}
        params2 = {
            "scale": 1,
            "alpha": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"scale": 1, "alpha": 2}
        return [params1, params2, params3]
