# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-logistic aka Fisk probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import fisk, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Fisk(_ScipyAdapter):
    r"""Fisk distribution, also known as the log-logistic distribution.

    Most methods wrap ``scipy.stats.fisk``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The Fisk (log-logistic) distribution is parameterized by a scale
    parameter :math:`\alpha > 0` and a shape parameter :math:`\beta > 0`,
    such that the cumulative distribution function (CDF) is given by:

    .. math:: F(x) = \frac{1}{1 + \left(\frac{x}{\alpha}\right)^{-\beta}}
        = \frac{(x / \alpha)^{\beta}}{1 + (x / \alpha)^{\beta}}, \quad x \geq 0

    The probability density function (PDF) is:

    .. math::

        f(x) = \frac{(\beta / \alpha)\,(x/\alpha)^{\beta-1}}
                    {\left[1 + (x/\alpha)^{\beta}\right]^2}, \quad x \geq 0

    The mean is defined only for :math:`\beta > 1` and is:

    .. math:: \mathbb{E}[X] = \frac{\alpha \pi / \beta}{\sin(\pi / \beta)}

    The variance is defined only for :math:`\beta > 2` and is:

    .. math:: \mathrm{Var}[X] = \alpha^2 \left(
            \frac{2\pi/\beta}{\sin(2\pi/\beta)}
            - \frac{(\pi/\beta)^2}{\sin^2(\pi/\beta)}
        \right)

    The scale parameter :math:`\alpha` is represented by the parameter ``alpha``,
    and the shape parameter :math:`\beta` by the parameter ``beta``.

    Parameters
    ----------
    alpha : float or array of float (1D or 2D), must be positive
        Scale parameter of the distribution.
    beta : float or array of float (1D or 2D), must be positive
        Shape parameter of the distribution.
        Mean is defined only for ``beta > 1``.
        Variance is defined only for ``beta > 2``.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.fisk import Fisk

    >>> d = Fisk(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "malikrafsan"],
        # estimator tags
        # --------------
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha=1, beta=1, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return fisk

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        return [], {"c": beta, "scale": alpha}

    def _mean(self):
        r"""Return expected value of the distribution.

        The mean is defined only when :math:`\beta > 1` and equals:

        .. math:: \mathbb{E}[X] = \frac{\alpha \pi / \beta}{\sin(\pi / \beta)}

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        # mean = alpha * (pi/beta) / sin(pi/beta)
        pb = np.pi / beta
        mean_arr = alpha * pb / np.sin(pb)
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        The variance is defined only when :math:`\beta > 2` and equals:

        .. math:: \mathrm{Var}[X] = \alpha^2 \left(
                \frac{2\pi/\beta}{\sin(2\pi/\beta)}
                - \frac{(\pi/\beta)^2}{\sin^2(\pi/\beta)}
            \right)

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of distribution (entry-wise)
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        pb = np.pi / beta
        two_pb = 2 * np.pi / beta
        var_arr = alpha**2 * (two_pb / np.sin(two_pb) - (pb / np.sin(pb)) ** 2)
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

        u = x / alpha
        pdf_arr = np.where(
            x > 0,
            (beta / alpha) * u ** (beta - 1) / (1 + u**beta) ** 2,
            0.0,
        )
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the log-pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log-pdf values at the given points
        """
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        u = x / alpha
        log_pdf_arr = np.where(
            x > 0,
            np.log(beta / alpha)
            + (beta - 1) * np.log(u)
            - 2 * np.log(1 + u**beta),
            -np.inf,
        )
        return log_pdf_arr

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

        u = x / alpha
        cdf_arr = np.where(
            x > 0,
            u**beta / (1 + u**beta),
            0.0,
        )
        return cdf_arr

    def _ppf(self, p):
        """Percent point function = quantile function = inverse cdf.

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

        # Analytically: quantile = alpha * (p / (1 - p))^(1/beta)
        ppf_arr = alpha * (p / (1 - p)) ** (1 / beta)
        return ppf_arr

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
