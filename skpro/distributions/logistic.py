# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Logistic probability distribution."""

__author__ = ["malikrafsan"]

import numpy as np
import pandas as pd
from scipy.integrate import quad

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

    Examples
    --------
    >>> from skpro.distributions.logistic import Logistic

    >>> l = Logistic(mu=[[0, 1], [2, 3], [4, 5]], scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
            "energy",
        ],
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

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        Closed-form formula: \mathbb{E}|X-Y| = 2s for Logistic(mu, s).

        Derivation: E|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t)) dt
        where F(t) = 1/(1+exp(-(t-mu)/s)). Using the identity
        F(t)(1-F(t)) = 1/(4 cosh^2((t-mu)/(2s))), we get
        2 \int_{-\infty}^{\infty} 1/(4 cosh^2((t-mu)/(2s))) dt = s.
        Therefore E|X-Y| = 2s.
        """
        scale = self._bc_params["scale"]
        energy_arr = 2 * scale
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        Uses numerical integration:
        \\mathbb{E}|X - x| = \\int_{-\\infty}^{\\infty} |t - x| f(t) dt.
        """
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]

        def energy_cell(m, s, xi):
            # Compute E|X - xi| by integrating |t - xi| * f(t)
            # Logistic PDF: f(t) = 1/(4*s) * sech^2((t-m)/(2*s))
            pdf = lambda t: 1 / (4 * s) / np.cosh((t - m) / (2 * s)) ** 2  # noqa: E731

            # Split integral at xi
            lower, _ = quad(lambda t: (xi - t) * pdf(t), m - 10 * s, xi, limit=200)
            upper, _ = quad(lambda t: (t - xi) * pdf(t), xi, m + 10 * s, limit=200)
            return lower + upper

        vec_energy = np.vectorize(energy_cell)
        energy_arr = vec_energy(mu, scale, x)
        if np.ndim(energy_arr) > 1:
            energy_arr = energy_arr.sum(axis=1)
        return energy_arr

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
