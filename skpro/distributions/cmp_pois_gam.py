# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tweedie probability distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.distributions.base import BaseDistribution

# import pandas as pd


class CmpPoissonGamma(BaseDistribution):
    """Compound Poisson Gamma Distribution.

    Parameters
    ----------
    pow : float or array of float (1D or 2D)
        Power parameter should be in range (1,2)
    mu : float or array of float (1D or 2D)
        mean of the normal distribution
    scale : float or array of float (1D or 2D)
        scale parameter
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "energy",
            "pdf",
            "log_pdf",
            "cdf",
            "ppf",
            "pmf",
            "log_pmf",
        ],
        "distr:measuretype": "mixed",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, pow, mu, scale, index=None, columns=None):
        self.pow = pow
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

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
        from scipy.special import wright_bessel

        pow = self.pow
        mu = self.mu
        scale = self.scale

        theta = np.power(mu, 1 - pow) / (1 - pow)
        kappa = np.power(mu, 2 - pow) / (2 - pow)
        alpha = (2 - pow) / (1 - pow)
        t = ((pow - 1) * scale / x) ** alpha
        t /= (2 - pow) * scale
        a = 1 / x * wright_bessel(-alpha, 0, t)
        return a * np.exp((x * theta - kappa) / scale)

    def _pmf(self, x):
        """Probability mass function.

        Private method, to be implemented by subclasses.
        """
        pow = self.pow
        mu = self.mu
        scale = self.scale

        return np.exp(-np.power(mu, 2 - pow) / (scale * (2 - pow)))

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
        from scipy.integrate import quad

        cdf_val = np.array(quad(self._pdf, 0, x)[0])
        return cdf_val

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
        from scipy.optimize import brentq

        def objective(x, p):
            return self._cdf(x) - p

        ppf_val = np.array(
            brentq(objective, 0, 1000, args=(p,))
        )  # Adjust the upper bound as necessary
        return ppf_val
