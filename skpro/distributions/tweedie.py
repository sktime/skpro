# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tweedie probability distribution."""

__author__ = ["ShreeshaM07"]

import numpy as np

from skpro.distributions.base import BaseDistribution

# import pandas as pd


class Tweedie(BaseDistribution):
    """Tweedie Distribution.

    Parameters
    ----------
    pw : float or array of float (1D or 2D)
        Power parameter
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

    def __init__(self, pw, mu, scale, index=None, columns=None):
        # from skpro.distributions.gamma import Gamma
        from skpro.distributions.normal import Normal
        from skpro.distributions.poisson import Poisson

        self.pw = pw
        self.mu = mu
        self.scale = scale

        if pw == 0:
            self._norm = Normal(mu=mu, sigma=scale, index=index, columns=columns)
        elif pw == 1:
            self._pois = Poisson(mu=mu, index=index, columns=columns)
        # elif pw == 2:
        #     self._gam = Gamma(alpha=alpha,beta=beta,index=index,columns=columns)

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

        pw = self.pw
        mu = self.mu
        scale = self.scale
        if pw == 0:
            return self._norm.pdf(x)
        elif pw > 1 and pw < 2:
            theta = np.power(mu, 1 - pw) / (1 - pw)
            kappa = np.power(mu, 2 - pw) / (2 - pw)
            alpha = (2 - pw) / (1 - pw)
            t = ((pw - 1) * scale / x) ** alpha
            t /= (2 - pw) * scale
            a = 1 / x * wright_bessel(-alpha, 0, t)
            return a * np.exp((x * theta - kappa) / scale)

    def _pmf(self, x):
        """Probability mass function.

        Private method, to be implemented by subclasses.
        """
        pw = self.pw
        mu = self.mu
        scale = self.scale
        if pw == 1:
            return self._pois.pdf(x)
        elif pw > 1 and pw < 2:
            return np.exp(-np.power(mu, 2 - pw) / (scale * (2 - pw)))

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

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Private method, to be implemented by subclasses.
        """

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
