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
    pow : float or array of float (1D or 2D)
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

    def __init__(self, pow, mu, scale, index=None, columns=None):
        from skpro.distributions.cmp_pois_gam import CmpPoissonGamma
        from skpro.distributions.gamma import Gamma
        from skpro.distributions.normal import Normal
        from skpro.distributions.poisson import Poisson

        self.pow = pow
        self.mu = mu
        self.scale = scale
        mu = np.array(mu)
        scale = np.array(scale)
        if pow == 0:
            self._norm = Normal(mu=mu, sigma=scale, index=index, columns=columns)
        elif pow == 1:
            self._pois = Poisson(mu=mu, index=index, columns=columns)
        elif pow > 1 and pow < 2:
            self._cmp_pg = CmpPoissonGamma(
                pow=pow, mu=mu, scale=scale, index=index, columns=columns
            )
        elif pow == 2:
            alpha = (mu / scale) ** 2
            beta = mu / scale**2
            self._gam = Gamma(alpha=alpha, beta=beta, index=index, columns=columns)

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
        pow = self.pow

        if pow == 0:
            return self._norm.pdf(x)
        elif pow == 1:
            return self._pois.pdf(x)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.pdf(x)
        elif pow == 2:
            return self._gam.pdf(x)

    def _pmf(self, x):
        """Probability mass function.

        Private method, to be implemented by subclasses.
        """
        pow = self.pow

        if pow == 0:
            return self._norm.pmf(x)
        elif pow == 1:
            return self._pois.pmf(x)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.pmf(x)
        elif pow == 2:
            return self._gam.pmf(x)

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
        pow = self.pow
        if pow == 0:
            return self._norm.log_pdf(x)
        elif pow == 1:
            return self._pois.log_pdf(x)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.log_pdf(x)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Private method, to be implemented by subclasses.
        """
        pow = self.pow
        if pow == 0:
            return self._norm.log_pmf(x)
        elif pow == 1:
            return self._pois.log_pmf(x)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.log_pmf(x)

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
        pow = self.pow

        if pow == 0:
            return self._norm.cdf(x)
        elif pow == 1:
            return self._pois.cdf(x)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.cdf(x)
        elif pow == 2:
            return self._gam.cdf(x)

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
        pow = self.pow

        if pow == 0:
            return self._norm.ppf(p)
        elif pow == 1:
            return self._pois.ppf(p)
        elif pow > 1 and pow < 2:
            return self._cmp_pg.ppf(p)
        elif pow == 2:
            return self._gam.ppf(p)
