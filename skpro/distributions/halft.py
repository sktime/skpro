# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Student-t probability distribution."""

__author__ = ["OmAmbole009"]

import numpy as np
import pandas as pd
from scipy.stats import t as t_scipy

from skpro.distributions.base import BaseDistribution


class HalfT(BaseDistribution):
    r"""Half-Student-t probability distribution.

    The Half-Student-t distribution is a Student-t distribution truncated
    to non-negative values. It is a generalization of both the Half-Normal
    (as ``df`` -> infinity) and Half-Cauchy (when ``df = 1``) distributions.

    The Half-Student-t distribution is parametrized by degrees of freedom
    :math:`\nu` and scale :math:`\sigma`, such that the pdf is

    .. math:: f(x; \nu, \sigma) = \frac{2\,\Gamma\!\left(\frac{\nu+1}{2}\right)}
                {\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi\,\nu\,\sigma^2}}
                \left(1+\frac{x^2}{\nu\,\sigma^2}\right)^{-\frac{\nu+1}{2}},
                \quad x \ge 0

    The parameters ``df`` and ``sigma`` correspond to :math:`\nu` and
    :math:`\sigma` respectively.

    Parameters
    ----------
    df : float or array of float (1D or 2D), must be positive
        Degrees of freedom of the Half-Student-t distribution.
    sigma : float or array of float (1D or 2D), must be positive
        Scale parameter of the Half-Student-t distribution.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.halft import HalfT

    >>> d = HalfT(df=3, sigma=1)
    """

    _tags = {
        "authors": ["OmAmbole009"],
        "maintainers": ["OmAmbole009"],
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, df, sigma=1.0, index=None, columns=None):
        self.df = df
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _mean(self):
        r"""Return expected value of the distribution.

        For Half-t with df > 1:
        E[X] = 2 * sigma * sqrt(df / pi)
               * Gamma((df+1)/2) / (Gamma(df/2) * (df-1))
        Undefined for df <= 1.
        """
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        from scipy.special import gamma

        mean = (
            2
            * sigma
            * np.sqrt(df / np.pi)
            * gamma((df + 1) / 2)
            / (gamma(df / 2) * (df - 1))
        )
        # undefined for df <= 1
        mean = np.where(df > 1, mean, np.nan)
        return mean

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        For Half-t with df > 2:
        Var[X] = sigma^2 * (df/(df-2) - (2*df/pi) *
                 (Gamma((df+1)/2) / (Gamma(df/2) * (df-1)))^2)
        Undefined for df <= 2.
        """
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        from scipy.special import gamma

        ratio = gamma((df + 1) / 2) / (gamma(df / 2) * (df - 1))
        var = sigma**2 * (df / (df - 2) - (2 * df / np.pi) * ratio**2)
        # undefined for df <= 2
        var = np.where(df > 2, var, np.nan)
        return var

    def _pdf(self, x):
        """Probability density function."""
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        # Half-t pdf = 2 * t.pdf(x, df, loc=0, scale=sigma) for x >= 0
        pdf_arr = 2 * t_scipy.pdf(x, df, loc=0, scale=sigma)
        pdf_arr = np.where(x >= 0, pdf_arr, 0.0)
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        lpdf_arr = np.log(2) + t_scipy.logpdf(x, df, loc=0, scale=sigma)
        lpdf_arr = np.where(x >= 0, lpdf_arr, -np.inf)
        return lpdf_arr

    def _cdf(self, x):
        """Cumulative distribution function."""
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        # Half-t cdf = 2 * t.cdf(x, df, loc=0, scale=sigma) - 1 for x >= 0
        cdf_arr = 2 * t_scipy.cdf(x, df, loc=0, scale=sigma) - 1
        cdf_arr = np.where(x >= 0, cdf_arr, 0.0)
        return cdf_arr

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        df = self._bc_params["df"]
        sigma = self._bc_params["sigma"]
        # inverse of cdf: t.ppf((p+1)/2, df, loc=0, scale=sigma)
        return t_scipy.ppf((p + 1) / 2, df, loc=0, scale=sigma)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"df": 3, "sigma": 1.0}
        params2 = {
            "df": 10,
            "sigma": 2.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"df": [[3, 5], [7, 10]], "sigma": [[1.0, 1.5], [2.0, 2.5]]}
        return [params1, params2, params3]