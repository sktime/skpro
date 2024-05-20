# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Student's t-distribution."""

__author__ = ["Alex-JG3", "ivarzap"]

import numpy as np
import pandas as pd
from scipy.special import betaincinv, gamma, hyp2f1, loggamma

from skpro.distributions.base import BaseDistribution


class TDistribution(BaseDistribution):
    """Student's t-distribution (skpro native).

    Parameters
    ----------
    mean : float or array of float (1D or 2D)
        mean of the t-distribution distribution
    sd : float or array of float (1D or 2D), must be positive
        standard deviation of the t-distribution distribution
    df : float or array of float (1D or 2D), must be positive
        Degrees of freedom of the t-distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.t import TDistribution

    >>> n = TDistribution(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, df=10)
    """

    _tags = {
        "authors": ["Alex-JG3"],
        "maintainers": ["Alex-JG3"],
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, df=1, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.df = df

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        mean_arr = self._bc_params["mu"]
        df = self._bc_params["df"]

        if self.ndim == 0:
            if df <= 1:
                return np.inf
            return mean_arr

        if (df <= 1).any():
            mean_arr = mean_arr.astype(np.float32)
            mean_arr[df <= 1] = np.inf
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns,

        .. math::
            \mathbb{V}[X] = \begin{cases}
                \frac{\nu}{\nu - 2} & \text{if} \nu > 2, \\
                \infty              & \text{if} \nu \le 2, \\
            \begin{cases}

        Where :math:`\nu` is the degrees of freedom of the t-distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        sigma = self._bc_params["sigma"]
        df = self._bc_params["df"]
        df_arr = df.astype(np.float32)

        if self.ndim == 0:
            if df <= 2:
                return np.inf
            return sigma**2 * df / (df - 2)

        df_arr[df_arr <= 2] = np.inf
        mask = (df_arr > 2) & (df_arr != np.inf)
        df_arr[mask] = sigma[mask] ** 2 * df_arr[mask] / (df_arr[mask] - 2)
        return df_arr

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
        df = self._bc_params["df"]
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        pdf_arr = gamma((df + 1) / 2)
        pdf_arr = pdf_arr / (np.sqrt(np.pi * df) * gamma(df / 2))
        pdf_arr = pdf_arr * (1 + ((x - mu) / sigma) ** 2 / df) ** (-(df + 1) / 2)
        pdf_arr = pdf_arr / sigma
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
        df = self._bc_params["df"]
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        lpdf_arr = loggamma((df + 1) / 2)
        lpdf_arr = lpdf_arr - 0.5 * np.log(df * np.pi)
        lpdf_arr = lpdf_arr - loggamma(df / 2)
        lpdf_arr = lpdf_arr - ((df + 1) / 2) * np.log(1 + ((x - mu) / sigma) ** 2 / df)
        lpdf_arr = lpdf_arr - np.log(sigma)
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
        df = self._bc_params["df"]
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        x_ = (x - mu) / sigma
        cdf_arr = x_ * gamma((df + 1) / 2)
        cdf_arr = cdf_arr * hyp2f1(0.5, (df + 1) / 2, 3 / 2, -(x_**2) / df)
        cdf_arr = 0.5 + cdf_arr / (np.sqrt(np.pi * df) * gamma(df / 2))
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
        df = self._bc_params["df"]
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        ppf_arr = p.copy()
        ppf_arr[p == 0.5] = 0.0
        ppf_arr[p <= 0] = -np.inf
        ppf_arr[p >= 1] = np.inf

        mask1 = (p < 0.5) & (p > 0)
        mask2 = (p < 1) & (p > 0.5)
        ppf_arr[mask1] = 1 / betaincinv(0.5 * df[mask1], 0.5, 2 * ppf_arr[mask1])
        ppf_arr[mask2] = 1 / betaincinv(0.5 * df[mask2], 0.5, 2 * (1 - ppf_arr[mask2]))
        ppf_arr[mask1 | mask2] = np.sqrt(ppf_arr[mask1 | mask2] - 1)
        ppf_arr[mask1 | mask2] = np.sqrt(df[mask1 | mask2]) * ppf_arr[mask1 | mask2]
        ppf_arr[mask1] = -ppf_arr[mask1]
        ppf_arr = sigma * ppf_arr + mu
        return ppf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {
            "mu": [[0, 1], [2, 3], [4, 5]],
            "sigma": 1,
            "df": [2, 3],
        }
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": -2, "sigma": 3, "df": 4}

        return [params1, params2, params3]
