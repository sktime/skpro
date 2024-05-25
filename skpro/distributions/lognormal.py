# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Log-Normal probability distribution."""

__author__ = ["bhavikar04", "fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv

from skpro.distributions.base import BaseDistribution


class LogNormal(BaseDistribution):
    r"""Log-Normal distribution (skpro native).

    Parameterized by mean and standard deviation of the logarithm of the distribution,
    :math:`\mu` and :math:`\sigma`, respectively, such that the cdf

    .. math:: F(x) = \frac{1}{2} + \frac{1}{2} \text{erf}\left(\frac{\log(x) - \mu}{\sigma \sqrt{2}}\right)  # noqa E501

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the logarithm of the distribution, :math:`\mu` above
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation the logarithm of the distribution, :math:`\sigma` above
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.lognormal import LogNormal

    >>> n = LogNormal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    """

    _tags = {
        "authors": ["bhavikar04", "fkiraly"],
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    # commented out, seems incorrect
    # def energy(self, x=None):
    #     r"""Energy of self, w.r.t. self or a constant frame x.

    #     Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

    #     If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (per row), "self-energy".
    #     If `x` is passed, returns :math:`\mathbb{E}[|X-x|]-0.5\mathbb{E}[|X-Y|]`
    #     (per row), "CRPS wrt x".

    #     Parameters
    #     ----------
    #     x : None or pd.DataFrame, optional, default=None
    #         if pd.DataFrame, must have same rows and columns as `self`

    #     Returns
    #     -------
    #     pd.DataFrame with same rows as `self`, single column `"energy"`
    #     each row contains one float, self-energy/energy as described above.
    #     """
    #     if x is None:
    #         return super().energy(x)
    #     # explicit formula for CRPS of log-normal cross-term
    #     # obtained by bhavikar04 via wolfram alpha
    #     else:
    #         d = self.loc[x.index, x.columns]
    #         mu_arr, sd_arr = d._mu, d._sigma
    #         c_arr = x * (2 * self.cdf(x) - 1)
    #         c_arr2 = -2 * np.exp((mu_arr + sd_arr**2) / 2)
    #         c_arr3 = self.cdf((np.log(x) - mu_arr - sd_arr**2) / sd_arr)
    #         c_arr3 = c_arr3 + self.cdf(sd_arr / mu_arr**0.5) - 1
    #         c_arr2 = c_arr2 * c_arr3
    #         c_arr = c_arr + c_arr2

    #         energy_arr = np.sum(c_arr, axis=1)
    #         energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
    #     return energy

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        mean_arr = np.exp(mu + sigma**2 / 2)
        return mean_arr

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        sd_arr = np.exp(2 * mu + 2 * sigma**2) - np.exp(2 * mu + sigma**2)
        return sd_arr

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
        sigma = self._bc_params["sigma"]

        pdf_arr = np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
        pdf_arr = pdf_arr / (x * sigma * np.sqrt(2 * np.pi))
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
        sigma = self._bc_params["sigma"]

        lpdf_arr = -0.5 * ((np.log(x) - mu) / sigma) ** 2
        lpdf_arr = lpdf_arr - np.log(x * sigma * np.sqrt(2 * np.pi))
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
        sigma = self._bc_params["sigma"]

        cdf_arr = 0.5 + 0.5 * erf((np.log(x) - mu) / (sigma * np.sqrt(2)))
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
        sigma = self._bc_params["sigma"]

        icdf_arr = mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)
        icdf_arr = np.exp(icdf_arr)
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": -2, "sigma": 2}

        return [params1, params2, params3]
