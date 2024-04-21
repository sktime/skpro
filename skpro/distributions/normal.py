# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Normal/Gaussian probability distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv

from skpro.distributions.base import BaseDistribution


class Normal(BaseDistribution):
    """Normal distribution (skpro native).

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the normal distribution
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.normal import Normal

    >>> n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.index = index
        self.columns = columns

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        self._mu, self._sigma = self._get_bc_params(self.mu, self.sigma)

        super().__init__(index=index, columns=columns)

    def _energy_self(self):
        sigma = self._bc_params["sigma"]
        energy_arr = 2 * sigma / np.sqrt(np.pi)
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _energy_x(self, x=None):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]

        cdf = self.cdf(x)
        pdf = self.pdf(x)
        energy_arr = (x - mu) * (2 * cdf - 1) + 2 * sigma**2 * pdf
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._bc_params["mu"]

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        return self._sigma**2

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
        pdf_arr = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        pdf_arr = pdf_arr / (sigma * np.sqrt(2 * np.pi))
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

        lpdf_arr = -0.5 * ((x - mu) / sigma) ** 2
        lpdf_arr = lpdf_arr - np.log(sigma * np.sqrt(2 * np.pi))
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

        cdf_arr = 0.5 + 0.5 * erf((x - mu) / (sigma * np.sqrt(2)))
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
        return icdf_arr

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
