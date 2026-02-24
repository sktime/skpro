# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Normal/Gaussian probability distribution."""

__author__ = ["fkiraly", "aliviahossain"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv

from skpro.distributions.base import BaseDistribution


class Normal(BaseDistribution):
    """Normal distribution.

    Parameters
    ----------
    mu : float or array-like
        Mean of the normal distribution.
    sigma : float or array-like
        Standard deviation of the normal distribution.
    index : pd.Index, optional
        Index of the distribution object.
    columns : pd.Index, optional
        Column names of the distribution object.
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "capabilities:update": True,
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        # Store the raw values in private attributes
        self._mu = mu
        self._sigma = sigma

        # Call the base constructor
        super().__init__(index=index, columns=columns)

    @property
    def mu(self):
        """Mean of the distribution."""
        # This ensures that whenever someone calls dist.mu, 
        # it is returned in the correct skpro format
        return pd.DataFrame(self._mu, index=self.index, columns=self.columns)

    @property
    def sigma(self):
        """Standard deviation of the distribution."""
        # This solves the 'no setter' error because we assigned to self._sigma above
        return pd.DataFrame(self._sigma, index=self.index, columns=self.columns)

    def mean(self):
        """Return the mean of the distribution, bypassing skpro boilerplate."""
        # This prevents the AttributeError by not looking for _bc_params
        return pd.DataFrame(self.mu, index=self.index, columns=self.columns)

    def var(self):
        """Return the variance of the distribution, bypassing skpro boilerplate."""
        res = np.array(self.sigma) ** 2
        return pd.DataFrame(res, index=self.index, columns=self.columns)

    def _update(self, data, obs_sigma=1.0):
        """Update Normal distribution via Normal-Normal conjugate prior."""
        x = np.array(data)
        n = x.size
        sum_x = np.sum(x)

        mu_0 = np.array(self.mu)
        sigma_0 = np.array(self.sigma)
        
        tau_0 = 1 / (sigma_0**2)
        tau_obs = 1 / (obs_sigma**2)

        tau_post = tau_0 + n * tau_obs
        mu_post = (tau_0 * mu_0 + tau_obs * sum_x) / tau_post
        sigma_post = np.sqrt(1 / tau_post)

        self.mu = mu_post
        self.sigma = sigma_post

        # Update shape metadata
        self._init_shape_bc(index=self.index, columns=self.columns)

        return self

    def _pdf(self, x):
        """Probability density function."""
        # Falling back to direct attributes if _bc_params is missing
        mu = getattr(self, "_bc_params", {"mu": self.mu})["mu"]
        sigma = getattr(self, "_bc_params", {"sigma": self.sigma})["sigma"]
        pdf_arr = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return pdf_arr / (sigma * np.sqrt(2 * np.pi))

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        mu = getattr(self, "_bc_params", {"mu": self.mu})["mu"]
        sigma = getattr(self, "_bc_params", {"sigma": self.sigma})["sigma"]
        lpdf_arr = -0.5 * ((x - mu) / sigma) ** 2
        return lpdf_arr - np.log(sigma * np.sqrt(2 * np.pi))

    def _cdf(self, x):
        """Cumulative distribution function."""
        mu = getattr(self, "_bc_params", {"mu": self.mu})["mu"]
        sigma = getattr(self, "_bc_params", {"sigma": self.sigma})["sigma"]
        return 0.5 + 0.5 * erf((x - mu) / (sigma * np.sqrt(2)))

    def _ppf(self, p):
        """Quantile function."""
        mu = getattr(self, "_bc_params", {"mu": self.mu})["mu"]
        sigma = getattr(self, "_bc_params", {"sigma": self.sigma})["sigma"]
        return mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {"mu": 0, "sigma": 1, "index": pd.Index([1, 2, 5]), "columns": pd.Index(["a", "b"])}
        params3 = {"mu": 1, "sigma": 2}
        return [params1, params2, params3]