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
        "capability:approx": ["pdfnorm"],
        "capability:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "capability:update": True,
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self._mu = mu
        self._sigma = sigma

        super().__init__(index=index, columns=columns)

    @property
    def mu(self):
        """Mean of the distribution."""
        return pd.DataFrame(self._mu, index=self.index, columns=self.columns)

    @property
    def sigma(self):
        """Standard deviation of the distribution."""
        return pd.DataFrame(self._sigma, index=self.index, columns=self.columns)

    def mean(self):
        """Return the mean of the distribution."""
        return self.mu

    def var(self):
        """Return the variance of the distribution."""
        res = np.array(self._sigma) ** 2
        return pd.DataFrame(res, index=self.index, columns=self.columns)

    def _update(self, data, obs_sigma=1.0):
        """Update Normal distribution via Normal-Normal conjugate prior."""
        # Convert inputs to numpy for stable math
        x = np.array(data)
        mu_0 = np.array(self._mu)
        sigma_0 = np.array(self._sigma)
        
        # Precision math
        tau_0 = 1 / (sigma_0**2)
        tau_obs = 1 / (obs_sigma**2)
        n = x.size
        
        # Posterior updates
        tau_post = tau_0 + n * tau_obs
        # We use _mu and _sigma directly to bypass the property setter error
        self._mu = (tau_0 * mu_0 + tau_obs * np.sum(x)) / tau_post
        self._sigma = np.sqrt(1 / tau_post)

        # Trigger skpro's internal metadata refresh if necessary
        if hasattr(self, "_init_shape_bc"):
            self._init_shape_bc(index=self.index, columns=self.columns)

        return self

    def _pdf(self, x):
        """Probability density function."""
        mu = np.array(self._mu)
        sigma = np.array(self._sigma)
        pdf_arr = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return pdf_arr / (sigma * np.sqrt(2 * np.pi))

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        mu = np.array(self._mu)
        sigma = np.array(self._sigma)
        lpdf_arr = -0.5 * ((x - mu) / sigma) ** 2
        return lpdf_arr - np.log(sigma * np.sqrt(2 * np.pi))

    def _cdf(self, x):
        """Cumulative distribution function."""
        mu = np.array(self._mu)
        sigma = np.array(self._sigma)
        return 0.5 + 0.5 * erf((x - mu) / (sigma * np.sqrt(2)))

    def _ppf(self, p):
        """Quantile function."""
        mu = np.array(self._mu)
        sigma = np.array(self._sigma)
        return mu + sigma * np.sqrt(2) * erfinv(2 * p - 1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {"mu": 0, "sigma": 1, "index": pd.Index([1, 2, 5]), "columns": pd.Index(["a", "b"])}
        params3 = {"mu": 1, "sigma": 2}
        return [params1, params2, params3]