"""Skellam probability distribution for skpro."""

from skpro.distributions.base import BaseDistribution
import numpy as np
from scipy.stats import skellam


class Skellam(BaseDistribution):
    """Skellam probability distribution.

    Parameters
    ----------
    mu1 : float
        Mean of the first Poisson distribution
    mu2 : float
        Mean of the second Poisson distribution
    """

    _tags = {
        "authors": ["your-github-id"],
        "distr:measuretype": "discrete",
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu1, mu2, index=None, columns=None):
        self.mu1 = mu1
        self.mu2 = mu2
        super().__init__(index=index, columns=columns)

    def _pmf(self, x):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.pmf(x, mu1, mu2)

    def _cdf(self, x):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.cdf(x, mu1, mu2)

    def _ppf(self, p):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.ppf(p, mu1, mu2)

    def _mean(self):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.mean(mu1, mu2)

    def _var(self):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.var(mu1, mu2)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for Skellam."""
        return {"mu1": 3.0, "mu2": 2.0}
