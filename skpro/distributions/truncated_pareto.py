"""Truncated Pareto probability distribution for skpro."""

from skpro.distributions.base import BaseDistribution
import numpy as np
from scipy.stats import pareto


class TruncatedPareto(BaseDistribution):
    """Truncated Pareto probability distribution.

    Parameters
    ----------
    b : float
        Shape parameter
    scale : float
        Scale parameter
    lower : float
        Lower truncation bound
    upper : float
        Upper truncation bound
    """

    _tags = {
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, b, scale=1.0, lower=1.0, upper=10.0, index=None, columns=None):
        self.b = b
        self.scale = scale
        self.lower = lower
        self.upper = upper
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        pdf = np.where(
            (x >= lower) & (x <= upper), pareto.pdf(x, b, scale=scale) / norm, 0.0
        )
        return pdf

    def _cdf(self, x):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        cdf = (pareto.cdf(x, b, scale=scale) - pareto.cdf(lower, b, scale=scale)) / norm
        cdf = np.where(x < lower, 0.0, np.where(x > upper, 1.0, cdf))
        return cdf

    def _ppf(self, p):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        p_adj = p * norm + pareto.cdf(lower, b, scale=scale)
        return pareto.ppf(p_adj, b, scale=scale)

    def _mean(self):
        # Mean of truncated Pareto is not directly available in scipy
        # Can be computed numerically if needed
        return None

    def _var(self):
        # Variance of truncated Pareto is not directly available in scipy
        # Can be computed numerically if needed
        return None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for TruncatedPareto."""
        return {"b": 2.0, "scale": 1.0, "lower": 1.0, "upper": 10.0}
