"""Generalized Pareto probability distribution for skpro."""

from scipy.stats import genpareto

from skpro.distributions.base import BaseDistribution


class GeneralizedPareto(BaseDistribution):
    """Generalized Pareto probability distribution.

    Parameters
    ----------
    c : float
        Shape parameter
    scale : float
        Scale parameter
    loc : float
        Location parameter
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, mu=0.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        self.mu = mu
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        mu = self._bc_params["mu"]
        return genpareto.pdf(x, c, loc=mu, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        mu = self._bc_params["mu"]
        return genpareto.cdf(x, c, loc=mu, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        mu = self._bc_params["mu"]
        return genpareto.ppf(p, c, loc=mu, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        mu = self._bc_params["mu"]
        return genpareto.mean(c, loc=mu, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        mu = self._bc_params["mu"]
        import numpy as np

        v = genpareto.var(c, loc=mu, scale=scale)
        return v if np.isfinite(v) and v >= 0 else np.inf

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for GeneralizedPareto."""
        params1 = {"c": 0.5, "scale": 1.0, "mu": 0.0}
        params2 = {"c": 1.0, "scale": 2.0, "mu": 1.0}
        return [params1, params2]
