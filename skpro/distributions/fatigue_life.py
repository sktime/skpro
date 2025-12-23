"""Fatigue-life (Birnbaum–Saunders) probability distribution for skpro."""

from scipy.stats import fatiguelife

from skpro.distributions.base import BaseDistribution


class FatigueLife(BaseDistribution):
    """Fatigue-life (Birnbaum–Saunders) probability distribution.

    Parameters
    ----------
    c : float
        Shape parameter
    scale : float
        Scale parameter
    """

    _tags = {
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.pdf(x, c, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.cdf(x, c, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.ppf(p, c, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.mean(c, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return fatiguelife.var(c, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for FatigueLife."""
        params1 = {"c": 2.0, "scale": 1.0}
        params2 = {"c": 1.5, "scale": 2.0}
        return [params1, params2]
