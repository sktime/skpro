"""Generalized Pareto probability distribution for skpro."""

from skpro.distributions.base import BaseDistribution
from scipy.stats import genpareto


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
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, loc=0.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        self.loc = loc
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.pdf(x, c, loc=loc, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.cdf(x, c, loc=loc, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.ppf(p, c, loc=loc, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.mean(c, loc=loc, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        loc = self._bc_params["loc"]
        return genpareto.var(c, loc=loc, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for GeneralizedPareto."""
        return {"c": 0.5, "scale": 1.0, "loc": 0.0}
