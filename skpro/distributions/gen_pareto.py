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
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1.0, loc=0.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        self.loc = loc
        # Ensure public attributes for sklearn compatibility
        self.__dict__["scale"] = scale
        self.__dict__["loc"] = loc
        super().__init__(index=index, columns=columns)

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

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
        import numpy as np
        v = genpareto.var(c, loc=loc, scale=scale)
        return v if np.isfinite(v) and v >= 0 else np.inf

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for GeneralizedPareto."""
        params1 = {"c": 0.5, "scale": 1.0, "loc": 0.0}
        params2 = {"c": 1.0, "scale": 2.0, "loc": 1.0}
        return [params1, params2]
