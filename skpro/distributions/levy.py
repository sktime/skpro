"""Levy probability distribution for skpro."""

from scipy.stats import levy

from skpro.distributions.base import BaseDistribution


class Levy(BaseDistribution):
    """Levy probability distribution.

    Parameters
    ----------
    loc : float
        Location parameter
    scale : float
        Scale parameter
    """

    _tags = {
        "authors": ["your-github-id"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, loc=0.0, scale=1.0, index=None, columns=None):
        self.loc = loc
        self.scale = scale
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
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]
        return levy.pdf(x, loc=loc, scale=scale)

    def _cdf(self, x):
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]
        return levy.cdf(x, loc=loc, scale=scale)

    def _ppf(self, p):
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]
        return levy.ppf(p, loc=loc, scale=scale)

    def _mean(self):
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]
        return levy.mean(loc=loc, scale=scale)

    def _var(self):
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]
        return levy.var(loc=loc, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for Levy."""
        params1 = {"loc": 0.0, "scale": 1.0}
        params2 = {"loc": 1.0, "scale": 2.0}
        return [params1, params2]
