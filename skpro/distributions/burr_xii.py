"""Burr XII probability distribution for skpro."""

from scipy.stats import burr12

from skpro.distributions.base import BaseDistribution


class BurrXII(BaseDistribution):
    """Burr XII probability distribution.

    Parameters
    ----------
    c : float
        Shape parameter
    d : float
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

    def __init__(self, c, d, scale=1.0, index=None, columns=None):
        self.c = c
        self.d = d
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return burr12.pdf(x, c, d, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return burr12.cdf(x, c, d, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return burr12.ppf(p, c, d, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return burr12.mean(c, d, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return burr12.var(c, d, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for BurrXII."""
        params1 = {"c": 2.0, "d": 3.0, "scale": 1.0}
        params2 = {"c": 4.0, "d": 2.0, "scale": 2.0}
        return [params1, params2]
