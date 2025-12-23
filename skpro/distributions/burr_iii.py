"""Burr III probability distribution for skpro."""

from scipy.stats import burr12

from skpro.distributions.base import BaseDistribution


class BurrIII(BaseDistribution):
    """Burr III probability distribution.

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
        return burr12.pdf(x, c, 1, scale=scale)

    def _cdf(self, x):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return burr12.cdf(x, c, 1, scale=scale)

    def _ppf(self, p):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return burr12.ppf(p, c, 1, scale=scale)

    def _mean(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return burr12.mean(c, 1, scale=scale)

    def _var(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        return burr12.var(c, 1, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for BurrIII."""
        return {"c": 2.0, "scale": 1.0}
