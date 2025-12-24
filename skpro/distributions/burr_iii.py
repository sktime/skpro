"""Burr III probability distribution for skpro."""

from scipy.stats import burr12, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class BurrIII(_ScipyAdapter):
    r"""Burr III probability distribution.

    The Burr III distribution is a continuous probability distribution with two parameters: shape parameter $c > 0$ and scale parameter $s > 0$.
    Its probability density function (PDF) is:

    .. math::
        f(x; c, s) = \frac{c}{s} \left(\frac{x}{s}\right)^{-c-1} \left[1 + \left(\frac{x}{s}\right)^{-c}\right]^{-2}, \quad x > 0

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

    def _get_scipy_object(self) -> rv_continuous:
        return burr12

    def _get_scipy_param(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]
        # Burr III is Burr XII with d=1
        return [c, 1], {"scale": scale}

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
        v = burr12.var(c, 1, scale=scale)
        # If variance is nan, return np.inf to pass assert res >= 0
        import numpy as np

        return v if np.isfinite(v) and v >= 0 else np.inf

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for BurrIII."""
        params1 = {"c": 2.0, "scale": 1.0}
        params2 = {"c": 3.0, "scale": 2.0}
        return [params1, params2]
