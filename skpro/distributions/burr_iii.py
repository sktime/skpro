"""Burr III probability distribution for skpro."""

from scipy.stats import burr, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class BurrIII(_ScipyAdapter):
    r"""Burr III probability distribution.

    The Burr III distribution is a continuous probability distribution with two
    shape parameters $c > 0$ and $d > 0$ and a scale parameter $s > 0$.
    Its probability density function (PDF) is:

    .. math::
        f(x; c, d, s) = \frac{c d}{s} \left(\frac{x}{s}\right)^{-c-1}
        \left[1 + \left(\frac{x}{s}\right)^{-c}\right]^{-d-1}, \quad x > 0

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
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c, d=1.0, scale=1.0, index=None, columns=None):
        self.c = c
        self.d = d
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return burr

    def _get_scipy_param(self):
        c = self._bc_params["c"]
        d = self._bc_params["d"]
        scale = self._bc_params["scale"]
        return [], {"c": c, "d": d, "scale": scale}

    def _var(self):
        """Return the variance of the Burr III distribution.

        Variance is infinite for c <= 2 (second moment diverges), else finite.
        Note: Returns inf for c=2 (mathematically correct) when scipy returns nan.
        """
        import numpy as np

        c = self.c if hasattr(self, "c") else self._bc_params["c"]
        if c <= 2:
            return np.inf
        return super()._var()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for BurrIII."""
        params1 = {"c": 2.0, "d": 3.0, "scale": 1.0}
        params2 = {"c": 3.0, "d": 2.0, "scale": 2.0}
        return [params1, params2]
