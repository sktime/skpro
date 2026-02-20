"""Burr III probability distribution for skpro."""

from scipy.stats import burr, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class BurrIII(_ScipyAdapter):
    r"""Burr III probability distribution.

    The Burr III distribution is a continuous probability distribution with two
    shape parameters $c > 0$, $d > 0$ and scale parameter $s > 0$.
    Its probability density function (PDF) is:

    .. math::
        f(x; c, d, s) = \frac{cd}{s} \left(\frac{x}{s}\right)^{-c-1}
        \left[1 + \left(\frac{x}{s}\right)^{-c}\right]^{-d-1}, \quad x > 0

    The mean exists for $c > 1$, and the variance exists for $c > 2$.

    Parameters
    ----------
    c : float
        Shape parameter
    d : float
        Shape parameter
    scale : float
        Scale parameter
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "authors": ["arnavk23", "Joiejoie1"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf", "energy"],
        "broadcast_init": "on",
    }

    def __init__(self, c, d, scale=1.0, index=None, columns=None):
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

    def _mean(self):
        """Return the mean of the Burr III distribution.

        Mean is infinite for c <= 1 (first moment diverges), else finite.
        Note: Returns inf for c=1 (mathematically correct) when scipy returns nan.
        """
        import numpy as np

        c = self._bc_params["c"]
        if np.any(c <= 1):
            mean = super()._mean()
            return np.where(c <= 1, np.inf, mean)
        return super()._mean()

    def _var(self):
        """Return the variance of the Burr III distribution.

        Variance is infinite for c <= 2 (second moment diverges), else finite.
        Note: Returns inf for c=2 (mathematically correct) when scipy returns nan.
        """
        import numpy as np

        c = self._bc_params["c"]
        if np.any(c <= 2):
            var = super()._var()
            return np.where(c <= 2, np.inf, var)
        return super()._var()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for BurrIII."""
        params1 = {"c": 3.0, "d": 1.0, "scale": 1.0}
        params2 = {"c": 4.0, "d": 2.0, "scale": 2.0}
        return [params1, params2]
