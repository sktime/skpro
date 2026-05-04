"""Generalized Pareto probability distribution for skpro."""
from scipy.stats import genpareto

from skpro.distributions.adapters.scipy import _ScipyAdapter


class GeneralizedPareto(_ScipyAdapter):
    r"""Generalized Pareto probability distribution.

    The Generalized Pareto distribution is parametrized by shape :math:`c`,
    scale :math:`s`, and location :math:`\mu`, such that the pdf is

    .. math:: f(x; c, s, \mu) = \frac{1}{s}\left(1 + c\frac{x-\mu}{s}\right)^{-1/c - 1}

    The location :math:`\mu` is represented by the parameter ``mu``,
    the scale :math:`s` by the parameter ``scale``,
    and the shape by the parameter ``c``.

    Parameters
    ----------
    c : float
        Shape parameter
    scale : float
        Scale parameter
    mu : float or array of float (1D or 2D)
        Location parameter
    index: pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.gen_pareto import GeneralizedPareto
    >>> dist = GeneralizedPareto(c=0.5, scale=1.0, mu=0.0)
    >>> mean = dist.mean()
    """

    _tags = {
        "authors": ["arnavk23", "direkkakkar319-ops"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, c=0.5, scale=1.0, mu=0.0, index=None, columns=None):
        self.c = c
        self.scale = scale
        self.mu = mu
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self):
        return genpareto

    def _get_scipy_param(self):
        c = self._bc_params["c"]
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return [c], {"loc": mu, "scale": scale}

    def _var(self):
        import numpy as np

        c = self._bc_params["c"]
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        v = genpareto.var(c, loc=mu, scale=scale)
        return np.where(np.isnan(v), np.inf, v)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for GeneralizedPareto."""
        params1 = {"c": 0.5, "scale": 1.0, "mu": 0.0}
        params2 = {"c": 1.0, "scale": 2.0, "mu": 1.0}
        return [params1, params2]
