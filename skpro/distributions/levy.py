"""Levy probability distribution for skpro."""

from scipy.stats import levy

from skpro.distributions.base import BaseDistribution


class Levy(BaseDistribution):
    r"""Levy probability distribution.

    The Levy distribution is parametrized by location :math:`\mu` and
    scale :math:`c`, such that the pdf is

    .. math:: f(x) = \sqrt{\frac{c}{2\pi}} \frac{e^{-c/(2(x-\mu))}}{(x-\mu)^{3/2}}

    The location :math:`\mu` is represented by the parameter ``mu``,
    and the scale :math:`c` by the parameter ``scale``.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        Location parameter of the distribution.
    scale : float or array of float (1D or 2D), must be positive
        Scale parameter of the distribution.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.levy import Levy
    >>> dist = Levy(mu=0.0, scale=1.0)
    >>> dist.mean()
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu=0.0, scale=1.0, index=None, columns=None):
        self.mu = mu
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _pdf(self, x):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return levy.pdf(x, loc=mu, scale=scale)

    def _cdf(self, x):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return levy.cdf(x, loc=mu, scale=scale)

    def _ppf(self, p):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return levy.ppf(p, loc=mu, scale=scale)

    def _mean(self):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return levy.mean(loc=mu, scale=scale)

    def _var(self):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return levy.var(loc=mu, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": 0.0, "scale": 1.0}
        params2 = {"mu": 1.0, "scale": 2.0}
        return [params1, params2]
