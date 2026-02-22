# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Gumbel Right probability distribution."""

from scipy.stats import gumbel_r, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter

class GumbelR(_ScipyAdapter):
    r"""Gumbel Right probability distribution.

    The Gumbel Right distribution is a continuous probability distribution with two
    parameters: location parameter $\mu$ and scale parameter $\sigma > 0$.
    Its probability density function (PDF) is:

    .. math::
        f(x; \mu, \sigma) = \frac{1}{\sigma} \exp\left(-\frac{x - \mu}{\sigma} - \exp\left(-\frac{x - \mu}{\sigma}\right)\right)

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        Location parameter
    sigma : float or array of float (1D or 2D), must be positive
        Scale parameter
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    
    Example
    -------
    >>> from skpro.distributions import GumbelR
    >>> gumbel_r_dist = GumbelR(mu=0.0, sigma=1.0)
    >>> gumbel_r_dist.mean()
    np.float64(0.5772156649015329)
    >>> gumbel_r_dist.var()
    np.float64(1.6449340668482264)
    """

    _tags = {
        "authors": ["an1k3sh"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu=0.0, sigma=1.0, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return gumbel_r

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]
        return [], {"loc": mu, "scale": sigma}
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        import pandas as pd

        # array case examples
        params1 = {"mu": [0.0, 1.0, 2.0], "sigma": [1.0, 1.5, 2.0]}
        params2 = {
            "mu": 2.0,
            "sigma": 0.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        # scalar case examples
        params3 = {"mu": 0.0, "sigma": 1.0}

        return [params1, params2, params3]
    