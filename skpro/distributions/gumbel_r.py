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
        f(x; \mu, \sigma) =
            \frac{1}{\sigma}
            \exp\left(
                -\frac{x - \mu}{\sigma}
                - \exp\left(-\frac{x - \mu}{\sigma}\right)
            \right)

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        Location parameter
    sigma : float or array of float (1D or 2D), must be positive
        Scale parameter
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
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

    _formula_docs = {
        "pdf": r"""
    The probability density function is given by:

    .. math::
        f(x) = \frac{1}{\sigma} \exp\left(-\frac{x - \mu}{\sigma}\right)
        \exp\left(-\exp\left(-\frac{x - \mu}{\sigma}\right)\right)

    See Johnson, Kotz & Balakrishnan, *Continuous Univariate Distributions*,
    Vol. 1, Chapter 22 (extreme value distribution of type I).
    """,
        #
        "log_pdf": r"""
    The log-density is given by:

    .. math::
        \log f(x) = -\log(\sigma) - \frac{x - \mu}{\sigma}
        - \exp\left(-\frac{x - \mu}{\sigma}\right)
    """,
        #
        "cdf": r"""
    The cumulative distribution function is:

    .. math::
        F(x) = \exp\left(-\exp\left(-\frac{x - \mu}{\sigma}\right)\right)
    """,
        #
        "ppf": r"""
    The quantile function (inverse cdf), for probability :math:`p \in (0, 1)`, is:

    .. math::
        F^{-1}(p) = \mu - \sigma \log\left(-\log(p)\right)
    """,
        #
        "mean": r"""
    The expected value is:

    .. math::
        \mathbb{E}[X] = \mu + \gamma \sigma

    where :math:`\gamma` denotes the Euler-Mascheroni constant.
    """,
        #
        "var": r"""
    The variance is:

    .. math::
        \operatorname{Var}[X] = \frac{\pi^2}{6} \sigma^2
    """,
        #
        "energy": r"""
    The self-energy is:

    .. math::
        \mathbb{E}[|X - Y|] = 2 \sigma \log(2)

    where :math:`X, Y` are independent random variables with the same
    distribution. This follows because the difference :math:`X - Y` of two iid
    Gumbel variables follows a logistic distribution with scale :math:`\sigma`,
    whose mean absolute value is :math:`2 \sigma \log(2)`.
    """,
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
