# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

"""Skew-Normal probability distribution."""

__author__ = ["Spinachboul"]

from scipy.stats import skewnorm

from skpro.distributions.adapters.scipy._distribution import _ScipyAdapter


class SkewNormal(_ScipyAdapter):
    r"""Skew-Normal Probability Distribution.

    The skew-normal distribution generalizes the normal distribution by introducing
    a shape parameter :math:`\\alpha` to control skewness. It is parameterized by
    ``mu``, ``sigma``, and ``alpha``:

    .. math:: f(x; \alpha, \\mu, \\sigma) = \frac{2}{\\sigma}
          \\phi\\left(\frac{x - \\mu}{\\sigma}\right)
          \\Phi\\left(\alpha \frac{x - \\mu}{\\sigma}\right)

    where :math:`\\phi` and :math:`\\Phi` are the pdf and cdf of the standard normal
    distribution, respectively, :math:`\\mu` is the location parameter,
    :math:`\\sigma` is the scale parameter (must be positive), and
    :math:`\alpha` is the shape parameter controlling skewness.

    Parameters
    ----------
    mu : float
        Location parameter of the distribution (mean shift).
    sigma : float
        Scale parameter of the distribution (standard deviation), must be positive.
    alpha : float
        Skewness parameter of the distribution:
        - A positive value skews to the right.
        - A negative value skews to the left.
        - A value of zero results in a standard normal distribution.
    index : array-like, optional, default=None
        Index for the distribution, providing pandas-like indexing for rows.
    columns : array-like, optional, default=None
        Columns for the distribution, providing pandas-like indexing for columns.

    Examples
    --------
    >>> from skpro.distributions import SkewNormal
    >>> s = SkewNormal(mu=0, sigma=1, alpha=3)
    """

    _tags = {
        "authors": ["Spinachboul"],
        "python_dependencies": ["scipy"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, alpha=0, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self):
        return skewnorm

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]
        alpha = self._bc_params["alpha"]

        return [], {
            "loc": mu,
            "scale": sigma,
            "a": alpha,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        r"""Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class. Each dict represents
            parameters to construct a test instance:
            - `MyClass(**params)` or `MyClass(**params[i])` creates a valid test.
            - `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return [
            {"mu": -1, "sigma": 2},
            {"mu": 0, "sigma": 1, "alpha": -2},
        ]
