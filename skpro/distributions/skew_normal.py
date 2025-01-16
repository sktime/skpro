# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

"""Skew-Normal probability distribution."""

__author__ = ["Spinachboul"]

from scipy.stats import skewnorm

from skpro.distributions.adapters import ScipyAdapter


class SkewNormal(ScipyAdapter):
    """Skew-Normal Probability Distribution.

    The skew-normal distribution generalizes the normal distribution by introducing
    a shape parameter :math:`\\alpha` to control skewness. It is parameterized by
    ``xi``, ``scale``, and ``shape``:

    .. math:: f(x; \alpha, \\mu, \\sigma) = \frac{2}{\\sigma} \\phi\\left(\frac{x - \\mu}{\\sigma}\right)
              \\Phi\\left(\alpha \frac{x - \\mu}{\\sigma}\right)

    where :math:`\\phi` and :math:`\\Phi` are the pdf and cdf of the standard normal
    distribution, respectively, :math:`\\mu` is the location parameter,
    :math:`\\sigma` is the scale parameter (must be positive), and
    :math:`\alpha` is the shape parameter controlling skewness.

    Parameters
    ----------
    xi : float
        Location parameter of the distribution (mean shift).
    scale : float
        Scale parameter of the distribution (standard deviation), must be positive.
    shape : float
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
    >>> s = SkewNormal(xi=0, scale=1, shape=3)
    >>> s.mean()  # Expected output: Mean of the skew-normal distribution
    >>> s.var()   # Expected output: Variance of the skew-normal distribution
    >>> s.pdf(0)  # Probability density function at x=0
    """

    _tags = {
        "authors": ["Spinachboul"],
        "maintainers": [],
        "python_version": ">=3.8",
        "python_dependencies": ["scipy"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, xi, scale, shape, index=None, columns=None):
        self.xi = xi
        self.scale = scale
        self.shape = shape
        self.index = index
        self.columns = columns

        self.params = {"a": self.shape, "loc": self.xi, "scale": self.scale}

        super().__init__(
            scipy_class=skewnorm, index=self.index, columns=self.columns, **self.params
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        if parameter_set == "default":
            return {"xi": 0, "scale": 1, "shape": 5}
        return [
            {"xi": -1, "scale": 2, "shape": 3},
            {"xi": 0, "scale": 1, "shape": -2},
        ]
