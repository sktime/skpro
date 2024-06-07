# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Normal probability distribution."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from scipy.stats import halfnorm, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class HalfNormal(_ScipyAdapter):
    r"""Half-Normal distribution.

    Most methods wrap ``scipy.stats.halfnorm``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The distribution is `cut off` at :math:`\( x = 0 \)`. There is no mass assigned to
    negative values; they are entirely excluded from the distribution.

    The half-normal distribution is parametrized by the standard deviation
    :math:`\sigma`, such that the pdf is

    .. math:: f(x) = \frac{\sqrt{2}}{\sigma \sqrt{\pi}}
                    \exp\left(-\frac{x^2}{2\sigma^2}\right), x>0 otherwise 0

    The standard deviation :math:`\sigma` is represented by the parameter ``sigma``.

    Parameters
    ----------
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the half-normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.halfnormal import HalfNormal

    >>> hn = HalfNormal(sigma=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, sigma, index=None, columns=None):
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return halfnorm

    def _get_scipy_param(self):
        sigma = self._bc_params["sigma"]
        return [sigma], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"sigma": [[1, 2], [3, 4]]}
        params2 = {
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"sigma": 2}
        return [params1, params2, params3]
