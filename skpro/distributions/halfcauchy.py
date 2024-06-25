# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Cauchy probability distribution."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from scipy.stats import halfcauchy, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class HalfCauchy(_ScipyAdapter):
    r"""Half-Cauchy distribution.

    Most methods wrap ``scipy.stats.halfcauchy``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The half-Cauchy distribution is a continuous probability distribution that
    is the positive half of the Cauchy distribution. It is commonly used in
    Bayesian statistics, especially as a prior distribution for scale parameters
    due to its heavy tails and non-negativity.

    The half-Cauchy distribution is parametrized by the scale parameter
    :math:`\beta`, such that the pdf is

    .. math::

        f(x) = \frac{2}{\pi \beta \left(1 + \left(\frac{x}{\beta}\right)^2\right)},
                x>0 otherwise 0

    The scale parameter :math:`\beta` is represented by the parameter ``beta``.

    Parameters
    ----------
    beta : float or array of float (1D or 2D), must be positive
        scale parameter of the half-Cauchy distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.halfcauchy import HalfCauchy

    >>> hc = HalfCauchy(beta=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, beta, index=None, columns=None):
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return halfcauchy

    def _get_scipy_param(self):
        beta = self._bc_params["beta"]
        return [beta], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"beta": [[1, 2], [3, 4]]}
        params2 = {
            "beta": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"beta": 2}
        return [params1, params2, params3]
