# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Logistic probability distribution."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from scipy.stats import halflogistic, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class HalfLogistic(_ScipyAdapter):
    r"""Half-Logistic distribution.

    Most methods wrap ``scipy.stats.halflogistic``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The half-logistic distribution is a continuous probability distribution derived
    from the logistic distribution by taking only the positive half. It is particularly
    useful in reliability analysis, lifetime modeling, and other applications where
    non-negative values are required.

    The half-logistic distribution is parametrized by the scale parameter
    :math:`\beta`, such that the pdf is

    .. math::

        f(x) = \frac{2 \exp\left(-\frac{x}{\beta}\right)}
                {\beta \left(1 + \exp\left(-\frac{x}{\beta}\right)\right)^2},
                x>0 otherwise 0

    The scale parameter :math:`\beta` is represented by the parameter ``beta``.

    Parameters
    ----------
    beta : float or array of float (1D or 2D), must be positive
        scale parameter of the half-logistic distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.halflogistic import HalfLogistic

    >>> hl = HalfLogistic(beta=1)
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
        return halflogistic

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
