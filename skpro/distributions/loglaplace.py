# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-Laplace probability distribution."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from scipy.stats import loglaplace, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class LogLaplace(_ScipyAdapter):
    r"""Log-Laplace distribution.

    Most methods wrap ``scipy.stats.loglaplace``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The log-Laplace distribution is a continuous probability distribution obtained by
    taking the logarithm of the Laplace distribution, commonly used in finance and
    hydrology due to its heavy tails and asymmetry.

    The log-Laplace distribution is parametrized by the scale parameter
    :math:`\c`, such that the pdf is

    .. math:: f(x) = \frac{c}{2} x^{c-1}, \quad 0<x<1

    and

    .. math:: f(x) = \frac{c}{2} x^{-c-1}, \quad x >= 1

    The scale parameter :math:`c` is represented by the parameter ``c``.

    Parameters
    ----------
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the log-Laplace distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.loglaplace import LogLaplace

    >>> ll = LogLaplace(scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, scale, index=None, columns=None):
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return loglaplace

    def _get_scipy_param(self):
        scale = self._bc_params["scale"]
        return [scale], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"scale": [[1, 2], [3, 4]]}
        params2 = {
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"scale": 2}
        return [params1, params2, params3]
