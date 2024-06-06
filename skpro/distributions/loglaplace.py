# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-Laplace probability distribution."""

__author__ = ["SaiRevanth25"]

import pandas as pd
from scipy.stats import loglaplace, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class LogLaplace(_ScipyAdapter):
    r"""Log-Laplace distribution.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The log-Laplace distribution is a continuous probability distribution obtained by
    taking the logarithm of the Laplace distribution, commonly used in finance and
    hydrology due to its heavy tails and asymmetry.

    The log-Laplace distribution is parametrized by the shape parameter
    :math:`\shape`, such that the pdf is

    .. math:: f(x) = \frac{c}{2} x^{c-1}, 0<x<1
    and
    .. math:: f(x) = \frac{c}{2} x^{-c-1}, x >= 1

    The shape parameter :math:`shape` is represented by the parameter ``c``

    Parameters
    ----------
    shape : float or array of float (1D or 2D), must be positive
        scale parameter of the log-Laplace distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.loglaplace import LogLaplace

    >>> ll = LogLaplace(shape=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, shape, index=None, columns=None):
        self.shape = shape

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return loglaplace

    def _get_scipy_param(self):
        shape = self._bc_params["shape"]
        return [shape], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"shape": [[1, 2], [3, 4]]}
        params2 = {
            "shape": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"shape": 2}
        return [params1, params2, params3]
