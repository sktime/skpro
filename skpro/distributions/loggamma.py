# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-Gamma probability distribution."""

__author__ = ["ali-john"]

import pandas as pd
from scipy.stats import loggamma, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class LogGamma(_ScipyAdapter):
    r"""Log-Gamma Distribution.

    Most methods wrap ``scipy.stats.loggamma``.

    The Log-Gamma distribution is a continuous probability distribution
    whose logarithm is related to the gamma distribution. It is useful
    in extreme value theory and reliability analysis.

    The Log-Gamma distribution is parameterized by the shape parameter
    :math:`c`, such that the pdf is

    .. math:: f(x) = \frac{\exp(cx - \exp(x))}{\Gamma(c)}

    where :math:`\Gamma(c)` is the Gamma function.

    The shape parameter :math:`c` is represented by the parameter ``c``.

    Parameters
    ----------
    c : float or array of float (1D or 2D), must be positive
        shape parameter of the log-gamma distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.loggamma import LogGamma

    >>> d = LogGamma(c=[[1, 2], [3, 4], [5, 6]])
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, c, index=None, columns=None):
        self.c = c

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return loggamma

    def _get_scipy_param(self):
        c = self._bc_params["c"]

        return [c], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"c": [[1, 2], [3, 4]]}
        params2 = {
            "c": 2,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"c": 1.5}

        return [params1, params2, params3]
