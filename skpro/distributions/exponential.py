# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Exponential probability distribution."""

__author__ = ["ShreeshaM07"]

import pandas as pd
from scipy.stats import expon, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Exponential(_ScipyAdapter):
    r"""Exponential Distribution.

    Most methods wrap ``scipy.stats.expon``.

    The Exponential distribution is parametrized by mean :math:`\mu` and
    scale :math:`b`, such that the pdf is

    .. math:: f(x) = \lambda*\exp\left(-\lambda*x\right)

    The rate :math:`\lambda` is represented by the parameter ``rate``,

    Parameter
    ---------
    rate : float or array of float (1D or 2D)
        rate of the distribution
        rate = 1/scale
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "capabilities:approx": ["ppf", "energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "continuous",
        "broadcast_init": "on",
    }

    def __init__(self, rate, index=None, columns=None):
        self.rate = rate

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return expon

    def _get_scipy_param(self):
        rate = self._bc_params["rate"]
        scale = 1 / rate
        return [], {"scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the distribution."""
        params1 = {"rate": [1, 2, 2.5, 3.5, 5]}
        params2 = {"rate": 2}
        params3 = {
            "rate": [
                [2, 2, 2],
                [4, 4, 4],
            ],
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b", "c"]),
        }

        return [params1, params2, params3]
