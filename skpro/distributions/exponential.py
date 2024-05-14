# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Exponential probability distribution."""

__author__ = ["ShreeshaM07"]

import pandas as pd
from scipy.stats import expon, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Exponential(_ScipyAdapter):
    r"""Exponential Distribution.

    .. math::

        f(x) = \exp(-x)

    for :math:`x \ge 0`

    Parameter
    ---------
    mu : float or array of float (1D or 2D)
        mean of the distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution, same as standard deviation / sqrt(2)
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex
    """

    _tags = {
        "capabilities:approx": ["ppf", "energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "discrete",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return expon

    def _get_scipy_param(self) -> dict:
        loc = self._bc_params["loc"]
        scale = self._bc_params["scale"]

        return [], {"mu": loc, "scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the distribution."""
        params1 = {"mu": [1, 2, 3, 4.2, 5], "scale": [1, 2, 2.5, 3.5, 5]}
        params2 = {
            "mu": 5,
            "scale": 2,
        }
        params3 = {
            "mu": [
                [1, 2, 3],
                [3, 4, 5],
            ],
            "scale": [
                [2, 2, 2],
                [4, 4, 4],
            ],
            "index": pd.Index([1, 2]),
            "columns": pd.Index(["a", "b", "c"]),
        }

        return [params1, params2, params3]
