# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Gompertz probability distribution."""

__author__ = ["abdulsobur"]

import pandas as pd
from scipy.stats import gompertz, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Gompertz(_ScipyAdapter):
    r"""Gompertz distribution.

    Most methods wrap ``scipy.stats.gompertz``.

    The Gompertz distribution is parametrized by shape parameter :math:`c`
    and scale parameter :math:`b`, such that the pdf in standardized form is

    .. math:: f(x, c) = c \exp(x) \exp(-c (e^x - 1))

    for :math:`x \ge 0` and :math:`c > 0`.

    Parameters
    ----------
    c : float or array of float (1D or 2D)
        shape parameter of the distribution, must be positive
    scale : float or array of float (1D or 2D), default=1
        scale parameter of the distribution, must be positive
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.gompertz import Gompertz

    >>> d = Gompertz(c=[[1, 1], [2, 3], [4, 5]], scale=2)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, c, scale=1, index=None, columns=None):
        self.c = c
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return gompertz

    def _get_scipy_param(self):
        c = self._bc_params["c"]
        scale = self._bc_params["scale"]

        return [], {"c": c, "scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"c": [[1, 1], [2, 3], [4, 5]], "scale": 2}
        params2 = {
            "c": 1.5,
            "scale": 2.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"c": 2.1, "scale": 1.2}

        return [params1, params2, params3]
