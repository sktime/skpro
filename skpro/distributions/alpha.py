# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Alpha probability distribution."""

__author__ = ["SaiReavanth25"]

import pandas as pd
from scipy.stats import alpha, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Alpha(_ScipyAdapter):
    """Alpha distribution.

    Parameters
    ----------
    a : float or array of float (1D or 2D), must be positive
        shape parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions import Alpha

    >>> distr = Alpha(a=[[1, 2], [3, 4]])
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, a, index=None, columns=None):
        self.a = a

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return alpha

    def _get_scipy_param(self) -> dict:
        a = self._bc_params["a"]

        return [a], {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"a": [[2, 3], [4, 5]]}
        params2 = {
            "a": 3,
            "index": pd.Index([1, 2, 3]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"a": 2.5}

        return [params1, params2, params3]
