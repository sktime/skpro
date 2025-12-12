# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Geometric probability distribution."""

__author__ = ["aryabhatta-dey"]

import pandas as pd
from scipy.stats import geom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Geometric(_ScipyAdapter):
    r"""Geometric Distribution.

    Most methods wrap ``scipy.stats.geom``.

    The Geometric distribution is parameterized by the probability of
    success :math:`p` in a given trial
    such that the probability mass function (PMF) is given by:

    .. math:: P(X = k) = p(1 - p)^{k - 1} \quad \text{where} \quad k = 1, 2, 3, \ldots

    Parameters
    ----------
    p : float or array of float (1D or 2D), must be in (0, 1]
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.geometric import Geometric
    >>> d = Geometric(p=0.5)
    """

    _tags = {
        "capabilities:approx": ["pmf"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, p, index=None, columns=None):
        self.p = p

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return geom

    def _get_scipy_param(self):
        p = self._bc_params["p"]

        return [], {"p": p}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"p": [0.2, 0.5, 0.8]}
        params2 = {
            "p": 0.4,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        # scalar case examples
        params3 = {"p": 0.7}

        return [params1, params2, params3]
