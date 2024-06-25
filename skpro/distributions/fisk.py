# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-logistic aka Fisk probability distribution."""

__author__ = ["fkiraly", "malikrafsan"]

import pandas as pd
from scipy.stats import fisk, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Fisk(_ScipyAdapter):
    r"""Fisk distribution, aka log-logistic distribution.

    Most methods wrap ``scipy.stats.fisk``.

    The Fisk distribution is parametrized by a scale parameter :math:`\alpha`
    and a shape parameter :math:`\beta`, such that the cumulative distribution
    function (CDF) is given by:

    .. math:: F(x) = 1 - \left(1 + \frac{x}{\alpha}\right)^{-\beta}\right)^{-1}

    Parameters
    ----------
    alpha : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    beta : float or array of float (1D or 2D), must be positive
        shape parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from skpro.distributions.fisk import Fisk

    >>> d = Fisk(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha=1, beta=1, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return fisk

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        return [], {"c": beta, "scale": alpha}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"alpha": [[1, 1], [2, 3], [4, 5]], "beta": 3}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
