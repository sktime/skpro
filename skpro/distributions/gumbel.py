# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Gumbel probability distribution."""

import pandas as pd
from scipy.stats import gumbel_l, gumbel_r, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Gumbel(_ScipyAdapter):
    r"""Gumbel distribution, aka extreme value distribution type I.

    Most methods wrap ``scipy.stats.gumbel_r`` or ``scipy.stats.gumbel_l``.

    The right-skewed Gumbel distribution is parametrized by a location parameter
    :math:`\mu` and a scale parameter :math:`\beta`, such that the cumulative
    distribution function (CDF) is given by:

    .. math:: F(x) = \exp(-\exp(-(x - \mu)/\beta))

    The left-skewed Gumbel distribution CDF is given by:

    .. math:: F(x) = 1 - \exp(-\exp((x - \mu)/\beta))

    Parameters
    ----------
    mu : float or array of float (1D or 2D), default=0.0
        location parameter of the distribution
    beta : float or array of float (1D or 2D), must be positive, default=1.0
        scale parameter of the distribution
    skew : str, {"right", "left"}, optional, default="right"
        determines whether to use right-skewed (gumbel_r) or left-skewed (gumbel_l)
        Gumbel distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.gumbel import Gumbel

    >>> d = Gumbel(mu=[[1, 1], [2, 3], [4, 5]], beta=2, skew="right")
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["direkkakkar319"],
        # estimator tags
        # --------------
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu=0.0, beta=1.0, skew="right", index=None, columns=None):
        self.mu = mu
        self.beta = beta
        self.skew = skew

        if skew not in ["right", "left"]:
            raise ValueError('skew parameter must be either "right" or "left".')

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        if self.skew == "right":
            return gumbel_r
        elif self.skew == "left":
            return gumbel_l

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        beta = self._bc_params["beta"]

        return [], {"loc": mu, "scale": beta}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]], "beta": 3, "skew": "right"}
        params2 = {
            "mu": 2,
            "beta": 3,
            "skew": "left",
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 1.5, "beta": 2.1}

        return [params1, params2, params3]
