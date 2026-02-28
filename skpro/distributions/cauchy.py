# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Cauchy probability distribution."""

__author__ = ["patelchaitany"]

import pandas as pd
from scipy.stats import cauchy, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Cauchy(_ScipyAdapter):
    r"""Cauchy distribution.

    Most methods wrap ``scipy.stats.cauchy``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The Cauchy distribution is a continuous probability distribution commonly
    used in robust statistics and as a prior in Bayesian inference. It has
    heavy tails and no finite moments (mean and variance are undefined).

    The Cauchy distribution is parametrized by location :math:`x_0` and
    scale :math:`\gamma`, such that the pdf is

    .. math::

        f(x) = \frac{1}{\pi \gamma \left[1 +
                \left(\frac{x - x_0}{\gamma}\right)^2\right]}

    The location parameter :math:`x_0` is represented by ``mu``,
    and the scale parameter :math:`\gamma` by ``scale``.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        location parameter of the Cauchy distribution
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the Cauchy distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.cauchy import Cauchy

    >>> c = Cauchy(mu=0, scale=1)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return cauchy

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return [], {"loc": mu, "scale": scale}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[0, 1], [2, 3]], "scale": [[1, 2], [3, 4]]}
        params2 = {
            "mu": 0,
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        params3 = {"mu": 0, "scale": 2}
        return [params1, params2, params3]
