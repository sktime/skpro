# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Alpha probability distribution."""

__author__ = ["SaiReavanth25"]

import pandas as pd
from scipy.stats import alpha, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Alpha(_ScipyAdapter):
    r"""Alpha distribution.

    Most methods wrap ``scipy.stats.alpha``.

    The alpha distribution is characterized by its shape parameter :math:`\a`,
    which determines its skewness and tail behavior.
    It is often used for modeling data with heavy right tails,
    unlike the Gaussian distribution(which is symmetric and bell-shaped).

    The probability density function (PDF) of the Alpha distribution is given by:
    .. math::

        f(x) = \frac{1}{x^2 \Phi(a) \sqrt{2\pi}}
                \exp\left(-\frac{1}{2}\left(\frac{a - 1}{x}\right)^2\right)

    where:
       - :math:`a` is the shape parameter.
       - :math:`Phi` is the cumulative distribution function (CDF) of the
                    standard normal distribution.

    Parameters
    ----------
    a : float or array of float (1D or 2D), must be positive
        Shape parameter controlling skewness and tail behavior.
        Higher values result in heavier tails and greater skewness towards the right.
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

    def _get_scipy_param(self):
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
