# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Multivariate Normal probability distribution."""

__author__ = ["HarshvirSandhu"]

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class MultiVariate_Normal(_ScipyAdapter):
    r"""Multivariate Normal distribution, aka log-logistic distribution.

    Most methods wrap ``scipy.stats.multivariate_normal``.

    The mean keyword specifies the mean and cov specifies the covariance matrix.
    The probability distribution function (PDF) is given by:

    .. math::

        f(x) = \frac{1}{\sqrt{(2 \pi)^k \det \Sigma}}
               \exp\left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right),

    where :math:`\mu` is the mean, :math:`\Sigma` the covariance matrix,
    :math:`k` the rank of :math:`\Sigma`.

    Parameters
    ----------
    mean : array_like, default: ``[0]``
        Mean of the distribution.
    cov : array_like or `Covariance`, default: ``[1]``
        Symmetric positive (semi)definite covariance matrix of the distribution.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.multivariatenormal import MultiVariate_Normal

    >>> d = MultiVariate_Normal(mean=[1, 2, 3, 4, 5], cov=3)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": [
            "mean",
            "var",
            "pdf",
            "log_pdf",
            "cdf",
        ],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mean, cov, index=None, columns=None):
        self.mean = mean
        self.cov = cov

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return multivariate_normal

    def _get_scipy_param(self):
        mean = self._bc_params["mean"].ravel()
        cov = np.diag(self._bc_params["cov"].ravel())

        return [], {"cov": cov, "mean": mean}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mean": [1, 2, 3, 4, 5], "cov": 3}
        params2 = {
            "mean": 2,
            "cov": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mean": 1.5, "cov": 2.1}

        return [params1, params2, params3]
