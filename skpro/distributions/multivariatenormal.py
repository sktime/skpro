# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Multivariate Normal probability distribution."""

__author__ = ["HarshvirSandhu"]

import pandas as pd
from scipy.stats import multivariate_normal

from skpro.distributions.base import BaseDistribution


class MultiVariate_Normal(BaseDistribution):
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
    mean : array_like, default = ``[0]``
        Mean of the distribution.
    cov : array_like or `Covariance`, default = ``[1]``
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

    def __init__(self, mean=None, cov=None, index=None, columns=None):
        if mean is None:
            mean = [0]
        if cov is None:
            cov = [1]
        self.mean = mean
        self.cov = cov
        self.scipy_obj = multivariate_normal(mean=self.mean, cov=self.cov)

        super().__init__(index=index, columns=columns)

    def pdf(self, x):
        """Probability density function."""
        return self.scipy_obj.pdf(x)

    def logpdf(self, x):
        """Logarithmic probability density function."""
        return self.scipy_obj.logpdf(x)

    def cdf(self, x):
        """Cumulative distribution function."""
        return self.scipy_obj.cdf(x)

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
